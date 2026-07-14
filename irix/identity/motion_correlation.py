"""Motion-correlation identity resolution: which camera-detected person
corresponds to which wristband, for the case ``irix.identity.ble_pairing``
alone can't resolve -- two or more members' bands are all within BLE
range of the *same* station, so RSSI proximity can rank stations for one
member, but can't tell two co-located members' bands apart from each
other.

Grounded in real prior art, not invented from scratch: systems that pair
a camera-tracked person to a specific wearable IMU commonly do it by
cross-correlating a vision-derived motion signal (from a tracked joint's
position over time) against the wearable's own raw accelerometer signal
over the same window -- the candidate pairing with the highest
correlation is assumed to be the same physical person, since two
different people's limb motion is essentially uncorrelated even when
they're doing similar exercises side by side, while a wristband's actual
motion should closely track its wearer's own tracked wrist. See the IEEE
"Person tracking association using multi-modal systems" work (matching a
Kinect-tracked skeleton to an inertial wearable via movement features),
and the broader acceleration-correlation-based identification/
synchronization literature (camera + wearable-accelerometer normalized
cross-correlation, used for both identity assignment and clock
synchronization without any calibration step or auxiliary sensor).

Also independently corroborated on the IMU side alone: Zou, Choi et al.,
"Person Re-Identification Using Deep Modeling of Temporally Correlated
Inertial Motion Patterns" (Sensors, 2020) re-identifies individuals from
wearable accelerometer/gyroscope motion signatures across 86 subjects,
confirming that an IMU's temporal motion pattern carries enough
person-distinguishing signal on its own to be a legitimate re-
identification feature -- the same premise this module leans on for the
IMU half of the cross-modal correlation, arrived at here independently
before that specific citation was found (2026-07-14 competitive/prior-art
review), not built from it.

**Known limitation, stated plainly** (same standard of honesty this repo
applies to every proxy signal -- see e.g. ``irix.rep_counting``'s
angular-velocity-proxy docstring): a wearable's raw accelerometer
measures gravity's changing projection as the wrist rotates, on top of
translational motion; a vision-derived signal built from tracked keypoint
*position* captures only translational motion. That mismatch degrades
the correlation somewhat -- the literature above explicitly calls out
gravity-direction compensation (estimating the wrist's 3D orientation and
subtracting gravity's projection) as the main accuracy lever for a
production system. This module does not attempt that compensation -- it
would need a full 3D wrist-orientation estimate this repo doesn't
otherwise compute -- and instead relies on vertical motion still being
the dominant, well-correlated component for gym exercises specifically
(squats/curls/presses/rows are all fundamentally vertical, cyclic
motions), which is enough to disambiguate a *small* number of candidates
at one station (2-3 co-located members), not to serve as a general-
purpose, camera-angle-agnostic re-identification system.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

from ..fusion.imu import IMUSample
from ..pose.estimator import PersonPose

CONFIDENCE_THRESHOLD = 0.3
MIN_VALID_SAMPLES = 5


@dataclass
class MotionCorrelationMatch:
    person_index: int  # index into the detected_people_poses list passed to resolve()
    member_id: str
    correlation: float  # |Pearson r| between the vision proxy and this member's IMU signal, [0, 1]
    confidence: float  # derived from the margin over the second-best candidate, [0, 1]

    def to_dict(self) -> dict:
        return {
            "person_index": self.person_index,
            "member_id": self.member_id,
            "correlation": self.correlation,
            "confidence": self.confidence,
        }


def _vertical_keypoint_signal(poses: List[PersonPose], keypoint_name: str) -> np.ndarray:
    """Vertical (y) pixel position of one keypoint across a pose
    sequence -- NaN wherever that keypoint wasn't confidently tracked
    that frame, so a gap doesn't get silently treated as "didn't move"."""
    ys = []
    for p in poses:
        kp = p.get(keypoint_name)
        if kp is None or kp.confidence < CONFIDENCE_THRESHOLD:
            ys.append(np.nan)
        else:
            ys.append(kp.y)
    return np.array(ys, dtype=float)


def _derive_acceleration_proxy(y_positions: np.ndarray, dt: float, window_frames: int = 9) -> np.ndarray:
    """A vision-derived acceleration proxy from a tracked keypoint's
    position, in arbitrary (pixels/frame^2-ish) units -- only the
    *shape*/timing of this signal is used (via correlation), never its
    absolute scale, so the arbitrary units are fine.

    Uses a Savitzky-Golay smoothed second derivative rather than naive
    double finite-differencing: numerically differentiating a position
    signal twice amplifies any pixel-tracking jitter by roughly
    ``1/dt**2`` (at 30fps, ~900x), which can swamp real motion for a
    keypoint detector with even a fraction of a pixel of frame-to-frame
    noise. Savitzky-Golay (fit a local polynomial over a short window,
    differentiate the polynomial) is the standard, well-established fix
    for exactly this -- smooths and differentiates in one step rather
    than differentiating raw noise. Falls back to plain double-differencing
    if there isn't enough data for the smoothing window (rare -- only for
    a very short pose sequence)."""
    if len(y_positions) < 3 or dt <= 0:
        return np.array([])
    if np.isnan(y_positions).any():
        # savgol_filter doesn't handle NaN gaps -- interpolate through
        # short occlusion gaps first (matches how irix.rep_counting's NaN
        # guard treats a missed keypoint: skip it, don't fabricate a
        # confident value, but don't let one bad frame break an otherwise
        # analyzable window either).
        idx = np.arange(len(y_positions))
        valid = ~np.isnan(y_positions)
        if valid.sum() < 3:
            return np.array([])
        y_positions = np.interp(idx, idx[valid], y_positions[valid])
    window = min(window_frames, len(y_positions) - (1 - len(y_positions) % 2))
    if window < 5:
        vel = np.diff(y_positions) / dt
        return np.diff(vel) / dt
    if window % 2 == 0:
        window -= 1
    polyorder = min(3, window - 2)
    return savgol_filter(y_positions, window_length=window, polyorder=polyorder, deriv=2, delta=dt)


def _imu_vertical_accel(samples: List[IMUSample]) -> np.ndarray:
    """Vertical-axis acceleration from a wristband IMU stream -- z-axis,
    gravity-sign-flipped to match the "positive = upward effort" convention
    ``irix.fusion.imu_rep_counting`` already uses elsewhere in this repo."""
    return np.array([-s.accel[2] for s in samples], dtype=float)


def _normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """|Pearson correlation| between two equal-length signals, NaN-safe
    (drops indices where either signal is NaN). 0.0 (not NaN) if fewer
    than MIN_VALID_SAMPLES valid overlapping samples remain, or either
    signal is constant (zero variance -- correlation undefined)."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < MIN_VALID_SAMPLES:
        return 0.0
    a_v, b_v = a[mask], b[mask]
    if np.std(a_v) == 0 or np.std(b_v) == 0:
        return 0.0
    return float(abs(np.corrcoef(a_v, b_v)[0, 1]))


class MotionCorrelationResolver:
    """Disambiguates which camera-detected person is which member when
    ``irix.identity.ble_pairing``'s RSSI-based resolution alone can't --
    multiple members' bands all within range of one station. NOT a
    replacement for RSSI-based station resolution; only invoke this when
    ``irix.topology.handoff.GymCoordinator.active_members_at()`` (or
    equivalent) reports more than one candidate member for a station with
    more than one camera-detected person.
    """

    def __init__(self, keypoint_name: str = "left_wrist", min_confidence_margin: float = 0.15):
        self.keypoint_name = keypoint_name
        self.min_confidence_margin = min_confidence_margin

    def resolve(
        self,
        candidate_imu_streams: Dict[str, List[IMUSample]],
        detected_people_poses: List[List[PersonPose]],
        pose_fps: float,
    ) -> List[Optional[MotionCorrelationMatch]]:
        """One result per entry in ``detected_people_poses``, in order.
        ``None`` for a detected person whose best- and second-best-
        correlated candidates are too close to call (``min_confidence_margin``)
        rather than guessing, or for whom no candidate had enough valid
        signal to correlate against at all.

        Each member_id in ``candidate_imu_streams`` may only be assigned
        to one detected person -- once a member is claimed by the
        detected person it correlates best with overall, it's removed
        from consideration for the rest (a simple greedy assignment;
        exact bipartite matching is unnecessary at the 2-3 candidate
        scale this is meant for).
        """
        dt = 1.0 / pose_fps
        vision_signals = [
            _derive_acceleration_proxy(_vertical_keypoint_signal(poses, self.keypoint_name), dt)
            for poses in detected_people_poses
        ]
        imu_signals = {member_id: _imu_vertical_accel(samples) for member_id, samples in candidate_imu_streams.items()}

        # Score every (person, candidate) pair up front, then greedily
        # assign highest-confidence pairs first so an early low-margin
        # tie doesn't starve a later, more confident pairing of its best
        # candidate.
        scored: List[Tuple[float, int, str, float]] = []  # (confidence, person_idx, member_id, correlation)
        for person_idx, vsig in enumerate(vision_signals):
            corrs = []
            for member_id, az in imu_signals.items():
                if len(az) < 2 or len(vsig) < 2:
                    corrs.append((member_id, 0.0))
                    continue
                resampled = np.interp(np.linspace(0, 1, len(vsig)), np.linspace(0, 1, len(az)), az)
                corrs.append((member_id, _normalized_cross_correlation(vsig, resampled)))
            corrs.sort(key=lambda c: -c[1])
            if not corrs:
                continue
            best_member, best_corr = corrs[0]
            second_corr = corrs[1][1] if len(corrs) > 1 else 0.0
            margin = best_corr - max(second_corr, 0.0)
            confidence = min(1.0, margin * 2)
            scored.append((confidence, person_idx, best_member, best_corr))

        scored.sort(key=lambda s: -s[0])
        results: List[Optional[MotionCorrelationMatch]] = [None] * len(detected_people_poses)
        claimed_members = set()
        assigned_people = set()
        for confidence, person_idx, member_id, corr in scored:
            if person_idx in assigned_people or member_id in claimed_members:
                continue
            if confidence < self.min_confidence_margin * 2:
                continue
            results[person_idx] = MotionCorrelationMatch(
                person_index=person_idx, member_id=member_id, correlation=corr, confidence=confidence,
            )
            claimed_members.add(member_id)
            assigned_people.add(person_idx)
        return results
