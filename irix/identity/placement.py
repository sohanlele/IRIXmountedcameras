"""Wristband physical-placement state machine (Section 5.2's wrist/ankle
distinction, made real-time and dynamic).

## Why this exists

``irix.rep_counting.exercises.ExerciseConfig.band_placement`` already
states *which* limb type (wrist vs. ankle) each exercise needs the band
worn on for its IMU signal to mean anything -- but nothing before this
module tracked *where the band is actually worn right now*, or handled
the moment a member (per a future IRIX app instruction, out of scope to
build here -- see the module docstring's product boundary note below)
moves it from one limb to the other between a wrist-group exercise
(squat, hack squat, lunge, Bulgarian split squat, calf raise -- feet stay
planted, the camera is the primary lower-body kinematics source, so the
wrist is left free for its usual sync/motion-onset/periodicity/identity/
tempo/set-boundary role) and an ankle-group one (leg press, leg
extension, leg curl, hip abduction, hip adduction -- true machine-
footplate exercises where the foot is the rigid contact point with the
resisted load, the same relationship the wrist has to a curl).

**Never reusing wrist thresholds for ankle data or vice versa** means two
things in practice: (1) a caller must know, with real confidence, which
limb a stream of samples is currently coming from before feeding them to
any placement-sensitive threshold/algorithm, and (2) during the physical
act of moving the band, incoming samples are neither -- they're
transitional noise (the strap being unfastened, carried, refastened) that
must not be miscounted as motion signal for whichever limb it used to (or
is about to) represent. This module is the thing that knows the answer to
"trust these samples as side X" at any given moment, and says "not yet"
(pauses) rather than guessing during a transition -- the same
"unknown over incorrect" principle as everywhere else in this repo.

## Product boundary

This does **not** build the member-facing IRIX app (out of scope per the
Phase 3 brief) -- ``request_change`` is the backend entry point a future
app, or front-desk staff console, calls once a member has been instructed
to move their band and has done so; this module has no opinion on how
that instruction reaches the member.

## The state machine

``STABLE`` -- placement confirmed, current ``side`` known, IMU trusted
and un-paused.

``SETTLING`` -- a change was just requested. Fastening/unfastening a
strap and carrying a band from one limb to another produces a burst of
high-variance accelerometer motion that is not a usable IMU signal for
*anything* (not the old placement, not the new one). Every incoming
sample is held in a short rolling window and discarded once it ages out;
IMU-derived events stay paused (``PlacementStatus.paused``) until that
window is quiet (see ``still_accel_std_threshold``) for
``settle_still_duration_s`` straight, in which case the window's already
been positioned to hold exactly the settled portion (fastening motion at
the start simply ages out as the window slides forward), and the state
advances.

``CALIBRATING`` -- motion has settled; the module doesn't yet know *which
way* the device is now oriented (a wrist strapped on with the buckle
facing left reads a different local "up" axis than one facing right, and
an ankle band's typical rest orientation is different from either wrist
orientation) -- it estimates that empirically from the settled window's
own mean acceleration vector (the axis closest to gravity, whichever one
that turns out to be, rather than assuming a fixed convention), then
reuses ``irix.wristband_sim.calibration.calibrate_stationary`` with that
estimated axis to produce a fresh bias calibration for the *new*
placement (a stale wrist-position bias must not carry over to an ankle
placement or vice versa, same principle as the threshold rule above). A
sanity check (the settled window's mean acceleration magnitude actually
being close to 1 g) gates confirmation -- something that looks "quiet"
by variance alone but isn't gravity-consistent (e.g. free-falling, or a
sensor fault) does not get confirmed as stationary. Once enough quiet,
gravity-consistent samples have accumulated, the state advances to
``STABLE`` at the new side and IMU processing resumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from ..fusion.imu import IMUSample
from ..rep_counting.exercises import BandPlacement
from ..wristband_sim.calibration import GRAVITY_M_S2, IMUCalibration, calibrate_stationary


class BandSide(Enum):
    """Which limb, and which side of the body, the band is currently
    worn on. Distinct from ``irix.rep_counting.exercises.BandPlacement``
    (wrist vs. ankle *limb type*, an exercise's static requirement) --
    this is the band's actual, real-time, side-aware physical state.
    ``UNKNOWN`` is a legitimate value, not just an initialization
    placeholder: any time this module can't confidently say where the
    band is, that's what it reports, never a guess.
    """

    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    UNKNOWN = "unknown"


_LIMB_TYPE_OF_SIDE = {
    BandSide.LEFT_WRIST: BandPlacement.WRIST,
    BandSide.RIGHT_WRIST: BandPlacement.WRIST,
    BandSide.LEFT_ANKLE: BandPlacement.ANKLE,
    BandSide.RIGHT_ANKLE: BandPlacement.ANKLE,
    BandSide.UNKNOWN: None,
}


def limb_type_of(side: BandSide) -> Optional[BandPlacement]:
    """The wrist-vs-ankle ``BandPlacement`` a given ``BandSide``
    corresponds to, or ``None`` for ``BandSide.UNKNOWN`` -- the bridge
    between this module's side-aware tracking and ``ExerciseConfig.
    band_placement``'s coarser limb-type requirement, e.g. ``irix.
    pipeline.rep_session.RepSession`` uses this to decide whether the
    band's *current* placement actually matches what the exercise being
    performed needs before trusting any IMU sample for fusion."""
    return _LIMB_TYPE_OF_SIDE[side]


class PlacementState(Enum):
    STABLE = "stable"
    SETTLING = "settling"
    CALIBRATING = "calibrating"


@dataclass
class PlacementStatus:
    """A snapshot of one wristband's placement tracking, returned by
    every ``WristbandPlacementTracker`` call that can change it."""

    wristband_id: str
    side: BandSide
    state: PlacementState
    confidence: float  # meaningful (>0) only once STABLE; 0.0 otherwise
    paused: bool  # True whenever IMU-derived events should be suppressed
    calibration: Optional[IMUCalibration] = None
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "wristband_id": self.wristband_id,
            "side": self.side.value,
            "state": self.state.value,
            "confidence": self.confidence,
            "paused": self.paused,
            "reason": self.reason,
        }


class WristbandPlacementTracker:
    """One wristband's real-time placement state machine -- see the
    module docstring for the full STABLE -> SETTLING -> CALIBRATING ->
    STABLE lifecycle. One instance per currently-tracked band (mirrors
    ``irix.fusion.clock_sync.ClockSyncEstimator``'s per-session lifetime
    in ``irix.live.station_runner.StationSessionRunner``).
    """

    def __init__(
        self,
        wristband_id: str,
        initial_side: BandSide = BandSide.LEFT_WRIST,
        settle_still_duration_s: float = 2.0,
        still_accel_std_threshold_m_s2: float = 1.0,
        min_calibration_samples: int = 10,
        gravity_tolerance_m_s2: float = 1.5,
    ):
        """``initial_side``: assumed placement for a band nothing has
        ever explicitly moved -- ``LEFT_WRIST`` (not ``UNKNOWN``) by
        default because that's how a real band physically ships and is
        checked out (see ``irix.identity.checkout``); a caller with
        actual side information at checkout time should pass it in
        instead. This intentionally does not disambiguate exercise
        joint-triplets by left/right (``ExerciseConfig.joint_triplet``
        hardcodes the left side throughout this repo already -- a
        pre-existing, separate simplification, not something this module
        changes).

        ``settle_still_duration_s``/``still_accel_std_threshold_m_s2``:
        how long, and how quiet (accel-magnitude standard deviation),
        the most recent samples must be before a change is considered
        "settled" -- see ``SETTLING``'s docstring above. Deliberately
        generous defaults (2s, 1 m/s^2) since a false-early settle
        (fastening motion mistaken for stillness) is worse than waiting
        an extra second: it would calibrate a fresh bias against motion,
        not rest, corrupting every later fusion/rep-counting reading
        against the wrong reference. Not first-principles derived from a
        validated dataset -- there isn't one for this yet (see
        docs/VALIDATION.md).

        ``min_calibration_samples``: passed straight through to
        ``calibrate_stationary`` (its own module explains why so few are
        enough: averaging cancels sensor noise, not sensor bias).

        ``gravity_tolerance_m_s2``: how far the settled window's mean
        acceleration magnitude may be from ``GRAVITY_M_S2`` and still be
        accepted as genuinely stationary (rather than e.g. still slowly
        moving in a way with low variance but real non-gravity net
        acceleration).
        """
        self.wristband_id = wristband_id
        self._side = initial_side
        self._state = PlacementState.STABLE
        self._target_side: Optional[BandSide] = None
        self._settle_buffer: List[IMUSample] = []
        self._calibration: Optional[IMUCalibration] = None
        self.settle_still_duration_s = settle_still_duration_s
        self.still_accel_std_threshold_m_s2 = still_accel_std_threshold_m_s2
        self.min_calibration_samples = min_calibration_samples
        self.gravity_tolerance_m_s2 = gravity_tolerance_m_s2

    @property
    def current_side(self) -> BandSide:
        return self._side

    @property
    def state(self) -> PlacementState:
        return self._state

    @property
    def paused(self) -> bool:
        return self._state != PlacementState.STABLE

    @property
    def limb_type(self) -> Optional[BandPlacement]:
        return limb_type_of(self._side)

    def status(self, reason: Optional[str] = None) -> PlacementStatus:
        confidence = 1.0 if self._state == PlacementState.STABLE and self._side != BandSide.UNKNOWN else 0.0
        return PlacementStatus(
            wristband_id=self.wristband_id, side=self._side, state=self._state,
            confidence=confidence, paused=self.paused, calibration=self._calibration, reason=reason,
        )

    def request_change(self, to_side: BandSide, at_time: Optional[float] = None) -> PlacementStatus:
        """Backend entry point for "this band has been moved" (called by
        a future app/front-desk console, per this module's product-
        boundary note -- never inferred automatically from motion alone,
        since a band simply going still doesn't mean it moved). A no-op
        (stays STABLE, no pause) if already stably at ``to_side``."""
        if to_side == self._side and self._state == PlacementState.STABLE:
            return self.status(reason="already at requested side")
        self._target_side = to_side
        self._state = PlacementState.SETTLING
        self._settle_buffer = []
        self._calibration = None
        return self.status(reason=f"placement change requested: {self._side.value} -> {to_side.value}")

    def feed_samples(self, samples: List[IMUSample]) -> PlacementStatus:
        """Feed newly-available samples in. While ``STABLE`` this is a
        no-op observer call (the caller is responsible for actually
        using/storing samples elsewhere, e.g. ``RepSession.
        add_imu_samples``) -- this method only does real work mid-
        change. Returns the current status either way."""
        if not samples or self._state == PlacementState.STABLE:
            return self.status()

        self._settle_buffer.extend(samples)
        now = self._settle_buffer[-1].timestamp
        window_start = now - self.settle_still_duration_s
        self._settle_buffer = [s for s in self._settle_buffer if s.timestamp >= window_start]

        window_span = self._settle_buffer[-1].timestamp - self._settle_buffer[0].timestamp
        # >=95% rather than a strict >= : the span between N discretely-
        # spaced samples is (N-1) * sample_period, not N * sample_period,
        # so a hard exact-equality threshold would almost always fall
        # just short by one sample period regardless of how much data is
        # fed in -- this tolerates that fencepost gap rather than
        # requiring the caller to always over-supply by one extra sample.
        if window_span < 0.95 * self.settle_still_duration_s:
            return self.status(reason="waiting for enough post-change samples")

        accel = np.stack([s.accel for s in self._settle_buffer])
        accel_mag = np.linalg.norm(accel, axis=1)
        if accel_mag.std() > self.still_accel_std_threshold_m_s2:
            self._state = PlacementState.SETTLING
            return self.status(reason="still settling -- motion above stillness threshold")

        self._state = PlacementState.CALIBRATING
        if len(self._settle_buffer) < self.min_calibration_samples:
            return self.status(reason="settled, accumulating enough samples to calibrate")

        mean_accel = accel.mean(axis=0)
        gravity_magnitude = float(np.linalg.norm(mean_accel))
        if abs(gravity_magnitude - GRAVITY_M_S2) > self.gravity_tolerance_m_s2:
            # Quiet by variance but not gravity-consistent (e.g. still
            # translating at constant velocity, or a sensor fault) --
            # not safe to confirm as genuinely at rest. Keep the window
            # sliding rather than confirming a bad calibration.
            return self.status(reason="settled but not gravity-consistent -- not confirming yet")

        # Estimate which local axis is "up" from the data itself, rather
        # than assuming a fixed convention -- see CALIBRATING's docstring
        # for why a fixed axis would be wrong across different physical
        # orientations (buckle-left vs. buckle-right wrist, or an ankle
        # placement's typically different rest orientation).
        gravity_axis = int(np.argmax(np.abs(mean_accel)))
        gravity_sign = float(np.sign(mean_accel[gravity_axis])) or 1.0
        self._calibration = calibrate_stationary(
            self._settle_buffer, gravity_axis=gravity_axis, gravity_sign=gravity_sign,
        )
        self._side = self._target_side
        self._target_side = None
        self._state = PlacementState.STABLE
        self._settle_buffer = []
        return self.status(reason="placement confirmed and recalibrated")
