"""Shared crowded-group disambiguation: given more than one checked-out
wristband simultaneously candidate for the same list of detected people,
resolve which detected person is which member via
``irix.identity.motion_correlation``, and keep routing them correctly for
as long as the same candidate group stays put.

Extracted from ``irix.live.station_runner.StationSessionRunner`` (the
single-camera-per-station case) so the exact same buffering/resolution/
sticky-routing logic can also drive ``irix.live.zone_runner.
MultiCameraZoneRunner`` (several cameras with overlapping fields of view
covering one shared physical area) -- see that module's docstring for why
a dense multi-camera zone needs *one of these per camera* rather than one
shared instance for the whole zone. This extraction is a pure refactor:
``StationSessionRunner`` now delegates to exactly one instance of this
class, same buffering window, same sticky-until-group-changes resolution,
same trade-offs -- verified via its existing test suite passing
unchanged.

**Trade-offs, stated plainly** (unchanged from before this class existed
as its own module):

1. While a window is buffering (or for any slot that never resolves
   confidently), frames for the still-ambiguous group aren't attributed
   to anyone -- reps genuinely happening during that short window are
   missed rather than guessed at.
2. Routing assumes a detected person's position in the ``people`` list
   passed to ``route()`` each tick stays consistent for the duration of
   one buffering window -- reasonable for a short window with a static
   camera, not a guarantee over a long session, which is why
   re-resolution happens fresh every time the candidate group changes
   rather than trusting one resolution indefinitely.
"""
from __future__ import annotations

from typing import Callable, Dict, FrozenSet, List, Optional

from ..fusion.imu import IMUSample
from ..identity.motion_correlation import MotionCorrelationResolver
from ..pose.estimator import PersonPose


def _transpose_pose_buffer(pose_buffer: List[List[PersonPose]], n_slots: int) -> List[List[PersonPose]]:
    """``pose_buffer`` is tick-major (one entry per tick, each the list of
    people detected that tick); ``MotionCorrelationResolver.resolve``
    wants person-major (one entry per detected-person slot, each that
    slot's poses over time). A tick where the detected person count
    didn't match ``n_slots`` (a missed/spurious detection) is dropped
    rather than guessed at."""
    slots: List[List[PersonPose]] = [[] for _ in range(n_slots)]
    for tick_people in pose_buffer:
        if len(tick_people) != n_slots:
            continue
        for i, pose in enumerate(tick_people):
            slots[i].append(pose)
    return slots


class CrowdedGroupDisambiguator:
    """One buffering/resolution state machine for one detection source
    (one camera) against a shared candidate wristband group. A caller
    with several detection sources over the same group (a multi-camera
    zone) uses one instance per source -- see the module docstring for
    why that's the right split rather than pooling every camera's
    detections into one buffer.
    """

    def __init__(
        self,
        motion_resolver: Optional[MotionCorrelationResolver] = None,
        disambiguation_window_frames: int = 60,
    ):
        self._motion_resolver = motion_resolver or MotionCorrelationResolver()
        self._disambiguation_window_frames = disambiguation_window_frames

        self._pending_wristband_ids: Optional[FrozenSet[str]] = None
        self._pose_buffer: List[List[PersonPose]] = []
        self._imu_buffer: Dict[str, List[IMUSample]] = {}
        self._slot_assignment: Dict[int, str] = {}
        self._buffer_started_at: Optional[float] = None
        self._buffer_span_s: float = 0.0

    @property
    def slot_assignment(self) -> Dict[int, str]:
        """Currently-resolved detected-person-slot -> wristband_id
        mapping for this source, empty while unresolved/buffering.
        Read-only, mainly for tests/introspection -- callers should use
        ``route()``'s return value for actual routing, not this."""
        return dict(self._slot_assignment)

    def reset(self) -> None:
        """Discard any buffered/resolved state and start fresh on the
        next ``route()`` call -- call whenever the candidate group this
        source should be disambiguating among has become stale for a
        reason the source itself can't detect from ``candidate_wristband_ids``
        alone (e.g. a session ending). Safe to call even when nothing is
        buffered."""
        self._pending_wristband_ids = None
        self._pose_buffer = []
        self._imu_buffer = {}
        self._slot_assignment = {}
        self._buffer_started_at = None
        self._buffer_span_s = 0.0

    def route(
        self,
        now: float,
        candidate_wristband_ids: FrozenSet[str],
        people: List[PersonPose],
        polled_imu: Dict[str, List[IMUSample]],
        resolve_member_id: Callable[[str], Optional[str]],
    ) -> Dict[str, PersonPose]:
        """One tick's worth of work for this detection source.

        ``candidate_wristband_ids`` is the group currently being
        disambiguated among -- a caller may share the exact same group
        across several ``CrowdedGroupDisambiguator`` instances at once
        (see ``irix.live.zone_runner``). ``people``/``polled_imu`` are
        specific to *this* source only. ``resolve_member_id`` maps a
        wristband id to the account it's checked out to (typically
        ``CheckoutRegistry.resolve_member``).

        Returns ``{wristband_id: PersonPose}`` for whichever candidates
        this source can confidently attribute a detected person to this
        tick -- empty while still buffering, or for any candidate this
        source doesn't currently have signal for (e.g. occluded from
        this particular camera's angle this tick).
        """
        if candidate_wristband_ids != self._pending_wristband_ids:
            self._pending_wristband_ids = candidate_wristband_ids
            self._pose_buffer = []
            self._imu_buffer = {wid: [] for wid in candidate_wristband_ids}
            self._slot_assignment = {}
            self._buffer_started_at = now
            self._buffer_span_s = 0.0

        if not self._slot_assignment:
            self._pose_buffer.append(people)
            for wristband_id in candidate_wristband_ids:
                self._imu_buffer.setdefault(wristband_id, []).extend(polled_imu.get(wristband_id, []))
            self._buffer_span_s = now - self._buffer_started_at if self._buffer_started_at is not None else 0.0
            if len(self._pose_buffer) >= self._disambiguation_window_frames:
                self._resolve(resolve_member_id)
            # Still ambiguous (or just resolved too late for this
            # particular tick) -- nothing gets routed this tick; see the
            # module docstring's trade-off note.
            return {}

        routed: Dict[str, PersonPose] = {}
        for slot, wristband_id in self._slot_assignment.items():
            if slot < len(people):
                routed[wristband_id] = people[slot]
        return routed

    def _resolve(self, resolve_member_id: Callable[[str], Optional[str]]) -> None:
        n_slots = len(self._pending_wristband_ids)
        candidate_imu_streams: Dict[str, List[IMUSample]] = {}
        wristband_by_member: Dict[str, str] = {}
        for wristband_id in self._pending_wristband_ids:
            member_id = resolve_member_id(wristband_id)
            if member_id is None:
                continue
            candidate_imu_streams[member_id] = self._imu_buffer.get(wristband_id, [])
            wristband_by_member[member_id] = wristband_id

        elapsed_ticks = max(len(self._pose_buffer) - 1, 1)
        # effective fps from how much real time this buffer actually
        # spans, not an assumed constant -- ticks in a live run don't
        # necessarily land at a perfectly uniform interval.
        pose_fps = float(elapsed_ticks) / self._buffer_span_s if self._buffer_span_s > 0 else 30.0

        results = self._motion_resolver.resolve(
            candidate_imu_streams=candidate_imu_streams,
            detected_people_poses=_transpose_pose_buffer(self._pose_buffer, n_slots),
            pose_fps=pose_fps,
        )
        self._slot_assignment = {}
        for match in results:
            if match is None:
                continue
            wristband_id = wristband_by_member.get(match.member_id)
            if wristband_id is not None:
                self._slot_assignment[match.person_index] = wristband_id

        # Start a fresh buffer regardless of whether every slot resolved
        # -- an unmatched slot gets another window to try again rather
        # than blocking forever on one ambiguous group.
        self._pose_buffer = []
        self._imu_buffer = {wid: [] for wid in self._pending_wristband_ids}
        self._buffer_started_at = None
        self._buffer_span_s = 0.0
