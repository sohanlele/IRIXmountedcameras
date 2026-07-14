"""RepSession's Phase 3 default weight-recognition wiring: color-coded
bumper-plate detection (irix.weight_recognition.plate_color_check) now
runs on every periodic weight-check frame, not just when a VLM backend
is configured -- see rep_session.py's weight-check block. This is the
zero-training, no-API-key method (Priority 7); a VLM backend, when
configured, stays the primary read and color-plate detection becomes a
cross-check instead (covered by tests/test_run_upload_wiring.py's
existing VLM tests, unaffected by this file)."""
from __future__ import annotations

import cv2
import numpy as np

from irix.pipeline.rep_session import RepSession

# Same swatches/canvas helpers as tests/test_plate_color_check.py.
_BGR_FOR_COLOR = {"green": (0, 180, 0), "yellow": (0, 220, 220), "blue": (200, 0, 0), "red": (0, 0, 220)}


def _canvas(w=600, h=400, bg=(40, 40, 40)):
    return np.full((h, w, 3), bg, dtype=np.uint8)


def _draw_plate(img, center, radius, color_name):
    cv2.circle(img, center, radius, _BGR_FOR_COLOR[color_name], thickness=-1)
    return img


def _frame_with_symmetric_plates(color_name="blue"):
    img = _canvas()
    _draw_plate(img, (150, 200), 60, color_name)
    _draw_plate(img, (450, 200), 60, color_name)  # symmetric pair -- required, see estimate_load_from_color_plates
    return img


def test_color_plate_check_confirms_weight_with_no_vlm_backend_configured():
    session = RepSession(
        exercise_name="squat", member_id="alice", station_id="squat-1", weight_check_every_n_frames=1,
    )
    frame = _frame_with_symmetric_plates("blue")  # 2x 20kg + 20kg bar = 40kg (MENS_OLYMPIC_BARBELL_WEIGHT_KG default)

    events = session.process_frame(frame=frame, ts=0.0, person=None)

    weight_events = [e for e in events if e.to_dict()["event_type"] == "weight_confirmed"]
    assert len(weight_events) == 1
    assert weight_events[0].method == "color_plate"
    assert weight_events[0].weight_kg == 20.0 * 2 + 20.0  # two 20kg plates + bar
    assert weight_events[0].color_check_consistent is True
    assert session._current_weight_kg == weight_events[0].weight_kg


def test_a_frame_with_no_color_coded_plates_produces_no_weight_event():
    session = RepSession(
        exercise_name="squat", member_id="alice", station_id="squat-1", weight_check_every_n_frames=1,
    )
    frame = _canvas()  # plain background, nothing color-coded in it

    events = session.process_frame(frame=frame, ts=0.0, person=None)

    assert not [e for e in events if e.to_dict()["event_type"] == "weight_confirmed"]
    assert session._current_weight_kg is None


def test_an_unpaired_single_plate_is_not_confidently_readable_and_produces_no_event():
    """estimate_load_from_color_plates deliberately refuses to guess at
    an odd/asymmetric count (see that function's docstring) -- a single
    detected plate (no symmetric partner) must not become a fabricated
    weight_confirmed event."""
    session = RepSession(
        exercise_name="squat", member_id="alice", station_id="squat-1", weight_check_every_n_frames=1,
    )
    img = _canvas()
    _draw_plate(img, (300, 200), 60, "red")  # single plate, no pair

    events = session.process_frame(frame=img, ts=0.0, person=None)

    assert not [e for e in events if e.to_dict()["event_type"] == "weight_confirmed"]
