"""Tests for the VLM-based VisionPlateClassifier, using FakeVLMBackend to
script responses -- no real local/cloud model call. See
irix/weight_recognition/vision_classifier.py for why this replaced the
printed-number / QR-sticker approaches (illegible at camera distance;
stickers are an environment edit that isn't allowed)."""
from irix.weight_recognition.vision_classifier import VisionPlateClassifier
from irix.weight_recognition.vlm_backend import FakeVLMBackend


def test_confirms_after_n_consistent_reads():
    backend = FakeVLMBackend([
        {"plates_visible": True, "total_weight_kg": 20.0, "confidence": 0.9},
        {"plates_visible": True, "total_weight_kg": 20.0, "confidence": 0.9},
        {"plates_visible": True, "total_weight_kg": 20.0, "confidence": 0.9},
    ])
    classifier = VisionPlateClassifier(backend, confirm_n=3, confirm_window=3)
    frame = None
    assert classifier.read_frame(frame) is None
    assert classifier.read_frame(frame) is None
    result = classifier.read_frame(frame)
    assert result is not None
    assert result.value == 20.0


def test_rejects_gaze_scan_across_different_stations():
    # e.g. camera briefly catches a neighboring station's loaded bar
    # before settling on the actual station -- values disagree, should
    # never confirm on the noisy sequence.
    backend = FakeVLMBackend([
        {"plates_visible": True, "total_weight_kg": 20.0, "confidence": 0.9},
        {"plates_visible": True, "total_weight_kg": 60.0, "confidence": 0.9},
        {"plates_visible": True, "total_weight_kg": 20.0, "confidence": 0.9},
    ])
    classifier = VisionPlateClassifier(backend, confirm_n=3, confirm_window=3)
    frame = None
    for _ in range(3):
        result = classifier.read_frame(frame)
    assert result is None


def test_no_plates_visible_resets_and_never_confirms():
    backend = FakeVLMBackend([
        {"plates_visible": True, "total_weight_kg": 20.0, "confidence": 0.9},
        {"plates_visible": True, "total_weight_kg": 20.0, "confidence": 0.9},
        {"plates_visible": False, "confidence": 0.1},
        {"plates_visible": True, "total_weight_kg": 20.0, "confidence": 0.9},
        {"plates_visible": True, "total_weight_kg": 20.0, "confidence": 0.9},
    ])
    classifier = VisionPlateClassifier(backend, confirm_n=3, confirm_window=3)
    frame = None
    results = [classifier.read_frame(frame) for _ in range(5)]
    assert all(r is None for r in results)  # occlusion mid-run resets the window


def test_low_confidence_read_does_not_confirm():
    backend = FakeVLMBackend([
        {"plates_visible": True, "total_weight_kg": 20.0, "confidence": 0.5},
        {"plates_visible": True, "total_weight_kg": 20.0, "confidence": 0.5},
        {"plates_visible": True, "total_weight_kg": 20.0, "confidence": 0.5},
    ])
    classifier = VisionPlateClassifier(backend, confirm_n=3, confirm_window=3, confidence_threshold=0.8)
    frame = None
    results = [classifier.read_frame(frame) for _ in range(3)]
    assert all(r is None for r in results)
