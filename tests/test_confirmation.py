from irix.weight_recognition.confirmation import ExtractionConfirmer, validate_weight_kg


def test_validate_weight_kg_snaps_and_clamps():
    assert validate_weight_kg(20.4) == 20.0  # snaps to nearest 1.25kg step... see below
    assert validate_weight_kg(0.5) is None    # below lo_kg
    assert validate_weight_kg(500) is None    # above hi_kg
    assert validate_weight_kg("not a number") is None


def test_confirmer_requires_n_consistent_reads():
    confirmer = ExtractionConfirmer(confirm_n=3, confirm_window=3, confidence_threshold=0.8)
    assert confirmer.push(20.0, 0.9) is None
    assert confirmer.push(20.0, 0.9) is None
    result = confirmer.push(20.0, 0.9)
    assert result is not None
    assert result.value == 20.0


def test_confirmer_rejects_gaze_scan_inconsistent_values():
    confirmer = ExtractionConfirmer(confirm_n=3, confirm_window=3, confidence_threshold=0.8)
    assert confirmer.push(20.0, 0.9) is None
    assert confirmer.push(50.0, 0.9) is None
    assert confirmer.push(75.0, 0.9) is None
    assert confirmer.push(75.0, 0.9) is None  # window now [50, 75, 75] post-append; still inconsistent
    assert confirmer.push(75.0, 0.9) is not None  # [75, 75, 75]


def test_confirmer_low_confidence_resets_window():
    confirmer = ExtractionConfirmer(confirm_n=2, confirm_window=3, confidence_threshold=0.8)
    assert confirmer.push(20.0, 0.9) is None
    assert confirmer.push(20.0, 0.5) is None  # low confidence -> window cleared
    assert confirmer.push(20.0, 0.9) is None  # only 1 in window now
    assert confirmer.push(20.0, 0.9) is not None


def test_confirmer_applies_validator_and_rejects_invalid():
    confirmer = ExtractionConfirmer(validator=validate_weight_kg, confirm_n=2, confirm_window=2, confidence_threshold=0.8)
    assert confirmer.push(1000.0, 0.9) is None  # invalid -> value becomes None -> window cleared
    result = confirmer.push(20.0, 0.9)
    assert result is None
    result = confirmer.push(20.0, 0.9)
    assert result is not None
    assert result.value == 20.0
