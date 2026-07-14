from unittest.mock import patch

from irix.weight_recognition.qr_reader import PlateQRReader

# Patched at PlateQRReader._decode (our own seam) rather than
# pyzbar.pyzbar.decode directly: the real pyzbar binding requires the
# system libzbar shared library, which isn't guaranteed to be present in
# every dev/CI environment, and decode() itself is exercised by pyzbar's
# own test suite, not ours.


def test_read_total_weight_sums_recognized_plates():
    reader = PlateQRReader()
    with patch.object(reader, "_decode", return_value=["IRIX-PLATE-20", "IRIX-PLATE-20", "IRIX-PLATE-5"]):
        total = reader.read_total_weight(frame=None)
    assert total == 45.0


def test_read_total_weight_returns_none_when_nothing_recognized():
    reader = PlateQRReader()
    with patch.object(reader, "_decode", return_value=[]):
        assert reader.read_total_weight(frame=None) is None


def test_read_total_weight_ignores_unrecognized_codes():
    reader = PlateQRReader()
    with patch.object(reader, "_decode", return_value=["some-other-gym-sticker"]):
        assert reader.read_total_weight(frame=None) is None
