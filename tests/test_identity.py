from irix.identity.ble_pairing import BLEReading, StationPairing


def test_resolves_to_strongest_rssi():
    pairing = StationPairing()
    readings = [
        BLEReading(station_id="s1", rssi=-70, timestamp=0.0),
        BLEReading(station_id="s2", rssi=-40, timestamp=0.0),
    ]
    assert pairing.resolve(readings) == "s2"


def test_tie_break_uses_recent_motion():
    pairing = StationPairing(rssi_tie_margin=3.0)
    readings = [
        BLEReading(station_id="s1", rssi=-50, timestamp=0.0, recent_motion=False),
        BLEReading(station_id="s2", rssi=-51, timestamp=0.0, recent_motion=True),
    ]
    assert pairing.resolve(readings) == "s2"


def test_empty_readings_returns_none():
    assert StationPairing().resolve([]) is None
