from irix.pipeline.aggregator import Aggregator
from irix.pipeline.cloud_sync import InMemoryCloudSync
from irix.pipeline.edge_buffer import LocalBuffer
from irix.pipeline.schema import (
    BandPlacementRequiredEvent,
    RepCompletedEvent,
    SetCompleteEvent,
    WeightConfirmedEvent,
)


def test_aggregator_drains_zones_and_forwards_events():
    buffer_a = LocalBuffer()
    buffer_b = LocalBuffer()
    buffer_a.push(RepCompletedEvent(member_id="m1", station_id="s1", exercise="squat", rep_count=1))
    buffer_b.push(RepCompletedEvent(member_id="m2", station_id="s2", exercise="bicep_curl", rep_count=3))

    cloud = InMemoryCloudSync()
    agg = Aggregator(cloud_sync=cloud)
    agg.register_zone("zone-a", buffer_a)
    agg.register_zone("zone-b", buffer_b)

    synced = agg.sync()

    assert synced == 2
    assert len(cloud.received) == 2
    assert len(buffer_a) == 0 and len(buffer_b) == 0


def test_aggregator_forwards_mixed_event_types():
    buffer = LocalBuffer()
    buffer.push(RepCompletedEvent(member_id="m1", station_id="s1", exercise="squat", rep_count=1))
    buffer.push(SetCompleteEvent(member_id="m1", station_id="s1", exercise="squat", total_reps=8))
    buffer.push(
        BandPlacementRequiredEvent(member_id="m1", exercise="leg_press", from_placement="wrist", to_placement="ankle")
    )
    buffer.push(
        WeightConfirmedEvent(member_id="m1", station_id="s1", exercise="squat", weight_kg=60.0, confidence=0.9)
    )

    cloud = InMemoryCloudSync()
    agg = Aggregator(cloud_sync=cloud)
    agg.register_zone("zone-a", buffer)
    synced = agg.sync()

    assert synced == 4
    event_types = {e.to_dict()["event_type"] for e in cloud.received}
    assert event_types == {"rep_completed", "set_complete", "band_placement_required", "weight_confirmed"}


def test_rep_completed_event_has_no_raw_video_fields():
    event = RepCompletedEvent(member_id="m1", station_id="s1", exercise="squat", rep_count=1)
    d = event.to_dict()
    assert "frame" not in d and "video" not in d and "image" not in d


def test_all_event_types_carry_no_raw_video_or_biometric_fields():
    events = [
        RepCompletedEvent(member_id="m1", station_id="s1", exercise="squat", rep_count=1),
        SetCompleteEvent(member_id="m1", station_id="s1", exercise="squat", total_reps=8),
        BandPlacementRequiredEvent(member_id="m1", exercise="leg_press", from_placement="wrist", to_placement="ankle"),
        WeightConfirmedEvent(member_id="m1", station_id="s1", exercise="squat", weight_kg=60.0, confidence=0.9),
    ]
    banned = {"frame", "video", "image", "face", "biometric"}
    for event in events:
        keys = set(event.to_dict().keys())
        assert not (keys & banned), f"{event} leaked a banned field: {keys & banned}"
