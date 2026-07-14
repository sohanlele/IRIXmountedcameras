from irix.pipeline.aggregator import Aggregator
from irix.pipeline.cloud_sync import InMemoryCloudSync
from irix.pipeline.edge_buffer import LocalBuffer
from irix.pipeline.schema import DerivedMetricsEvent


def test_aggregator_drains_zones_and_forwards_to_cloud():
    buffer_a = LocalBuffer()
    buffer_b = LocalBuffer()
    buffer_a.push(DerivedMetricsEvent(member_id="m1", station_id="s1", exercise="squat", rep_count=1))
    buffer_b.push(DerivedMetricsEvent(member_id="m2", station_id="s2", exercise="bicep_curl", rep_count=3))

    cloud = InMemoryCloudSync()
    agg = Aggregator(cloud_sync=cloud)
    agg.register_zone("zone-a", buffer_a)
    agg.register_zone("zone-b", buffer_b)

    synced = agg.sync()

    assert synced == 2
    assert len(cloud.received) == 2
    assert len(buffer_a) == 0 and len(buffer_b) == 0


def test_derived_metrics_event_has_no_raw_video_fields():
    event = DerivedMetricsEvent(member_id="m1", station_id="s1", exercise="squat", rep_count=1)
    d = event.to_dict()
    assert "frame" not in d and "video" not in d and "image" not in d
