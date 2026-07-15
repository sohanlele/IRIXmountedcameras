"""irix.config.gym_config -- Priority 10's per-gym configuration file
and the factories that build real runtime objects from it."""
from __future__ import annotations

import os

import pytest

from irix.config import (
    BLEGatewayConfig,
    GymConfig,
    StationConfig,
    ThresholdsConfig,
    build_station_registry,
    load_gym_config,
    rep_session_kwargs_for,
    save_gym_config,
    station_runner_kwargs_for,
)

_EXAMPLE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "example_gym.yaml")


def _small_config() -> GymConfig:
    return GymConfig(
        gym_id="test-gym",
        stations=[
            StationConfig(station_id="squat-1", camera_id="cam-1", zone="free_weights", exercise="squat",
                          adjacent_station_ids=["curl-1"]),
            StationConfig(station_id="curl-1", camera_id="cam-2", zone="dumbbell", exercise="bicep_curl",
                          adjacent_station_ids=["squat-1"], bar_weight_kg=15.0),
        ],
        thresholds=ThresholdsConfig(presence_timeout_s=7.0),
        ble_gateway=BLEGatewayConfig(packet_loss_pct=0.05),
    )


def test_the_bundled_example_config_loads_and_has_ten_stations():
    config = load_gym_config(_EXAMPLE_CONFIG_PATH)
    assert config.gym_id == "example-gym-1"
    assert len(config.stations) == 10
    hack_squat = config.station("hack-squat-1")
    assert hack_squat is not None
    assert hack_squat.bar_weight_kg == 25.0
    squat_1 = config.station("squat-1")
    assert squat_1.bar_weight_kg is None  # uses RepSession's own default


def test_build_station_registry_reflects_layout_and_adjacency():
    config = load_gym_config(_EXAMPLE_CONFIG_PATH)
    registry = build_station_registry(config)
    assert registry.get("squat-1").camera_id == "cam-1"
    assert registry.get("squat-1").default_exercise == "squat"
    assert registry.is_adjacent("squat-1", "squat-2") is True
    assert registry.is_adjacent("squat-1", "hack-squat-1") is False


def test_rep_session_kwargs_uses_station_override_when_present():
    config = _small_config()
    kwargs = rep_session_kwargs_for(config, "curl-1")
    assert kwargs["bar_weight_kg"] == 15.0
    assert kwargs["rest_gap_s"] == ThresholdsConfig().rest_gap_s  # default, not overridden here


def test_rep_session_kwargs_falls_back_to_default_bar_weight_when_unset():
    from irix.barbell.calibration import MENS_OLYMPIC_BARBELL_WEIGHT_KG

    config = _small_config()
    kwargs = rep_session_kwargs_for(config, "squat-1")
    assert kwargs["bar_weight_kg"] == MENS_OLYMPIC_BARBELL_WEIGHT_KG


def test_rep_session_kwargs_for_unknown_station_raises():
    config = _small_config()
    with pytest.raises(KeyError):
        rep_session_kwargs_for(config, "nonexistent-station")


def test_station_runner_kwargs_includes_presence_and_tracking_thresholds():
    config = _small_config()
    kwargs = station_runner_kwargs_for(config, "squat-1")
    assert kwargs["presence_timeout_s"] == 7.0
    assert kwargs["tracking_lost_after_frames"] == ThresholdsConfig().tracking_lost_after_frames


def test_rep_session_kwargs_actually_construct_a_working_rep_session():
    """End-to-end: config -> kwargs -> a real RepSession that behaves as
    configured, not just a dict that looks right."""
    from irix.pipeline.rep_session import RepSession

    config = _small_config()
    kwargs = rep_session_kwargs_for(config, "curl-1")
    session = RepSession(exercise_name="bicep_curl", member_id="alice", station_id="curl-1", **kwargs)
    assert session.bar_weight_kg == 15.0


def test_save_and_load_json_round_trips(tmp_path):
    config = _small_config()
    path = str(tmp_path / "gym.json")
    save_gym_config(config, path)
    reloaded = load_gym_config(path)
    assert reloaded.gym_id == "test-gym"
    assert len(reloaded.stations) == 2
    assert reloaded.station("curl-1").bar_weight_kg == 15.0
    assert reloaded.thresholds.presence_timeout_s == 7.0
    assert reloaded.ble_gateway.packet_loss_pct == 0.05


def test_save_and_load_yaml_round_trips(tmp_path):
    config = _small_config()
    path = str(tmp_path / "gym.yaml")
    save_gym_config(config, path)
    reloaded = load_gym_config(path)
    assert reloaded.to_dict() == config.to_dict()


def test_loading_a_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_gym_config("/nonexistent/path/gym.yaml")


def test_loading_an_unrecognized_extension_raises_value_error(tmp_path):
    path = tmp_path / "gym.txt"
    path.write_text("not a real config")
    with pytest.raises(ValueError):
        load_gym_config(str(path))
