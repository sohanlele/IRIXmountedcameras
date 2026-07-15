"""Per-gym external configuration (Priority 10) -- every gym-specific
assumption named in the founding brief (camera layouts, station layouts,
equipment, camera calibration, bar weights, machine increments,
thresholds, BLE gateway, CPU/GPU mode) as one loadable/saveable file,
plus factories that build the actual runtime objects
(``irix.topology.registry.StationRegistry``, per-station ``RepSession``/
``StationSessionRunner`` constructor kwargs) from it.

## Why this didn't already exist

Every demo/test in this repo before this module builds its station
layout, thresholds, and equipment assumptions as Python literals inline
(``StationInfo("squat-1", "cam-1", ...)``, ``RepSession(..., rest_gap_s=
20.0)``) -- correct for a demo needing exactly one hardcoded scenario,
wrong for "add a second real gym" (a second hardcoded Python module isn't
configuration, it's a second deployment of the code). This module is the
actual configuration layer: one file per gym, loaded at startup, no code
change needed to point this repo at a different gym's layout/equipment/
thresholds.

## What's deliberately NOT in here

Camera/BLE *hardware* connections (``frame_source``, ``ble_reader``,
``imu_stream_factory`` -- real device handles/URLs) stay caller-supplied
callables, not configuration values: which physical camera driver or BLE
stack a deployment uses is an integration decision, not a per-gym data
value, and forcing it into this file would mean this module importing
hardware-specific code it has no business depending on. This file
configures *parameters*, not *hardware bindings* -- see
``docs/DEPLOYMENT.md`` for where hardware wiring actually happens.

``irix.pose.calibration.CalibrationProfile`` itself (the actual
intrinsic/extrinsic numbers from a checkerboard calibration run) is
referenced here only by file path (``StationConfig.calibration_profile_
path``) -- loading a calibration profile is ``CalibrationProfile.load``'s
job (see that module), not duplicated here.

## Not yet consumed by anything

``StationConfig.machine_weight_increment_kg`` -- forward-looking
metadata for machine weight-stack reading, which does not exist yet in
this repo at all (see ``docs/TODO.md``'s "machine weight-stack reading"
entry). Included now so a gym's config file doesn't need a breaking
schema change once that capability is built.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

try:
    import yaml
    _HAVE_YAML = True
except ImportError:
    _HAVE_YAML = False

from ..barbell.calibration import MENS_OLYMPIC_BARBELL_WEIGHT_KG
from ..topology.registry import StationInfo, StationRegistry
from ..wristband_sim.simulator import DEFAULT_RSSI_AT_STATION_DBM, DEFAULT_RSSI_NOISE_STD


@dataclass
class StationConfig:
    station_id: str
    camera_id: str
    zone: str
    exercise: str
    adjacent_station_ids: List[str] = field(default_factory=list)
    # Equipment metadata (Priority 7/10) -- None means "use RepSession's
    # own default" (the men's Olympic 20kg bar) rather than this config
    # silently overriding it with a duplicated copy of that same default.
    bar_weight_kg: Optional[float] = None
    machine_weight_increment_kg: Optional[float] = None  # not yet consumed -- see module docstring
    camera_tilt_deg: float = 0.0
    calibration_profile_path: Optional[str] = None


@dataclass
class ThresholdsConfig:
    presence_timeout_s: float = 5.0
    rest_gap_s: float = 20.0
    tracking_lost_after_frames: int = 15
    min_consecutive: int = 3
    rssi_tie_margin: float = 3.0
    weight_check_every_n_frames: int = 30


@dataclass
class BLEGatewayConfig:
    """Only meaningful for ``irix.wristband_sim`` (simulation/testing) --
    a real BLE gateway's packet-loss/RSSI characteristics are measured,
    not configured, but the simulator needs these same names as
    constructor args to simulate a *specific* gym's radio environment
    (a large steel-frame gym floor genuinely has worse RSSI/packet-loss
    behavior than a small studio -- see ``irix.wristband_sim.simulator``'s
    module docstring for the BLE spec this models)."""

    packet_loss_pct: float = 0.0
    rssi_at_station_dbm: float = DEFAULT_RSSI_AT_STATION_DBM
    rssi_noise_std: float = DEFAULT_RSSI_NOISE_STD


@dataclass
class ComputeConfig:
    mode: str = "cpu"  # "cpu" | "gpu" -- irix.pose.estimator/benchmark's own hardware detection is still authoritative;
    # this is a deployment's *intent* (what it expects to run on), useful for validating an install against what the
    # edge box actually reports (irix.benchmark.run_benchmarks' environment.gpu block) rather than a switch this
    # module itself flips anything based on.


@dataclass
class GymConfig:
    gym_id: str
    stations: List[StationConfig] = field(default_factory=list)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    ble_gateway: BLEGatewayConfig = field(default_factory=BLEGatewayConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gym_id": self.gym_id,
            "stations": [asdict(s) for s in self.stations],
            "thresholds": asdict(self.thresholds),
            "ble_gateway": asdict(self.ble_gateway),
            "compute": asdict(self.compute),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GymConfig":
        return GymConfig(
            gym_id=d["gym_id"],
            stations=[StationConfig(**s) for s in d.get("stations", [])],
            thresholds=ThresholdsConfig(**d.get("thresholds", {})),
            ble_gateway=BLEGatewayConfig(**d.get("ble_gateway", {})),
            compute=ComputeConfig(**d.get("compute", {})),
        )

    def station(self, station_id: str) -> Optional[StationConfig]:
        return next((s for s in self.stations if s.station_id == station_id), None)


def load_gym_config(path: str) -> GymConfig:
    """Load a ``GymConfig`` from a ``.yaml``/``.yml`` or ``.json`` file.
    Raises ``ValueError`` on an unrecognized extension or a missing
    ``.yaml``/``.yml`` reader (PyYAML not installed) rather than
    guessing the format from content -- same "fail loudly, don't guess"
    posture as ``irix.fusion.imu_io.load_imu_samples``."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"gym config not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    with open(path) as f:
        if ext in (".yaml", ".yml"):
            if not _HAVE_YAML:
                raise ValueError(f"{path}: PyYAML is not installed -- cannot load a .yaml config")
            d = yaml.safe_load(f)
        elif ext == ".json":
            d = json.load(f)
        else:
            raise ValueError(f"{path}: unrecognized config extension {ext!r} -- expected .yaml/.yml or .json")
    return GymConfig.from_dict(d)


def save_gym_config(config: GymConfig, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "w") as f:
        if ext in (".yaml", ".yml"):
            if not _HAVE_YAML:
                raise ValueError(f"{path}: PyYAML is not installed -- cannot save a .yaml config")
            yaml.safe_dump(config.to_dict(), f, sort_keys=False)
        elif ext == ".json":
            json.dump(config.to_dict(), f, indent=2)
        else:
            raise ValueError(f"{path}: unrecognized config extension {ext!r} -- expected .yaml/.yml or .json")


def build_station_registry(config: GymConfig) -> StationRegistry:
    """The camera/station-layout half of this config, as the
    ``irix.topology.registry.StationRegistry`` every handoff/zone-routing
    module already expects -- this function is the only place a caller
    needs to translate one into the other."""
    return StationRegistry([
        StationInfo(
            station_id=s.station_id, camera_id=s.camera_id, zone=s.zone,
            default_exercise=s.exercise, adjacent_station_ids=list(s.adjacent_station_ids),
        )
        for s in config.stations
    ])


def rep_session_kwargs_for(config: GymConfig, station_id: str) -> Dict[str, Any]:
    """Keyword args for ``irix.pipeline.rep_session.RepSession`` (or,
    equivalently, the subset ``irix.live.station_runner.
    StationSessionRunner`` forwards into one per session) reflecting this
    station's equipment metadata and this gym's thresholds. Raises
    ``KeyError`` for an unknown ``station_id`` -- silently falling back
    to defaults for a station that isn't actually configured would hide
    a real config-file bug (a typo'd station_id) behind quietly-wrong
    equipment assumptions."""
    station = config.station(station_id)
    if station is None:
        raise KeyError(f"{station_id!r} is not a configured station in gym {config.gym_id!r}")
    kwargs: Dict[str, Any] = {
        "rest_gap_s": config.thresholds.rest_gap_s,
        "weight_check_every_n_frames": config.thresholds.weight_check_every_n_frames,
        "camera_tilt_deg": station.camera_tilt_deg,
    }
    kwargs["bar_weight_kg"] = (
        station.bar_weight_kg if station.bar_weight_kg is not None else MENS_OLYMPIC_BARBELL_WEIGHT_KG
    )
    return kwargs


def station_runner_kwargs_for(config: GymConfig, station_id: str) -> Dict[str, Any]:
    """Keyword args for ``irix.live.station_runner.StationSessionRunner``
    combining this station's ``rep_session_kwargs_for`` with the
    gym-wide presence/tracking thresholds. Still needs ``station_id``,
    ``exercise_name`` (``config.station(station_id).exercise``),
    ``checkout_registry``, and every hardware-binding constructor arg
    (``frame_source``, ``ble_reader``, ``imu_stream_factory``,
    ``pose_estimator``) supplied by the caller -- see the module
    docstring for why those stay outside configuration."""
    kwargs = rep_session_kwargs_for(config, station_id)
    kwargs["presence_timeout_s"] = config.thresholds.presence_timeout_s
    kwargs["tracking_lost_after_frames"] = config.thresholds.tracking_lost_after_frames
    return kwargs
