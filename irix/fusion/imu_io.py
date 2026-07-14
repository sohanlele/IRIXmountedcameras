"""Loads a real, recorded wristband IMU export into ``IMUSample`` objects.

Every IMU-driven feature in this repo so far (``RecoFitCounter``/
``ULiftCounter`` in ``imu_rep_counting.py``, ``RepCountFusion``, the
motion-correlation re-ID in ``irix.identity``) has only ever been
exercised against ``irix.demo.mock_pose.synthetic_imu_stream`` -- there
was no code path that turned a real recorded wristband export into
``IMUSample`` objects at all. This module is that path, so
``irix.demo.run_upload`` can run the real fusion/fatigue pipeline against
an actually-uploaded wristband recording instead of only synthetic data.

**File format.** Two are supported, chosen by extension:

- CSV, with a header row containing exactly these seven columns (any
  order): ``timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y,
  gyro_z``. Units match ``IMUSample`` itself: ``timestamp`` in seconds
  (monotonically increasing, same convention as the rest of this repo --
  seconds since recording start is the simplest choice), ``accel_*`` in
  m/s^2, ``gyro_*`` in rad/s.
- JSON: a top-level list of objects, each with the same seven keys.

This is a single, explicit contract, not an attempt to auto-detect every
wristband vendor's native export format -- whatever a real device
actually produces should be reshaped into this format (a short
conversion script per vendor) before being handed to
``load_imu_samples``. Malformed input fails loudly with the offending row
number rather than being silently skipped or coerced: a rep-counting or
fusion result computed from a silently-dropped or silently-wrong sample
is worse than no result at all, especially once ``RepCountFusion`` is
deciding whether to trust the IMU count over the camera's.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List

from .imu import IMUSample

_REQUIRED_FIELDS = ("timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z")


def _row_to_sample(row: Dict[str, Any], row_number: int, source: str) -> IMUSample:
    missing = [f for f in _REQUIRED_FIELDS if f not in row or row[f] is None or row[f] == ""]
    if missing:
        raise ValueError(f"{source}: row {row_number} is missing field(s) {missing} -- required: {_REQUIRED_FIELDS}")
    try:
        values = {f: float(row[f]) for f in _REQUIRED_FIELDS}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: row {row_number} has a non-numeric value in {row} -- {exc}") from exc
    return IMUSample(
        timestamp=values["timestamp"],
        accel=[values["accel_x"], values["accel_y"], values["accel_z"]],
        gyro=[values["gyro_x"], values["gyro_y"], values["gyro_z"]],
    )


def _load_csv(path: str) -> List[IMUSample]:
    samples: List[IMUSample] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: file has no header row")
        missing_cols = [c for c in _REQUIRED_FIELDS if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(
                f"{path}: header is missing column(s) {missing_cols} -- "
                f"found {reader.fieldnames}, need {_REQUIRED_FIELDS}"
            )
        for i, row in enumerate(reader, start=2):  # row 1 is the header
            samples.append(_row_to_sample(row, i, path))
    return samples


def _load_json(path: str) -> List[IMUSample]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a top-level JSON list of sample objects, got {type(data).__name__}")
    samples: List[IMUSample] = []
    for i, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: entry {i} is not an object ({row!r})")
        samples.append(_row_to_sample(row, i, path))
    return samples


def load_imu_samples(path: str) -> List[IMUSample]:
    """Load a real wristband IMU recording (CSV or JSON, see module
    docstring for the exact format) into ``IMUSample`` objects, sorted by
    timestamp.

    Raises ``ValueError`` (with the offending row number) on any
    malformed row, and ``FileNotFoundError``/``ValueError`` for a missing
    or unrecognized-extension path -- deliberately strict, since a
    silently-dropped or silently-wrong sample would corrupt
    ``RepCountFusion``/``RecoFitCounter`` results without any visible
    error.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"IMU data file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        samples = _load_csv(path)
    elif ext == ".json":
        samples = _load_json(path)
    else:
        raise ValueError(f"{path}: unrecognized IMU file extension {ext!r} -- expected .csv or .json")
    if not samples:
        raise ValueError(f"{path}: no IMU samples found in file")
    samples.sort(key=lambda s: s.timestamp)
    return samples
