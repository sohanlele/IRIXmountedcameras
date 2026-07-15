# Deployment

## Current status: not a deployable build

This repo is, in its own README's words, "early scaffold, not production
code." No camera/network hardware, wristband firmware, or edge-device
(e.g. NVIDIA Jetson) deployment configs are included -- those are
explicitly out of scope for a pure-software repo per the founding brief,
and nothing here should be read as ready to install in a real gym today.
This document describes how to run what exists (locally, for
development/demo purposes) and what real deployment would additionally
require.

## Local install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # numpy, opencv, ultralytics, pyzbar, google-genai, pytest
# or, lighter:
pip install -e .                     # core only (numpy, opencv-headless, scipy)
pip install -e ".[pose,qr,vlm,dev]"  # + real pose inference, QR reading, GeminiVLMBackend, tests
```

`ultralytics` (pose inference) pulls in `torch` and is the heaviest
dependency by far -- installable in a constrained environment (e.g. a
disk-limited sandbox), core (`numpy`/`opencv-headless`/`scipy`/`pytest`)
installs comfortably without it, and all tests except the two that
directly need `ultralytics`/`google-genai` pass with just the core
install (3 skipped without those extras, out of 237+ total as of this
writing -- see `docs/VALIDATION.md`).

## Runnable entry points (no camera/hardware required)

```bash
python -m irix.demo.run_demo --mock-pose --exercise squat
python -m irix.demo.run_gym_demo
python -m irix.demo.run_live_gym_demo   # exercises the actual live orchestration (StationSessionRunner/GymSessionRunner)
python -m irix.demo.run_upload --video squat.mp4 --exercise squat
```

See the top-level README for the full flag reference on each.

## What a real deployment additionally needs (not built here)

- **Edge compute hardware selection** -- an NVIDIA Jetson class device
  (Orin Nano/NX or similar) is the natural target for on-prem GPU pose
  inference per station/zone, consistent with "GPU inference, CPU
  fallback" in the founding brief and `ultralytics`' own CUDA support --
  no Jetson-specific deployment config (container image, systemd unit,
  power/thermal profile) exists in this repo yet.
- **Containerization** -- no `Dockerfile`/`docker-compose.yml` exists in
  this repo (contrast `open-wearables-mirror`, an unrelated adjacent
  project in the business workspace, which does ship one -- worth using
  as a structural reference, not reusing directly, since its stack is
  FastAPI + React + Postgres, not this repo's edge-inference shape).
- ~~**Configuration system**~~ -- done (Phase 3, Priority 10):
  `irix.config.gym_config.GymConfig` loads a per-gym YAML/JSON file
  (station/camera layout, per-station equipment metadata, thresholds,
  BLE gateway parameters, compute mode) and
  `build_station_registry`/`rep_session_kwargs_for`/
  `station_runner_kwargs_for` build the real runtime objects from it --
  see `configs/example_gym.yaml` for a complete example matching this
  repo's existing 10-station demo layout. Deliberately does **not**
  configure hardware bindings (`frame_source`/`ble_reader`/
  `imu_stream_factory`/`pose_estimator` stay caller-supplied callables --
  see the module docstring for why) or camera calibration numbers
  themselves (`StationConfig.calibration_profile_path` points at a
  separately-produced `irix.pose.calibration.CalibrationProfile` file,
  not duplicated inline).
- **Wristband hardware + firmware** -- see `docs/WRISTBAND_SYSTEM.md`'s
  hardware recommendation section. Not purchased, not built.
- **Camera network/PoE topology, RTSP credentials management** -- out of
  a software repo's scope; `ReconnectingFrameSource` accepts any RTSP
  URL a deployer supplies, including one with embedded credentials, but
  doesn't manage secrets/rotation.
- **`irix-mvp-app` live-ingestion endpoint** -- `HTTPCloudSync` is an
  unwired placeholder; a real endpoint path/auth scheme doesn't exist to
  point it at yet (see `docs/API_SPEC.md`).
- **Monitoring/observability** -- no metrics export (Prometheus or
  similar) or structured logging configuration exists yet; see
  "Engineering standards" gap in `docs/TODO.md`.

## Environment / secrets

`GeminiVLMBackend` requires an API key supplied by the deployer at
construction time (`GeminiVLMBackend(api_key=...)`) -- **no key is
bundled or hardcoded anywhere in this repo.** No other external service
credentials are required to run anything in this repo today.
