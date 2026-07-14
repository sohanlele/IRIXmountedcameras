# Camera system

## Supported sources today

`cv2.VideoCapture` (via `irix.pose.estimator.PoseEstimator`'s caller and
`irix.live.camera_source.ReconnectingFrameSource`) accepts, uniformly:

- a webcam index (`0`, `1`, ...)
- a video file path (used throughout `tests/` and by `run_upload.py`)
- an RTSP/live stream URL -- `ReconnectingFrameSource("rtsp://...")` and
  `run_demo.py --source rtsp://...` both work today; nothing in the
  frame-reading path is file/webcam-specific.

USB and RTSP cameras are therefore both supported through the same code
path -- there is no separate "USB camera" vs. "RTSP camera" module,
because OpenCV's `VideoCapture` already abstracts that difference. What
*is* genuinely built per-source-type is reconnection behavior (below).

## Camera health / reconnection

`irix.live.camera_source.ReconnectingFrameSource` is the 24/7-camera
piece: `.frames()` yields indefinitely, releasing and reopening the
capture with exponential backoff on any read failure instead of raising
and exiting the process. Real, tested against a fake capture object that
fails on cue (`tests/test_camera_source.py`) -- independent of whether a
real camera is available to test against.

**What's not built yet**: a structured camera *health* signal beyond
reconnect/no-reconnect (e.g. frame-rate degradation short of total
failure, a stuck/frozen frame detector, exposure/focus drift alerts) --
see `docs/TODO.md`. `ReconnectingFrameSource` currently only
distinguishes "reads are succeeding" from "reads are failing," not
degraded-but-still-succeeding states.

## GPU inference / CPU fallback

`irix.pose.estimator.PoseEstimator` wraps `ultralytics` (YOLO-Pose),
which auto-selects GPU (CUDA) if available and falls back to CPU
otherwise -- this is `ultralytics`' own device-selection behavior, not
something this repo re-implements. No explicit device pinning/config
exists yet in this repo's own code (e.g. a config flag to force CPU on a
Jetson without discrete GPU, or to pin a specific GPU index on a
multi-GPU edge box) -- see `docs/DEPLOYMENT.md` and `docs/TODO.md`.

## Camera calibration

Two distinct calibration concerns exist in this repo, for two different
purposes:

- **Bar-path pixel-to-mm calibration** (`irix.barbell.calibration`) --
  self-calibrated from a known real-world object (a competition bumper
  plate's published diameter) visible in-frame, no manual calibration
  step or environment edit required. Per-camera-aware (each `camera_id`
  self-calibrates independently) so a member's set switching from one
  physical camera to another mid-set doesn't misapply the wrong
  camera's scale -- see `docs/ARCHITECTURE.md`'s "Overlapping
  multi-camera zones" section.
- **Geometric camera calibration for 3D triangulation**
  (`irix.pose.multiview.CameraProjection`) -- an optional, explicitly
  *supplied* per-camera projection matrix (intrinsics + extrinsics),
  used only when a deployer wants `MultiCameraZoneRunner` to triangulate
  a 3D pose from 2+ overlapping cameras (DLT triangulation) rather than
  relying on one camera's 2D joint angle. Not self-calibrating -- this
  is a real, one-time installation step (checkerboard/ArUco-based
  intrinsic+extrinsic calibration is the standard approach; not
  implemented in this repo, which only consumes an already-computed
  projection matrix).

## Station zones and overlapping coverage

`irix.topology.registry.StationRegistry` is the static configuration of
a gym's camera/station layout: one `StationInfo` per fixed camera/station
pairing, grouped into `zone`s (routes to `LocalBuffer`/`Aggregator`) and
carrying `adjacent_station_ids` (which stations a member could plausibly
walk to directly, used by `irix.topology.handoff` to flag an implausible
jump as likely a mis-resolved BLE reading rather than a real handoff).

For genuinely overlapping coverage (several cameras over one shared area,
not fixed one-camera-per-station), `irix.live.zone_runner.
MultiCameraZoneRunner` is the real, tested generalization -- see
`docs/ARCHITECTURE.md`'s "Overlapping multi-camera zones" section for the
full design (wristband-IMU correlation ties multiple cameras' views of
one person together, not pixel-level cross-camera re-identification; a
fixed camera-priority rule prevents double-counting when multiple
cameras agree on the same person at once).

## Occlusion

Handled at two levels, deliberately not conflated:

- **Within one camera's view**: `irix.fusion.rep_fusion.RepCountFusion`
  falls back toward the wristband IMU's rep count when camera
  `tracking_confidence` was low for a set (heavy occlusion), rather than
  trusting a camera count produced during a period it likely
  undercounted.
- **Across a multi-camera zone**: `MultiCameraZoneRunner` gives occlusion
  tolerance "for free" -- a member invisible to one camera's angle still
  gets routed correctly via another camera that sees them, since
  identity resolution there is wristband-IMU-correlation-based, not
  pixel-tracking-based.

## What's not built

Real hardware/network deployment configs (camera firmware, PoE/network
topology, edge-box provisioning) are explicitly out of this software
repo's scope -- see `docs/DEPLOYMENT.md`. No frame-rate/quality
degradation health signal beyond binary reconnect success/failure (see
above). No automatic multi-camera extrinsic calibration tooling (the
projection matrices `CameraProjection` consumes must be computed
externally today).
