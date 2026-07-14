# Tracking

## Pose estimation

`irix.pose.estimator.PoseEstimator` wraps Ultralytics YOLO-Pose
(`yolov8n-pose.pt`, a real, freely available, auto-downloading
pretrained checkpoint on COCO's 17-keypoint layout). Generic single-RGB-
camera human pose estimation is treated as a solved, commodity problem
here -- no custom training, verified end-to-end against a real image and
a real video in `tests/test_pose_estimator_integration.py` with no
mocking in that chain. `irix.pose.geometry` turns keypoints into joint
angles; `irix.pose.multiview.CameraProjection` does DLT triangulation of
a 3D pose from 2+ calibrated overlapping cameras (optional, see
`docs/CAMERA_SYSTEM.md`).

## Multi-person tracking

`PoseEstimator.estimate()` returns a list of `PersonPose` per frame with
**no persistent cross-frame tracking** -- `PersonPose.track_id` is just
that frame's list index, not a stable identity across frames. This is a
deliberate, stated limitation (see `irix.live.station_runner`'s
docstring): short-window routing (e.g. crowded-station disambiguation)
assumes a detected person's list position stays consistent for the
duration of one short buffering window, which is reasonable for a static
camera over a few seconds but not a guarantee over a whole session --
which is exactly why re-resolution happens fresh every time the
present-band group changes rather than trusting one resolution
indefinitely.

## Identity persistence: wristband, never face

Identity is never derived from pixels. Every identity decision in this
repo (`CheckoutRegistry`, `GymCoordinator`, `MotionCorrelationResolver`)
resolves *which wristband*, never *whose face* -- `PoseEstimator` outputs
a skeleton, not a face-geometry embedding. See `docs/PRODUCT_SPEC.md`'s
non-goals and `docs/ARCHITECTURE.md`'s "Where this repo ends" section for
the BIPA/privacy reasoning this is grounded in.

## Camera handoff

`irix.topology.handoff.GymCoordinator` resolves, gym-wide, which single
station is authoritative for each checked-out member at a time, with
hysteresis (`min_consecutive` readings favoring a different station
before an actual handoff fires) to absorb BLE RSSI jitter near a station
boundary rather than emitting a spurious handoff on every noisy reading.
Wired into the live path via `irix.live.gym_runner.GymSessionRunner` --
see `irix/demo/run_live_gym_demo.py` for a runnable example producing a
real `StationHandoffEvent`.

## Camera + IMU association

Two distinct problems, solved differently:

- **Which station's BLE radio sees this band** -- RSSI proximity
  (`irix.identity.ble_pairing.StationPairing`), the coarse-grained
  signal.
- **Which detected skeleton is which band, when RSSI alone can't say**
  (two checked-out members' bands both resolving to the same crowded
  station) -- `irix.identity.motion_correlation.MotionCorrelationResolver`
  cross-correlates each candidate detected person's wrist motion against
  each candidate wristband's raw IMU signal. Grounded in published prior
  art: Sensors (2020), "Person Re-Identification Using Deep Modeling of
  Temporally Correlated Inertial Motion Patterns" (86-subject validation
  of the same wearable-IMU-to-tracked-body-motion matching idea) -- see
  `irix_competitive_research.md` in the business workspace and
  `docs/RESEARCH_LOG.md` for the full citation.

## Occlusion recovery

See `docs/CAMERA_SYSTEM.md`'s "Occlusion" section -- within-camera
fallback to IMU rep count (`RepCountFusion`), cross-camera tolerance via
`MultiCameraZoneRunner`'s wristband-correlation-based association rather
than pixel re-identification.

## Ambiguity handling: unknown over wrong

Every ambiguous case in this repo is designed to leave frames/events
unattributed rather than guess:

- A crowded-station disambiguation window that hasn't resolved
  confidently yet -- frames aren't routed to anyone during that window.
- An unregistered/implausible station jump -- flagged
  (`plausible_adjacency=False` on `StationHandoffEvent`) rather than
  silently trusted.
- A band that isn't currently checked out -- never tracked, regardless
  of what BLE readings say about it.

This is a stated design principle from the founding brief ("never guess
identities... unknown is better than incorrect") applied consistently
across every identity-adjacent module, not just one of them.

## What's not built

No cross-frame person re-identification model (relies on wristband-IMU
correlation instead, by design -- see "Multi-person tracking" above). No
BLE Angle-of-Arrival or UWB-based positioning (documented upgrade path in
`irix.identity.ble_pairing`, not implemented -- no field data yet
justifying the hardware cost).
