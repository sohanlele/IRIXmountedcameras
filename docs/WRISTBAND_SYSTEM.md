# Wristband system

## IMU sample model

`irix.fusion.imu.IMUSample`: `timestamp`, `accel` (3-axis, m/s^2, wrist
frame), `gyro` (3-axis, rad/s, wrist frame). The wristband samples at
100-200+ Hz -- much faster than a camera's 30-60 fps -- so it fills in
motion detail between camera frames and bridges short occlusion gaps
(see `docs/SENSOR_FUSION.md`).

## What's real software today

- **`irix.identity.checkout.CheckoutRegistry`** -- the front-desk step:
  check a wristband out to a member account, check it back in, resolve a
  wristband id to a member id. One active checkout per physical band at
  a time. Real and complete.
- **`irix.identity.ble_pairing.StationPairing`** -- resolves which
  station a wristband's BLE readings say it's nearest to (RSSI, with a
  motion-correlated tiebreak) -- the *resolution logic*, not a BLE radio
  stack.
- **`irix.fusion.imu_io`** -- loads an already-recorded wristband export
  (CSV/JSON; see that module for the exact format) into `IMUSample`s, for
  offline analysis of a finished workout.
- **`irix.fusion.imu_stream.IMUStream`** -- the `poll()`-based protocol
  a live consumer (`RepSession`, `StationSessionRunner`) uses regardless
  of whether samples are already-recorded or arriving live.
  `RecordedIMUStream` (real) wraps an already-loaded batch;
  `LiveBLEIMUStream` is a documented, deliberately unimplemented stub
  (see "What's a hardware decision, not a software gap" below).
- **`irix.wristband_sim`** (added 2026-07-14) -- the software-only
  simulator that stands in for real hardware at exactly that seam. See
  "Simulator" below.

## Simulator

`irix.wristband_sim.simulator.SimulatedBLEGateway` +
`SimulatedWristband` generate the same shapes a real gateway would --
`BLEReading`s (via `ble_reader()`) and `IMUSample`s (via
`SimulatedBLEIMUStream`, implementing `IMUStream`) -- with configurable
BLE packet loss and a scriptable `disconnect(wristband_id, ticks)` total
dropout, so `StationSessionRunner`/`GymSessionRunner` can be driven
end-to-end (multiple concurrent wristbands, station handoff, a radio
dropout and recovery) without any real hardware. See
`irix/demo/run_live_gym_demo.py` for the runnable demo and
`docs/ARCHITECTURE.md`'s "Software wristband + BLE gateway simulator"
section for the full design rationale.

This directly satisfies the founding brief's "Implement simulator" /
"Implement BLE gateway" / "Implement packet replay" items in pure
software: "packet replay" specifically is `irix.fusion.imu_io` (load a
real recorded export) plus `RecordedIMUStream` (replay it through the
live-consumer interface); the gateway/disconnect pieces are new.

## Calibration

`irix.wristband_sim.calibration.calibrate_stationary(samples) ->
IMUCalibration` -- standard strapdown-IMU static calibration: gyro bias
= mean gyro during a stationary period (true angular velocity is exactly
zero then), accel bias = mean accel minus expected gravity along
whichever axis is "up" while resting. `apply_calibration`/
`apply_calibration_batch` subtract it back out before samples reach
`irix.fusion`. Deliberately bias-only, not a full multi-orientation
scale-factor/misalignment calibration (needs a turntable or several
known orientations) -- unnecessary precision for rep counting rather
than dead-reckoning navigation.

**Wired in (Phase 3), but only at one specific moment**:
`irix.identity.placement.WristbandPlacementTracker` calls it every time a
band settles after a placement change (see "Placement" below) -- that's
a real, live entry point now, not just available-but-unused. What's
still **not** wired: an initial calibration at ordinary session/checkout
start (a band that's never had its placement explicitly changed keeps
using raw, uncalibrated samples for its whole session) -- `run_upload.py`
and a fresh `StationSessionRunner` session both still consume raw
samples from the moment a session opens; see `docs/TODO.md`.

## Placement

`irix.identity.placement.WristbandPlacementTracker` -- the real-time
state machine for *where the band is currently worn* (`BandSide`:
`left_wrist` / `right_wrist` / `left_ankle` / `right_ankle` / `unknown`),
as opposed to `irix.rep_counting.exercises.ExerciseConfig.
band_placement`'s static wrist-vs-ankle *requirement* per exercise, and
`irix.pipeline.events.BandPlacementTracker`'s top-down "the next exercise
needs a different placement" signal (`BandPlacementRequiredEvent`).

By exercise (Phase 3, per the founding brief's explicit placement
guidance): squat, hack squat, lunge, Bulgarian split squat, and calf
raise keep the band on the wrist -- feet stay in continuous ground
contact for all five, so the camera is the primary lower-body kinematics
source and the wrist IMU's role is sync/motion-onset/periodicity/
identity/tempo/set-boundaries, not primary rep counting. Leg press, leg
extension, leg curl, hip abduction, and hip adduction move the band to
the ankle -- true machine-footplate exercises where the foot is the
rigid contact point with the resisted load, the same relationship the
wrist has to a curl.

`irix.live.station_runner.StationSessionRunner.
request_wristband_placement_change(wristband_id, to_side, at_time)` is
the backend entry point a future IRIX app or front-desk console calls
once a member has actually moved their band (this repo deliberately does
not build that app). From that call until the tracker confirms the new
side (`SETTLING` -> `CALIBRATING` -> `STABLE`, `calibrate_stationary`
above providing the recalibration step, with the "up" axis estimated
from the settled window's own data rather than assumed, since a band's
rest orientation differs by which side/limb it's on), `RepSession.
add_imu_samples` drops every incoming batch instead of fusing it --
covering both the fastening-motion transition itself and any exercise in
the meantime whose required limb type the band isn't confirmed at yet
(**never reusing wrist thresholds for ankle data or vice versa**). A
`BandPlacementConfirmedEvent` (`docs/API_SPEC.md`) fires the moment a
change is confirmed.

## What's a hardware decision, not a software gap

`irix.fusion.imu_stream.LiveBLEIMUStream` stays an unimplemented stub on
purpose: which BLE GATT client library (e.g. `bleak`) and which
notify-characteristic protocol a real wristband firmware exposes is a
hardware/firmware decision this software scaffold cannot correctly guess
at. What's settled is the interface (`IMUStream.poll()`) a real
implementation must satisfy, so nothing downstream (`RepSession`,
`RepCountFusion`) needs to change once real hardware exists.

## Hardware recommendation (research-based, not yet built)

No firmware exists in this repo (firmware is out of a software repo's
scope per the founding brief) -- but a concrete, realistic platform
recommendation, so a hardware decision doesn't start from zero:

- **BLE SoC**: Nordic nRF52 or nRF54 series -- the de facto standard for
  low-power BLE wearables, with mature GATT/notify support and wide
  firmware tooling (Zephyr RTOS, Nordic's own SDK). Battery life for a
  gym-visit-duration (1-2 hour) continuous-broadcast use case is not a
  binding constraint the way it would be for a multi-day wearable.
- **IMU**: a 6-DOF (accel+gyro) MEMS IMU in the InvenSense ICM-42xxx or
  Bosch BMI2xx class -- both are common, well-documented choices with
  I2C/SPI interfaces any of the above SoCs support natively, and
  sufficient sample-rate headroom (typically up to 1-8 kHz internally,
  far above the 100-200 Hz this repo's fusion code expects) to
  comfortably hit target rates with onboard low-pass filtering.
- **BLE client library (edge-box side)**: `bleak` -- cross-platform
  (Linux/Windows/macOS), actively maintained, the natural choice for
  whatever process implements a real `LiveBLEIMUStream` subclass on an
  edge box's BLE receiver.
- **Identity resolution**: BLE RSSI proximity (current design,
  `irix.identity.ble_pairing`) is the v1 approach, with an explicitly
  documented upgrade path to BLE Angle-of-Arrival or UWB anchors per
  station if RSSI-based false pairings become a practical problem (see
  that module's docstring). Not revisited here since no field data
  exists yet to justify the added hardware cost.

None of the above is implemented or vendor-committed in this repo --
it's a starting point for whoever makes the real hardware purchasing
decision, not a dependency of any code here.
