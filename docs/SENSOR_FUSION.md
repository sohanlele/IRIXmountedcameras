# Sensor fusion (camera + wristband IMU)

## Visual-inertial fusion

`irix.fusion.ekf` -- an Extended Kalman Filter over a position/velocity
state, fusing camera-tracked position with wristband accelerometer data.
`irix.fusion.zupt` -- Zero-velocity Update: detects genuine dead-stops
(e.g. the bottom of a rep) from the IMU signal and corrects accumulated
velocity/position drift at those points, the standard technique for
bounding integration drift in a strapdown-INS-style filter over a short
window (a rep, not a whole session).

## IMU-only rep counting

`irix.fusion.imu_rep_counting` ports two literature rep-counting
algorithms (RecoFit- and uLift-style amplitude-percentile filters) that
work from wristband IMU data alone, with no camera input -- used both as
an independent cross-check against the camera-based count and as the
fallback source when camera tracking is unreliable.

## Rep-count reconciliation, not a parallel cross-check

`irix.fusion.rep_fusion.RepCountFusion` is real fusion, not two
independent counts reported side by side: it reconciles the camera's
rep count and the IMU-only count into one authoritative
`fused_rep_count` per set, falling back toward the IMU count when the
camera's own `tracking_confidence` was low for that set (heavy
occlusion) rather than blindly trusting whichever source happened to run
first. `SetCompleteEvent` carries both raw counts plus the fused count
and `rep_count_agreement`/`rep_count_source` so a consumer can see
*why* the fused count is what it is, not just the number.

## Data rates and synchronization

Camera frames arrive at 30-60 fps; wristband IMU samples arrive at
100-200+ Hz -- a genuine rate mismatch, handled by timestamp-based
alignment throughout (`IMUSample.timestamp`, frame timestamps from
either `frame_index / fps` for an uploaded file or `time.monotonic()`
for a live source -- see `docs/ARCHITECTURE.md`'s "End-to-end demo"
section for why that distinction matters). There is currently **no
explicit clock-offset estimation/correction** between a wristband's own
onboard clock and the edge box's clock -- `irix.fusion.imu_io`/
`imu_stream` assume timestamps are already on a shared clock (true for
`RecordedIMUStream`/simulated data, not yet validated against real
wristband firmware's actual clock behavior). See `docs/TODO.md`.

## Where fusion output surfaces

`RepCompletedEvent`'s velocity fields are two-tier: `peak/mean_velocity_
deg_s` (camera joint-angle proxy, always available) and
`peak/mean_velocity_m_s`/`velocity_loss_pct`/`estimated_rpe` (calibrated,
only when a barbell/dumbbell is being tracked -- see
`irix.barbell.tracker`/`irix.barbell.rpe`). `SetCompleteEvent` carries
the reconciled rep count (above). `SetFatigueSummaryEvent`
(`irix.fatigue`) aggregates both into set/session-level fatigue trends.
See `docs/API_SPEC.md` for the full event reference.

## What's not built

Full 3D orientation estimation to strip gravity's changing projection
out of the raw wrist accelerometer signal (a Versatile Quaternion Filter
or similar) -- flagged honestly in `irix.identity.motion_correlation`'s
own docstring as the main accuracy lever left on the table for
motion-correlation disambiguation specifically; see
`docs/RESEARCH_LOG.md` for the citation. No cross-clock synchronization
protocol for real hardware (see "Data rates and synchronization" above).
