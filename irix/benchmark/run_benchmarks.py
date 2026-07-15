"""Benchmark suite entrypoint: ``python -m irix.benchmark.run_benchmarks``.

Measures wall-clock latency/throughput for every pure-software subsystem
in this repo (pose tracking, exercise recognition, sensor fusion, clock
sync, identity/motion-correlation resolution, per-frame event-producing
pipeline latency, the full simulated live-gym pipeline), plus
camera-reconnect timing, BLE-disconnect-recovery timing, BLE packet-loss
degradation behavior, CPU time, and peak memory -- using only the Python
standard library (``time``, ``resource``) so this runs anywhere this
repo's core dependencies already run, no extra install required.

**What's honestly not measured here, and why:** GPU utilization and real
pose-inference FPS need ``ultralytics``/``torch`` and (for GPU numbers) an
actual NVIDIA GPU -- neither is available in this sandboxed environment
(disk-constrained; see ``docs/DEPLOYMENT.md``). Rather than fabricate a
plausible-looking number, this script detects their absence and reports
``None`` with a clear reason, exactly like the honest-"unknown"-over-
fabrication principle every algorithm in this repo already follows for
its predictions. If ``ultralytics``/CUDA *are* available (a real edge-
device run), the relevant benchmarks below automatically run for real
instead of reporting "unavailable" -- see ``_pose_inference_available()``/
``_gpu_available()``.
"""
from __future__ import annotations

import json
import math
import platform
import resource
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TimingResult:
    name: str
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float
    fps: float
    n_iterations: int

    def to_dict(self) -> dict:
        return {
            "name": self.name, "mean_ms": round(self.mean_ms, 4), "p50_ms": round(self.p50_ms, 4),
            "p95_ms": round(self.p95_ms, 4), "max_ms": round(self.max_ms, 4),
            "fps": round(self.fps, 1) if math.isfinite(self.fps) else None,
            "n_iterations": self.n_iterations,
        }


def _time_it(fn: Callable[[], None], n: int = 200, warmup: int = 10) -> TimingResult:
    for _ in range(warmup):
        fn()
    times_s = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        fn()
        times_s[i] = time.perf_counter() - t0
    times_ms = times_s * 1000.0
    mean_s = float(times_s.mean())
    return TimingResult(
        name=fn.__name__ if hasattr(fn, "__name__") else "benchmark",
        mean_ms=float(times_ms.mean()), p50_ms=float(np.percentile(times_ms, 50)),
        p95_ms=float(np.percentile(times_ms, 95)), max_ms=float(times_ms.max()),
        fps=(1.0 / mean_s) if mean_s > 0 else float("inf"), n_iterations=n,
    )


def _pose_inference_available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False


def _gpu_available() -> Optional[Dict]:
    """Returns a dict with GPU info if ``nvidia-smi`` is present and
    reports at least one device, else ``None``."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        if not out:
            return None
        name, util, mem_used, mem_total = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"name": name, "utilization_pct": float(util), "memory_used_mb": float(mem_used), "memory_total_mb": float(mem_total)}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Individual subsystem benchmarks
# ---------------------------------------------------------------------------

def benchmark_pose_tracker(n_people: int = 3) -> TimingResult:
    from irix.pose.estimator import Keypoint, PersonPose
    from irix.pose.tracker import PoseTracker

    tracker = PoseTracker()
    t = [0.0]

    def _people():
        return [
            PersonPose(
                keypoints=[Keypoint(x=50.0 + i * 200, y=50.0, confidence=0.9) for _ in range(17)],
                bbox=(i * 200.0, 0.0, i * 200.0 + 100.0, 200.0),
            )
            for i in range(n_people)
        ]

    def _tick():
        t[0] += 1.0 / 30.0
        tracker.update(_people(), now=t[0])

    result = _time_it(_tick, n=300)
    result.name = f"pose_tracker.update (n_people={n_people})"
    return result


def benchmark_exercise_recognition() -> TimingResult:
    from irix.demo.mock_pose import synthetic_pose_stream
    from irix.exercise_recognition import recognize_exercise
    from irix.rep_counting.exercises import SQUAT

    poses = [pose for _, _, pose in synthetic_pose_stream(SQUAT, n_frames=90, reps_per_second=0.5)]

    def _run():
        recognize_exercise(poses)

    result = _time_it(_run, n=100)
    result.name = "exercise_recognition.recognize_exercise (90-frame window, 6 candidates)"
    return result


def benchmark_rep_fusion() -> TimingResult:
    from irix.demo.mock_pose import synthetic_imu_stream
    from irix.fusion.rep_fusion import RepCountFusion

    fusion = RepCountFusion()
    samples = synthetic_imu_stream(n_seconds=16.0, reps_per_second=0.5, seed=1)

    def _run():
        fusion.fuse(camera_count=8, camera_confidence=0.9, imu_samples=samples, camera_rep_durations=[2.0] * 8)

    result = _time_it(_run, n=100)
    result.name = "rep_fusion.fuse (16s / 1600-sample set)"
    return result


def benchmark_ekf() -> TimingResult:
    from irix.fusion.ekf import VisualInertialEKF

    def _run():
        ekf = VisualInertialEKF()
        t = 0.0
        for i in range(160):  # ~1.6s @ 100Hz IMU
            t += 0.01
            ekf.predict(accel=0.5, timestamp=t)
            if i % 3 == 0:
                ekf.update(measured_position=0.01 * i)

    result = _time_it(_run, n=100)
    result.name = "fusion.ekf (160-sample predict/update cycle)"
    return result


def benchmark_clock_sync() -> TimingResult:
    from irix.fusion.clock_sync import estimate_offset_via_cross_correlation

    ref_t = np.linspace(0, 10, 500)
    ref_signal = np.sin(2 * np.pi * 0.5 * ref_t)
    target_t = ref_t - 0.2
    target_signal = ref_signal.copy()

    def _run():
        estimate_offset_via_cross_correlation(ref_t, ref_signal, target_t, target_signal)

    result = _time_it(_run, n=100)
    result.name = "clock_sync.estimate_offset_via_cross_correlation (500-sample, 10s window)"
    return result


def benchmark_identity_resolution_latency(n_candidates: int = 3, window_frames: int = 60) -> TimingResult:
    """Priority 9's "identity latency" -- how long
    ``irix.identity.motion_correlation.MotionCorrelationResolver.resolve``
    takes to disambiguate a crowded station once its buffering window is
    full (the actual per-resolution cost; the buffering wait itself is a
    fixed, configured window duration, not a compute cost -- see
    ``irix.live.disambiguation.CrowdedGroupDisambiguator``). Scales with
    ``n_candidates`` (default 3, this repo's documented "small number of
    co-located members" design point -- see that resolver's own module
    docstring) and the window length."""
    from irix.demo.mock_pose import synthetic_imu_stream, synthetic_pose_stream
    from irix.identity.motion_correlation import MotionCorrelationResolver
    from irix.rep_counting.exercises import SQUAT

    resolver = MotionCorrelationResolver()
    candidate_imu_streams = {
        f"member-{i}": synthetic_imu_stream(n_seconds=window_frames / 30.0, reps_per_second=0.4 + i * 0.05, seed=i)
        for i in range(n_candidates)
    }
    detected_people_poses = [
        [p for _, _, p in synthetic_pose_stream(SQUAT, n_frames=window_frames, reps_per_second=0.4 + i * 0.05)]
        for i in range(n_candidates)
    ]

    def _run():
        resolver.resolve(candidate_imu_streams, detected_people_poses, pose_fps=30.0)

    result = _time_it(_run, n=50)
    result.name = f"identity.MotionCorrelationResolver.resolve ({n_candidates} candidates, {window_frames}-frame window)"
    return result


def benchmark_event_latency() -> TimingResult:
    """Priority 9's "event latency" -- wall-clock time from one
    ``RepSession.process_frame`` call (a frame + pose arriving) to
    whatever events it produces being returned, at the point in a set
    where a rep actually completes (the most expensive frame -- also
    exercises FormScorer/fatigue-sample construction, unlike a
    no-op mid-rep frame)."""
    import math as _math

    import numpy as _np

    from irix.pipeline.rep_session import RepSession
    from irix.pose.estimator import COCO_KEYPOINT_NAMES, KEYPOINT_INDEX, Keypoint, PersonPose
    from irix.rep_counting.exercises import SQUAT

    def _pose_for_angle(angle_deg: float) -> PersonPose:
        knee = _np.array([0.0, 0.0])
        hip = knee + _np.array([0.0, -100.0])
        theta = _math.radians(-90 + angle_deg)
        ankle = knee + 100.0 * _np.array([_math.cos(theta), _math.sin(theta)])
        keypoints = [Keypoint(x=0.0, y=0.0, confidence=0.0) for _ in COCO_KEYPOINT_NAMES]

        def _set(name, xy):
            keypoints[KEYPOINT_INDEX[name]] = Keypoint(x=float(xy[0]), y=float(xy[1]), confidence=0.9)

        _set("left_hip", hip)
        _set("left_knee", knee)
        _set("left_ankle", ankle)
        return PersonPose(keypoints=keypoints, bbox=(0.0, 0.0, 200.0, 200.0))

    session = RepSession(exercise_name="squat", member_id="bench-member", station_id="bench-station")
    angles = list(_np.linspace(90.0, 170.0, 10))  # one full rep's worth of frames
    frame = _np.zeros((2, 2, 3), dtype=_np.uint8)
    ts_counter = [0.0]

    def _run():
        for angle in angles:
            ts_counter[0] += 1.0 / 30.0
            session.process_frame(frame=frame, ts=ts_counter[0], person=_pose_for_angle(angle))

    result = _time_it(_run, n=30, warmup=3)
    result.name = "pipeline.RepSession.process_frame (per rep-completing frame batch)"
    return result


def benchmark_live_gym_pipeline(n_ticks: int = 260) -> Dict:
    """The most representative "processing FPS" number available without
    real camera/pose-model hardware: the full simulated live pipeline
    (BLE gateway -> presence resolution -> pose tracking input ->
    RepSession -> fusion -> fatigue -> event emission) end to end, the
    same code path ``irix/demo/run_live_gym_demo.py`` exercises."""
    from irix.demo.run_live_gym_demo import run

    t0 = time.perf_counter()
    events = run(n_ticks=n_ticks, seed=7, verbose=False)
    elapsed_s = time.perf_counter() - t0

    return {
        "n_ticks": n_ticks, "elapsed_s": round(elapsed_s, 4),
        "ticks_per_second": round(n_ticks / elapsed_s, 1) if elapsed_s > 0 else None,
        "ms_per_tick": round((elapsed_s / n_ticks) * 1000.0, 4) if n_ticks > 0 else None,
        "n_events_produced": len(events),
    }


def benchmark_packet_loss_impact(loss_levels=(0.0, 0.1, 0.3, 0.5)) -> Dict:
    """Priority 9's "packet loss" -- not a timing number, a *behavioral*
    one: how ``irix.fusion.rep_fusion.RepCountFusion`` actually degrades
    as a wristband's BLE packet loss rate rises, using the same
    ``imu_sample_completeness`` signal that fusion's own confidence
    discount is based on (``irix.fusion.rep_fusion``'s module docstring).
    Confirms graceful degradation (completeness drops roughly with the
    loss rate; fusion never silently keeps trusting the IMU count as if
    nothing were missing) rather than a hard failure at some threshold.
    """
    from irix.demo.mock_pose import synthetic_imu_stream
    from irix.fusion.rep_fusion import RepCountFusion

    rng = np.random.default_rng(11)
    base_samples = synthetic_imu_stream(n_seconds=16.0, reps_per_second=0.5, seed=3)
    fusion = RepCountFusion()

    results = []
    for loss_pct in loss_levels:
        if loss_pct <= 0.0:
            kept = base_samples
        else:
            keep_mask = rng.random(len(base_samples)) >= loss_pct
            kept = [s for s, keep in zip(base_samples, keep_mask) if keep]
        fused = fusion.fuse(
            camera_count=8, camera_confidence=0.9, imu_samples=kept,
            camera_rep_durations=[2.0] * 8,
        )
        results.append({
            "packet_loss_pct": loss_pct,
            "n_samples_kept": len(kept),
            "n_samples_expected": len(base_samples),
            "imu_sample_completeness": fused.imu_sample_completeness,
            "fused_rep_count": fused.fused_count,
            "rep_count_source": fused.source,
            "agreement": fused.agreement,
        })

    return {"loss_levels": list(loss_levels), "results": results}


def benchmark_camera_reconnect_schedule(backoff_s: float = 2.0, max_backoff_s: float = 30.0, n_failures: int = 6) -> Dict:
    """``ReconnectingFrameSource``'s backoff is a deterministic
    exponential schedule -- compute it directly (accurate, and doesn't
    require actually sleeping through a multi-minute benchmark run) and
    separately verify via a real (sleep-mocked) run that the class
    actually reconnects successfully after ``n_failures`` failed opens."""
    from irix.live.camera_source import ReconnectingFrameSource

    schedule = []
    backoff = backoff_s
    for _ in range(n_failures):
        schedule.append(round(backoff, 2))
        backoff = min(backoff * 2.0, max_backoff_s)

    class _FlakyCapture:
        _fail_remaining = n_failures

        def isOpened(self):
            return True

        def read(self):
            if _FlakyCapture._fail_remaining > 0:
                _FlakyCapture._fail_remaining -= 1
                return False, None
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def release(self):
            pass

    sleep_calls = []
    source = ReconnectingFrameSource(
        source="fake", backoff_s=backoff_s, max_backoff_s=max_backoff_s,
        capture_factory=lambda _src: _FlakyCapture(),
    )
    frames = list(source.frames(max_frames=1, sleep=lambda s: sleep_calls.append(s)))

    return {
        "backoff_schedule_s": schedule,
        "total_backoff_wait_s": round(sum(schedule), 2),
        "n_reconnect_attempts_before_success": len(sleep_calls),
        "recovered_successfully": len(frames) == 1,
    }


def benchmark_ble_disconnect_recovery() -> Dict:
    """How much of a station's ``presence_timeout_s`` grace period a
    scripted disconnect actually consumes before the band's BLE presence
    resumes -- the real number ``irix/demo/run_live_gym_demo.py``'s
    scripted 10-tick disconnect (at 30 ticks/s) exercises, extracted here
    as its own reusable measurement rather than only visible buried in
    that demo's console output."""
    from irix.wristband_sim.simulator import SimulatedBLEGateway, SimulatedWristband

    gateway = SimulatedBLEGateway(packet_loss_pct=0.0, seed=0)
    band = SimulatedWristband("band-1", seed=0)
    gateway.add_wristband(band)
    gateway.move_to_station("band-1", "squat-1")

    tick_hz = 30.0
    disconnect_ticks = 10
    gateway.disconnect("band-1", ticks=disconnect_ticks)

    ticks_without_presence = 0
    for i in range(disconnect_ticks + 2):
        gateway.tick(now=i / tick_hz)
        if not gateway.ble_reader():
            ticks_without_presence += 1

    outage_s = ticks_without_presence / tick_hz
    presence_timeout_s = 1.0  # matches run_live_gym_demo.py's configured value
    return {
        "outage_ticks": ticks_without_presence, "outage_s": round(outage_s, 3),
        "presence_timeout_s": presence_timeout_s,
        "recovery_margin_s": round(presence_timeout_s - outage_s, 3),
        "session_survives": outage_s < presence_timeout_s,
    }


def benchmark_pose_inference() -> Optional[Dict]:
    """Real pose-inference latency/FPS against the actual pretrained
    YOLO-Pose checkpoint -- only runs if ``ultralytics`` is installed
    (not the case in this sandboxed environment; see module docstring).
    """
    if not _pose_inference_available():
        return None
    from irix.pose.estimator import PoseEstimator

    estimator = PoseEstimator()
    frame = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)

    def _run():
        estimator.estimate(frame)

    result = _time_it(_run, n=30, warmup=3)
    return result.to_dict()


def resource_usage() -> Dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "peak_rss_mb": round(usage.ru_maxrss / 1024.0, 1),  # ru_maxrss is KB on Linux
        "user_cpu_s": round(usage.ru_utime, 3),
        "system_cpu_s": round(usage.ru_stime, 3),
    }


def run_all() -> Dict:
    started = time.perf_counter()
    cpu_before = resource.getrusage(resource.RUSAGE_SELF)

    report = {
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": __import__("os").cpu_count(),
            "pose_inference_available": _pose_inference_available(),
            "gpu": _gpu_available(),
        },
        "timing_benchmarks": [
            benchmark_pose_tracker(n_people=1).to_dict(),
            benchmark_pose_tracker(n_people=3).to_dict(),
            benchmark_exercise_recognition().to_dict(),
            benchmark_rep_fusion().to_dict(),
            benchmark_ekf().to_dict(),
            benchmark_clock_sync().to_dict(),
            benchmark_identity_resolution_latency().to_dict(),
            benchmark_event_latency().to_dict(),
        ],
        "pose_inference": benchmark_pose_inference(),
        "live_pipeline_throughput": benchmark_live_gym_pipeline(n_ticks=260),
        "camera_reconnect": benchmark_camera_reconnect_schedule(),
        "ble_disconnect_recovery": benchmark_ble_disconnect_recovery(),
        "packet_loss_impact": benchmark_packet_loss_impact(),
    }

    cpu_after = resource.getrusage(resource.RUSAGE_SELF)
    elapsed_s = time.perf_counter() - started
    cpu_delta_s = (cpu_after.ru_utime - cpu_before.ru_utime) + (cpu_after.ru_stime - cpu_before.ru_stime)
    report["resource_usage"] = {
        **resource_usage(),
        "benchmark_wall_time_s": round(elapsed_s, 3),
        "benchmark_cpu_time_s": round(cpu_delta_s, 3),
        "approx_cpu_utilization_pct": round(100.0 * cpu_delta_s / elapsed_s, 1) if elapsed_s > 0 else None,
    }
    return report


def format_report(report: Dict) -> str:
    lines = ["IRIX benchmark report", "=" * 40, ""]
    env = report["environment"]
    lines.append(f"platform: {env['platform']}")
    lines.append(f"python: {env['python_version']}, cpu_count: {env['cpu_count']}")
    lines.append(f"pose inference (ultralytics) available: {env['pose_inference_available']}")
    lines.append(f"GPU: {env['gpu'] if env['gpu'] else 'not available in this environment'}")
    lines.append("")
    lines.append("Timing benchmarks (pure-software subsystems):")
    for b in report["timing_benchmarks"]:
        lines.append(f"  {b['name']}: {b['mean_ms']} ms mean, {b['p95_ms']} ms p95, {b['fps']} calls/s")
    lines.append("")
    if report["pose_inference"] is not None:
        p = report["pose_inference"]
        lines.append(f"Pose inference: {p['mean_ms']} ms mean ({p['fps']} FPS)")
    else:
        lines.append("Pose inference: not available (ultralytics not installed in this environment)")
    lines.append("")
    lp = report["live_pipeline_throughput"]
    lines.append(
        f"Full simulated live pipeline: {lp['ticks_per_second']} ticks/s "
        f"({lp['ms_per_tick']} ms/tick, {lp['n_events_produced']} events over {lp['n_ticks']} ticks)"
    )
    lines.append("")
    cr = report["camera_reconnect"]
    lines.append(
        f"Camera reconnect: schedule {cr['backoff_schedule_s']}s, "
        f"total wait {cr['total_backoff_wait_s']}s across {cr['n_reconnect_attempts_before_success']} attempts, "
        f"recovered={cr['recovered_successfully']}"
    )
    ble = report["ble_disconnect_recovery"]
    lines.append(
        f"BLE disconnect recovery: {ble['outage_s']}s outage vs {ble['presence_timeout_s']}s timeout "
        f"(margin {ble['recovery_margin_s']}s, survives={ble['session_survives']})"
    )
    lines.append("")
    pl = report["packet_loss_impact"]
    lines.append("Packet loss impact on IMU/camera fusion:")
    for r in pl["results"]:
        lines.append(
            f"  {int(r['packet_loss_pct'] * 100)}% loss: completeness={r['imu_sample_completeness']}, "
            f"source={r['rep_count_source']}, agreement={r['agreement']}"
        )
    lines.append("")
    ru = report["resource_usage"]
    lines.append(
        f"Resource usage: peak RSS {ru['peak_rss_mb']} MB, "
        f"CPU time {ru['benchmark_cpu_time_s']}s over {ru['benchmark_wall_time_s']}s wall "
        f"(~{ru['approx_cpu_utilization_pct']}% of one core)"
    )
    return "\n".join(lines)


def main():
    report = run_all()
    print(format_report(report))
    return report


if __name__ == "__main__":
    main()
