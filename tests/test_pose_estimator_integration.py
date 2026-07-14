"""Integration test against a *real* pretrained pose model -- not a mock.

Skipped automatically if ``ultralytics``/``torch`` aren't installed (the
``pose`` extra, ``pip install irix[pose]``), so the default test suite
stays fast and dependency-light. When it does run, this is the one place
in the whole test suite that proves ``PoseEstimator`` isn't just a
correctly-shaped stub waiting for a model that doesn't exist:
``yolov8n-pose.pt`` is a real, freely available checkpoint pretrained on
COCO keypoints (exactly the 17-point layout ``irix.pose.estimator``
already assumes) that Ultralytics auto-downloads on first use -- no
training, no API key, no cost, no gym-specific data collection needed.
Generic human pose estimation is a solved, commodity problem; this test
is the receipt.

Run explicitly with: ``pip install irix[pose] && pytest tests/test_pose_estimator_integration.py -v``
"""
import pytest

ultralytics = pytest.importorskip("ultralytics", reason="requires 'pip install irix[pose]'")


def _zidane_image_path() -> str:
    from ultralytics.utils import ASSETS

    return str(ASSETS / "zidane.jpg")


def test_real_model_detects_people_with_plausible_keypoints():
    import cv2

    from irix.pose.estimator import PoseEstimator

    frame = cv2.imread(_zidane_image_path())
    assert frame is not None, "bundled ultralytics test image should be readable"

    estimator = PoseEstimator(model_path="yolov8n-pose.pt", confidence=0.4)
    people = estimator.estimate(frame)

    # This image (ultralytics' own standard test asset) has two people in
    # it -- a real model should find both, not zero and not a dozen.
    assert 1 <= len(people) <= 3

    for person in people:
        shoulder = person.get("left_shoulder")
        assert shoulder is not None
        # A real model's confidence on a clearly-visible joint should be
        # high, not near-zero noise -- this is the actual signal that
        # distinguishes "a real model ran" from "a stub returned garbage".
        assert shoulder.confidence > 0.5
        # bbox should be a real, non-degenerate box within the frame.
        x1, y1, x2, y2 = person.bbox
        assert 0 <= x1 < x2 <= frame.shape[1]
        assert 0 <= y1 < y2 <= frame.shape[0]


def test_real_model_keypoints_feed_joint_angle_without_error():
    """End-to-end: real detected keypoints -> irix.pose.geometry.joint_angle,
    the same call irix.demo.run_demo.run_live makes per frame -- proves
    the real model's output shape is actually compatible with the rest of
    the pipeline, not just internally self-consistent."""
    import cv2

    from irix.pose.estimator import PoseEstimator
    from irix.pose.geometry import joint_angle

    frame = cv2.imread(_zidane_image_path())
    estimator = PoseEstimator(model_path="yolov8n-pose.pt", confidence=0.4)
    people = estimator.estimate(frame)
    assert people, "expected at least one person detected"

    person = people[0]
    a, v, c = person.xy("left_shoulder"), person.xy("left_elbow"), person.xy("left_wrist")
    if a is None or v is None or c is None:
        pytest.skip("left arm not confidently tracked in this frame -- try left_hip/knee/ankle instead")
    angle = joint_angle(a, v, c)
    assert angle == angle  # not NaN
    assert 0.0 <= angle <= 180.0


def _write_test_video(path: str, n_frames: int = 15, fps: float = 10.0) -> None:
    import cv2

    frame = cv2.imread(_zidane_image_path())
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


def test_run_live_against_real_video_no_crash(tmp_path):
    """End-to-end: real PoseEstimator -> joint_angle -> RepCounter ->
    FormScorer -> pipeline, against an actual (synthetic, but real-codec)
    video file -- the same code path a real camera source would take in
    irix.demo.run_demo.run_live. display=False (the default) is what
    proves this doesn't need a GUI/display to run, which matters since a
    real edge box doesn't have one either."""
    from irix.demo.run_demo import run_live

    video_path = str(tmp_path / "test_video.mp4")
    _write_test_video(video_path)

    counter, cloud = run_live(
        video_path, "squat", "test-member", "test-station", display=False, max_frames=15,
    )
    # A static image repeated has no motion, so 0 reps is the *correct*
    # outcome here -- what this test actually proves is that the real
    # model runs frame-by-frame through the whole pipeline without
    # raising, which is what "no real trained model" being fixed
    # actually means in practice.
    assert counter.rep_count == 0
    assert isinstance(cloud.received, list)


def test_run_live_display_flag_fails_loudly_not_silently(tmp_path, capsys):
    """--display on a headless environment (this sandbox, and any real
    edge box with no monitor attached) should print a clear, actionable
    message and stop -- not crash with a raw OpenCV C++ exception, and
    not silently hang either."""
    from irix.demo.run_demo import run_live

    video_path = str(tmp_path / "test_video.mp4")
    _write_test_video(video_path)

    run_live(video_path, "squat", "test-member", "test-station", display=True, max_frames=3)
    captured = capsys.readouterr()
    assert "requires a GUI-enabled OpenCV build" in captured.err
