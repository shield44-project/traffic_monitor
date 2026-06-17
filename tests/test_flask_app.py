import config
from app import app, _safe_video_max_frames, analyze_video_file
from core.exceptions import ModelNotFoundError
from database import db
from detection.vehicle_detector import Detection, FrameAnalysis


def test_login_and_summary_api(tmp_path, monkeypatch):
    db.close_connection()
    monkeypatch.setattr(config, "DATABASE_PATH", tmp_path / "app.db")
    db.init_db(seed_admin=True)

    app.config.update(TESTING=True, SECRET_KEY="test")
    client = app.test_client()
    response = client.post(
        "/login",
        data={"username": config.ADMIN_USERNAME, "password": config.ADMIN_PASSWORD},
        follow_redirects=False,
    )
    assert response.status_code in {302, 303}

    response = client.get("/api/summary")
    assert response.status_code == 200
    assert "summary" in response.get_json()

    db.close_connection()


def test_safe_video_max_frames_allows_all_frames():
    assert _safe_video_max_frames("-1", 120) == -1
    assert _safe_video_max_frames("0", 120) == 1
    assert _safe_video_max_frames("5000", 120) == 1000


def test_video_analysis_reports_unique_tracked_counts(monkeypatch, tmp_path):
    import cv2
    import numpy as np

    video_path = tmp_path / "one_car.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (160, 120),
    )
    for _ in range(3):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    class FakeDetector:
        model_path = "fake.pt"
        model_labels = {0: "car"}
        model_warning = None

        def detect(self, frame, frame_index=0, tracker=None):
            return FrameAnalysis(
                counts={"car": 1, "motorcycle": 0, "bus": 0, "truck": 0, "bicycle": 0, "vehicle": 0},
                detections=[Detection("car", 0.9, (10, 10, 50, 50), 2, track_id=7)],
                tracks={7: (30, 30)},
                frame_index=frame_index,
                width=160,
                height=120,
                total_crossings=0,
                roi_density={
                    "left_lane": {"count": 1, "intensity": "Smooth", "heavy": False},
                    "right_lane": {"count": 0, "intensity": "Smooth", "heavy": False},
                },
            )

        def draw(self, frame, result):
            return frame

    monkeypatch.setattr("app.get_vehicle_detector", lambda: FakeDetector())
    monkeypatch.setattr("app.get_emergency_detector", lambda: (_ for _ in ()).throw(ModelNotFoundError("unused")))

    with app.test_request_context("/"):
        result = analyze_video_file(video_path, every_n_frames=1, max_frames=3, analysis_width=0)

    assert result["sampled_detection_totals"]["car"] == 3
    assert result["vehicle_totals"]["car"] == 1
    assert result["unique_vehicle_count"] == 1
    assert result["emission_totals"]["co2e"] > 0
