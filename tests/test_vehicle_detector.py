import numpy as np

from detection.vehicle_detector import Detection, VehicleDetector


def test_vehicle_detector_maps_custom_labels():
    detector = VehicleDetector(model_path="missing.pt")
    detector._model_names = {0: "mobil", 1: "motor", 2: "truk", 3: "vehicle"}

    assert detector._class_label(0) == "car"
    assert detector._class_label(1) == "motorcycle"
    assert detector._class_label(2) == "truck"
    assert detector._class_label(3) is None


def test_generic_vehicle_label_can_be_enabled(monkeypatch):
    monkeypatch.setattr("config.ALLOW_GENERIC_VEHICLE_CLASS", True)
    detector = VehicleDetector(model_path="missing.pt")
    detector._model_names = {0: "vehicle"}

    assert detector._class_label(0) == "vehicle"


def test_vehicle_box_filter_rejects_face_sized_full_frame_generic(monkeypatch):
    monkeypatch.setattr("config.ALLOW_GENERIC_VEHICLE_CLASS", True)

    assert not VehicleDetector._is_valid_box(
        "vehicle",
        0.95,
        (40, 10, 600, 470),
        width=640,
        height=480,
    )


def test_vehicle_box_filter_accepts_reasonable_car_box():
    assert VehicleDetector._is_valid_box(
        "car",
        0.90,
        (120, 220, 360, 350),
        width=640,
        height=480,
    )


def test_roi_density_counts_lanes():
    detector = VehicleDetector(model_path="missing.pt")
    detections = [
        Detection("car", 0.9, (120, 380, 180, 430), 2),
        Detection("truck", 0.8, (860, 380, 930, 450), 7),
    ]

    roi = detector._roi_density(detections, 1280, 720)

    assert roi["left_lane"]["count"] == 1
    assert roi["right_lane"]["count"] == 1
    assert roi["total_roi_vehicles"] == 2
    assert np.array(roi["polygons"]["left"]).shape == (4, 2)
