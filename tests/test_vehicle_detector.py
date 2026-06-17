import numpy as np

from detection.vehicle_detector import Detection, VehicleDetector


def test_vehicle_detector_maps_custom_labels():
    detector = VehicleDetector(model_path="missing.pt")
    detector._model_names = {0: "mobil", 1: "motor", 2: "truk", 3: "vehicle"}

    assert detector._class_label(0) == "car"
    assert detector._class_label(1) == "motorcycle"
    assert detector._class_label(2) == "truck"
    assert detector._class_label(3) == "vehicle"


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
