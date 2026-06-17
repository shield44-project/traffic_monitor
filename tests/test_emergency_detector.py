import pytest

from core.exceptions import ModelLoadError
from detection.emergency_detector import EmergencyDetector


def test_emergency_detector_rejects_generic_vehicle_checkpoint():
    detector = EmergencyDetector(model_path="missing.pt")
    detector._model_names = {0: "Vehicle"}

    with pytest.raises(ModelLoadError):
        detector._validate_model_labels()


def test_emergency_detector_accepts_expected_labels():
    detector = EmergencyDetector(model_path="missing.pt")
    detector._model_names = {0: "ambulance", 1: "fire_truck", 2: "police"}

    detector._validate_model_labels()
