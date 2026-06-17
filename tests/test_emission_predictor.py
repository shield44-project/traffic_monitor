from prediction.emission_predictor import EmissionPredictor


def test_formula_fallback_returns_non_negative_estimate():
    predictor = EmissionPredictor(model_paths={"co2": "missing", "nox": "missing", "pm25": "missing"})
    estimate = predictor.predict(
        {"car": 10, "truck": 2, "bus": 1, "motorcycle": 3, "bicycle": 1},
        density=45,
        avg_speed=28,
    )

    assert estimate.co2 >= 0
    assert estimate.nox >= 0
    assert estimate.pm25 >= 0
    assert estimate.pm10 >= 0
    assert estimate.co >= 0
    assert estimate.co2e >= estimate.co2
    assert "car" in estimate.vehicle_breakdown
    assert "pm25" in estimate.gas_risk
    assert 0 <= estimate.emission_score <= 100
    assert estimate.source == "factor_table"
