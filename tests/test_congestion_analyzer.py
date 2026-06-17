from detection.congestion_analyzer import CongestionAnalyzer


def test_congestion_bands_and_density():
    analyzer = CongestionAnalyzer(saturation_count=20)
    result = analyzer.analyze({"car": 5, "truck": 2, "bus": 1})

    assert result.total_count == 8
    assert 0 < result.density <= 100
    assert result.level in {"Low", "Medium", "High", "Severe"}
    assert result.avg_speed >= 5


def test_empty_traffic_is_low():
    result = CongestionAnalyzer().analyze({})

    assert result.total_count == 0
    assert result.density == 0
    assert result.congestion_score == 0
    assert result.level == "Low"
