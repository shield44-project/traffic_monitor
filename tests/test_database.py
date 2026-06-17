import config
from database import db


def test_database_initialization_and_summary(tmp_path, monkeypatch):
    db.close_connection()
    monkeypatch.setattr(config, "DATABASE_PATH", tmp_path / "test.db")

    db.init_db(seed_admin=False)
    db.insert_vehicle_counts({"car": 3, "bus": 1})
    db.insert_traffic_data(total_count=4, density=20, score=22, level="Low", avg_speed=50)
    db.insert_emission(
        co2=120, co=4, nox=3, pm25=1.2, pm10=1.8, hc=0.3, voc=0.2,
        so2=0.01, ch4=0.02, n2o=0.01, co2e=123.2,
        score=12, category="Good",
        gas_risk={"pm25": 80},
        vehicle_breakdown={"car": {"count": 1, "co2e": 123.2}},
    )

    summary = db.fetch_summary()

    assert summary["total_vehicles"] == 4
    assert summary["current_congestion"]["congestion_level"] == "Low"
    assert summary["current_emission"]["category"] == "Good"
    assert db.fetch_recent_emissions(1)[0]["vehicle_breakdown"]["car"]["count"] == 1

    db.close_connection()
