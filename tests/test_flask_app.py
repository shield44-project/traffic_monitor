import config
from app import app
from database import db


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
