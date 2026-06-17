"""SQLite data-access layer.

Design notes
------------
* One connection per thread (``check_same_thread=False`` + a thread-local) so
  the Flask dev server and background capture threads can share the same DB
  file safely. WAL mode (set in schema.sql) allows concurrent readers.
* ``row_factory = sqlite3.Row`` so every row behaves like a dict.
* All writes use parameterised queries -> no SQL injection.
* High-level ``insert_*`` / ``fetch_*`` helpers keep SQL out of the rest of the
  codebase.
"""
from __future__ import annotations

import sqlite3
import threading
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import config
from core.exceptions import DatabaseError
from core.logger import get_logger

log = get_logger("database")

_local = threading.local()


def _now() -> str:
    """Current UTC time as an ISO-8601 string (seconds precision)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def get_connection() -> sqlite3.Connection:
    """Return a per-thread SQLite connection (created lazily)."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        try:
            conn = sqlite3.connect(
                config.DATABASE_PATH, check_same_thread=False, timeout=30.0
            )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not open database: {exc}") from exc
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        _local.conn = conn
    return conn


def close_connection() -> None:
    """Close the current thread's SQLite connection if one exists."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None


@contextmanager
def transaction():
    """Context manager yielding a cursor, committing on success."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def execute(query: str, params: Sequence[Any] = ()) -> int:
    """Run a write query; return the lastrowid."""
    try:
        with transaction() as cur:
            cur.execute(query, params)
            return cur.lastrowid
    except sqlite3.Error as exc:
        raise DatabaseError(f"Query failed: {exc}\nSQL: {query}") from exc


def query(query_str: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
    """Run a read query; return all rows."""
    try:
        cur = get_connection().execute(query_str, params)
        rows = cur.fetchall()
        cur.close()
        return rows
    except sqlite3.Error as exc:
        raise DatabaseError(f"Query failed: {exc}\nSQL: {query_str}") from exc


# ---------------------------------------------------------------------------
# Schema initialisation + seeding
# ---------------------------------------------------------------------------
def init_db(seed_admin: bool = True) -> None:
    """Create tables from schema.sql (idempotent) and seed an admin user."""
    if not config.SCHEMA_PATH.exists():
        raise DatabaseError(f"Schema file not found: {config.SCHEMA_PATH}")

    sql = config.SCHEMA_PATH.read_text(encoding="utf-8")
    conn = get_connection()
    conn.executescript(sql)
    conn.commit()
    _migrate_schema(conn)
    log.info("Database initialised at %s", config.DATABASE_PATH)

    if seed_admin:
        _seed_admin()
    _seed_default_camera()


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Apply small additive migrations for existing SQLite databases."""
    emission_cols = {row["name"] for row in conn.execute("PRAGMA table_info(emissions)")}
    additions = {
        "co": "REAL NOT NULL DEFAULT 0",
        "pm10": "REAL NOT NULL DEFAULT 0",
        "hc": "REAL NOT NULL DEFAULT 0",
        "voc": "REAL NOT NULL DEFAULT 0",
        "so2": "REAL NOT NULL DEFAULT 0",
        "ch4": "REAL NOT NULL DEFAULT 0",
        "n2o": "REAL NOT NULL DEFAULT 0",
        "co2e": "REAL NOT NULL DEFAULT 0",
        "gas_risk_json": "TEXT",
        "vehicle_breakdown_json": "TEXT",
    }
    for column, ddl in additions.items():
        if column not in emission_cols:
            conn.execute(f"ALTER TABLE emissions ADD COLUMN {column} {ddl}")
    conn.commit()


def _seed_admin() -> None:
    """Create the default admin account if the users table is empty."""
    # Imported here to avoid a circular import (web.auth imports db).
    from werkzeug.security import generate_password_hash

    rows = query("SELECT COUNT(*) AS n FROM users")
    if rows and rows[0]["n"] == 0:
        execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?,?,?)",
            (config.ADMIN_USERNAME,
             generate_password_hash(config.ADMIN_PASSWORD),
             "admin"),
        )
        log.info("Seeded default admin user '%s'", config.ADMIN_USERNAME)


def _seed_default_camera() -> None:
    rows = query("SELECT COUNT(*) AS n FROM cameras")
    if rows and rows[0]["n"] == 0:
        execute(
            "INSERT INTO cameras (name, source, location) VALUES (?,?,?)",
            ("Demo Camera", "samples/sample_traffic.mp4", "Main Junction"),
        )
        log.info("Seeded default demo camera")


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------
def insert_vehicle_counts(counts: dict[str, int], camera_id: int | None = None,
                          timestamp: str | None = None) -> None:
    """Insert one row per vehicle type from a {type: count} mapping."""
    ts = timestamp or _now()
    rows = [(ts, camera_id, vtype, int(n)) for vtype, n in counts.items()]
    if not rows:
        return
    with transaction() as cur:
        cur.executemany(
            "INSERT INTO vehicles (timestamp, camera_id, vehicle_type, count) "
            "VALUES (?,?,?,?)", rows,
        )


def insert_emergency_event(vehicle_type: str, confidence: float,
                           camera_id: int | None = None,
                           timestamp: str | None = None) -> int:
    return execute(
        "INSERT INTO emergency_events (timestamp, camera_id, vehicle_type, "
        "confidence) VALUES (?,?,?,?)",
        (timestamp or _now(), camera_id, vehicle_type, float(confidence)),
    )


def insert_traffic_data(total_count: int, density: float, score: float,
                        level: str, avg_speed: float | None = None,
                        camera_id: int | None = None,
                        timestamp: str | None = None) -> int:
    return execute(
        "INSERT INTO traffic_data (timestamp, camera_id, total_count, density, "
        "congestion_score, congestion_level, avg_speed) VALUES (?,?,?,?,?,?,?)",
        (timestamp or _now(), camera_id, int(total_count), float(density),
         float(score), level, avg_speed),
    )


def insert_emission(co2: float, nox: float, pm25: float, score: float,
                    category: str, camera_id: int | None = None,
                    timestamp: str | None = None, co: float = 0.0,
                    pm10: float = 0.0, hc: float = 0.0, voc: float = 0.0,
                    so2: float = 0.0, ch4: float = 0.0, n2o: float = 0.0,
                    co2e: float = 0.0, gas_risk: dict | None = None,
                    vehicle_breakdown: dict | None = None) -> int:
    return execute(
        "INSERT INTO emissions (timestamp, camera_id, co2, co, nox, pm25, "
        "pm10, hc, voc, so2, ch4, n2o, co2e, gas_risk_json, "
        "vehicle_breakdown_json, emission_score, category) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            timestamp or _now(), camera_id, float(co2), float(co), float(nox),
            float(pm25), float(pm10), float(hc), float(voc), float(so2),
            float(ch4), float(n2o), float(co2e),
            json.dumps(gas_risk or {}, separators=(",", ":")),
            json.dumps(vehicle_breakdown or {}, separators=(",", ":")),
            float(score), category,
        ),
    )


def insert_prediction(horizon_min: int, future_congestion: float,
                      future_level: str, camera_id: int | None = None,
                      timestamp: str | None = None) -> int:
    return execute(
        "INSERT INTO predictions (timestamp, camera_id, horizon_min, "
        "future_congestion, future_level) VALUES (?,?,?,?,?)",
        (timestamp or _now(), camera_id, int(horizon_min),
         float(future_congestion), future_level),
    )


# ---------------------------------------------------------------------------
# Fetch helpers (return plain dicts for easy JSON serialisation)
# ---------------------------------------------------------------------------
def _dicts(rows: Iterable[sqlite3.Row]) -> list[dict]:
    decoded = []
    for row in rows:
        item = dict(row)
        for key in ("gas_risk_json", "vehicle_breakdown_json"):
            if key in item and item[key]:
                try:
                    item[key.replace("_json", "")] = json.loads(item[key])
                except json.JSONDecodeError:
                    item[key.replace("_json", "")] = {}
        decoded.append(item)
    return decoded


def fetch_recent_traffic(limit: int = 100) -> list[dict]:
    return _dicts(query(
        "SELECT * FROM traffic_data ORDER BY timestamp DESC LIMIT ?", (limit,)))


def fetch_recent_emissions(limit: int = 100) -> list[dict]:
    return _dicts(query(
        "SELECT * FROM emissions ORDER BY timestamp DESC LIMIT ?", (limit,)))


def fetch_recent_emergencies(limit: int = 50) -> list[dict]:
    return _dicts(query(
        "SELECT * FROM emergency_events ORDER BY timestamp DESC LIMIT ?",
        (limit,)))


def fetch_recent_predictions(limit: int = 30) -> list[dict]:
    return _dicts(query(
        "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,)))


def fetch_vehicle_type_totals() -> dict[str, int]:
    rows = query(
        "SELECT vehicle_type, SUM(count) AS total FROM vehicles "
        "GROUP BY vehicle_type")
    return {r["vehicle_type"]: int(r["total"] or 0) for r in rows}


def fetch_recent_vehicle_counts(limit: int = 200) -> list[dict]:
    return _dicts(query(
        "SELECT * FROM vehicles ORDER BY timestamp DESC LIMIT ?", (limit,)))


def fetch_recent_traffic_chronological(limit: int = 120) -> list[dict]:
    rows = query(
        "SELECT * FROM traffic_data ORDER BY timestamp DESC LIMIT ?", (limit,))
    return [dict(r) for r in reversed(rows)]


def fetch_summary() -> dict[str, Any]:
    """Aggregate numbers for the home dashboard cards."""
    total_vehicles = query("SELECT COALESCE(SUM(count),0) AS n FROM vehicles")[0]["n"]
    emergencies = query("SELECT COUNT(*) AS n FROM emergency_events")[0]["n"]
    latest = query(
        "SELECT * FROM traffic_data ORDER BY timestamp DESC LIMIT 1")
    latest_emission = query(
        "SELECT * FROM emissions ORDER BY timestamp DESC LIMIT 1")
    return {
        "total_vehicles": int(total_vehicles or 0),
        "total_emergencies": int(emergencies or 0),
        "current_congestion": dict(latest[0]) if latest else None,
        "current_emission": dict(latest_emission[0]) if latest_emission else None,
    }


def fetch_cameras(active_only: bool = False) -> list[dict]:
    sql = "SELECT * FROM cameras"
    if active_only:
        sql += " WHERE active = 1"
    sql += " ORDER BY id"
    return _dicts(query(sql))


def add_camera(name: str, source: str, location: str | None = None,
               active: bool = True) -> int:
    return execute(
        "INSERT INTO cameras (name, source, location, active) VALUES (?,?,?,?)",
        (name, source, location, int(active)),
    )


def set_camera_active(camera_id: int, active: bool) -> int:
    return execute(
        "UPDATE cameras SET active = ? WHERE id = ?", (int(active), camera_id))


def acknowledge_emergency(event_id: int) -> int:
    return execute(
        "UPDATE emergency_events SET acknowledged = 1 WHERE id = ?",
        (event_id,))


def get_user(username: str) -> dict | None:
    rows = query("SELECT * FROM users WHERE username = ?", (username,))
    return dict(rows[0]) if rows else None


def add_user(username: str, password_hash: str, role: str = "viewer") -> int:
    return execute(
        "INSERT INTO users (username, password_hash, role) VALUES (?,?,?)",
        (username, password_hash, role),
    )


def fetch_hourly_traffic(limit: int = 24) -> list[dict]:
    """Return average congestion by hour for analytics cards/charts."""
    return _dicts(query(
        """
        SELECT substr(timestamp, 1, 13) || ':00:00' AS hour_bucket,
               ROUND(AVG(density), 2) AS avg_density,
               ROUND(AVG(congestion_score), 2) AS avg_congestion,
               SUM(total_count) AS total_count
        FROM traffic_data
        GROUP BY hour_bucket
        ORDER BY hour_bucket DESC
        LIMIT ?
        """,
        (limit,),
    ))


def fetch_emission_summary() -> dict[str, float]:
    row = query(
        """
        SELECT COALESCE(AVG(co2),0) AS avg_co2,
               COALESCE(AVG(co),0) AS avg_co,
               COALESCE(AVG(nox),0) AS avg_nox,
               COALESCE(AVG(pm25),0) AS avg_pm25,
               COALESCE(AVG(pm10),0) AS avg_pm10,
               COALESCE(AVG(hc),0) AS avg_hc,
               COALESCE(AVG(voc),0) AS avg_voc,
               COALESCE(AVG(so2),0) AS avg_so2,
               COALESCE(AVG(ch4),0) AS avg_ch4,
               COALESCE(AVG(n2o),0) AS avg_n2o,
               COALESCE(AVG(co2e),0) AS avg_co2e,
               COALESCE(AVG(emission_score),0) AS avg_score
        FROM emissions
        """
    )[0]
    return {k: round(float(row[k] or 0), 3) for k in row.keys()}


def fetch_vehicle_emission_totals(limit: int = 500) -> dict[str, dict[str, float]]:
    """Aggregate per-vehicle-class emission JSON from recent snapshots."""
    rows = fetch_recent_emissions(limit)
    totals: dict[str, dict[str, float]] = {}
    for row in rows:
        breakdown = row.get("vehicle_breakdown") or {}
        for vehicle_type, values in breakdown.items():
            vehicle_total = totals.setdefault(vehicle_type, {})
            for pollutant, value in values.items():
                if pollutant == "count":
                    vehicle_total[pollutant] = vehicle_total.get(pollutant, 0) + int(value or 0)
                else:
                    vehicle_total[pollutant] = round(
                        vehicle_total.get(pollutant, 0.0) + float(value or 0.0), 4
                    )
    return totals


def fetch_history(table: str, start: str | None = None, end: str | None = None,
                  limit: int = 1000) -> list[dict]:
    """Generic time-range fetch for the Historical Data page / CSV export."""
    allowed = {"vehicles", "emergency_events", "traffic_data",
               "emissions", "predictions"}
    if table not in allowed:
        raise DatabaseError(f"Unknown table for history: {table}")
    clauses, params = [], []
    if start:
        clauses.append("timestamp >= ?")
        params.append(start)
    if end:
        clauses.append("timestamp <= ?")
        params.append(end)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    return _dicts(query(
        f"SELECT * FROM {table}{where} ORDER BY timestamp DESC LIMIT ?", params))
