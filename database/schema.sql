-- ===========================================================================
-- SQLite schema for the AI-Powered Smart Traffic System.
-- All timestamps are stored as ISO-8601 strings (UTC) for easy sorting/filter.
-- ===========================================================================

PRAGMA journal_mode = WAL;      -- better concurrency for the dashboard
PRAGMA foreign_keys = ON;

-- --- Camera registry (multi-camera support) -------------------------------
CREATE TABLE IF NOT EXISTS cameras (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    source      TEXT NOT NULL,           -- file path, webcam index, or rtsp url
    location    TEXT,                    -- human-readable location label
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- --- Per-detection vehicle counts -----------------------------------------
CREATE TABLE IF NOT EXISTS vehicles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL,
    camera_id    INTEGER,
    vehicle_type TEXT NOT NULL,          -- car/bus/truck/motorcycle/bicycle
    count        INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_vehicles_ts ON vehicles(timestamp);

-- --- Emergency vehicle events ---------------------------------------------
CREATE TABLE IF NOT EXISTS emergency_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL,
    camera_id    INTEGER,
    vehicle_type TEXT NOT NULL,          -- ambulance/fire_truck/police
    confidence   REAL NOT NULL,
    acknowledged INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_emergency_ts ON emergency_events(timestamp);

-- --- Aggregated traffic / congestion snapshots ----------------------------
CREATE TABLE IF NOT EXISTS traffic_data (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    camera_id        INTEGER,
    total_count      INTEGER NOT NULL DEFAULT 0,
    density          REAL NOT NULL DEFAULT 0,     -- 0-100 %
    congestion_score REAL NOT NULL DEFAULT 0,     -- 0-100
    congestion_level TEXT NOT NULL DEFAULT 'Low', -- Low/Medium/High/Severe
    avg_speed        REAL,                        -- km/h (estimated)
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_traffic_ts ON traffic_data(timestamp);

-- --- Emission estimates ----------------------------------------------------
CREATE TABLE IF NOT EXISTS emissions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL,
    camera_id    INTEGER,
    co2          REAL NOT NULL DEFAULT 0,         -- g/km (or kg/h, see docs)
    co           REAL NOT NULL DEFAULT 0,
    nox          REAL NOT NULL DEFAULT 0,
    pm25         REAL NOT NULL DEFAULT 0,
    pm10         REAL NOT NULL DEFAULT 0,
    hc           REAL NOT NULL DEFAULT 0,
    voc          REAL NOT NULL DEFAULT 0,
    so2          REAL NOT NULL DEFAULT 0,
    ch4          REAL NOT NULL DEFAULT 0,
    n2o          REAL NOT NULL DEFAULT 0,
    co2e         REAL NOT NULL DEFAULT 0,
    gas_risk_json TEXT,
    vehicle_breakdown_json TEXT,
    emission_score REAL NOT NULL DEFAULT 0,       -- 0-100 composite
    category     TEXT NOT NULL DEFAULT 'Good',    -- Good/Moderate/Unhealthy/Hazardous
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_emissions_ts ON emissions(timestamp);

-- --- Forecasting outputs ---------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,        -- when the prediction was made
    camera_id         INTEGER,
    horizon_min       INTEGER NOT NULL,     -- 5 / 10 / 15
    future_congestion REAL NOT NULL,        -- predicted congestion score 0-100
    future_level      TEXT NOT NULL,        -- mapped band label
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(timestamp);

-- --- Users (authentication + RBAC) ----------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'viewer',  -- admin / viewer
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
