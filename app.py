"""Flask entrypoint for the AI-Powered Smart Traffic System."""
from __future__ import annotations

import json
from collections import Counter
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

import config
from core.exceptions import TrafficAIError, ValidationError
from core.exceptions import ModelNotFoundError
from core.logger import get_logger
from database import db
from detection.congestion_analyzer import CongestionAnalyzer
from detection.emergency_detector import EmergencyDetector
from detection.vehicle_detector import VehicleDetector, normalize_source
from prediction.emission_predictor import EmissionPredictor
from prediction.emission_factors import (
    EMISSION_FACTORS_G_PER_KM,
    POLLUTANT_HEALTH,
    SOURCE_NOTES,
)
from prediction.traffic_predictor import TrafficPredictor
from reports.report_generator import generate_csv, generate_pdf
from training.download_emergency_model import (
    install_model as install_emergency_model,
    install_vehicle_model,
)

log = get_logger("app")

import numpy as np
from flask.json.provider import DefaultJSONProvider

app = Flask(
    __name__,
    template_folder=str(config.TEMPLATES_DIR),
    static_folder=str(config.STATIC_DIR),
)

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json = NumpyJSONProvider(app)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 750 * 1024 * 1024

analyzer = CongestionAnalyzer()
traffic_predictor = TrafficPredictor()
emission_predictor = EmissionPredictor()

_detector_lock = threading.Lock()
_vehicle_detector: VehicleDetector | None = None
_emergency_detector: EmergencyDetector | None = None

_live_state = {
    "running": False,
    "source": None,
    "camera_id": None,
    "latest_frame": None,
    "latest_payload": {},
    "error": None,
    "last_frame_at": None,
    "mode": None,
}
_live_thread: threading.Thread | None = None
_db_ready = False
_db_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def get_vehicle_detector() -> VehicleDetector:
    global _vehicle_detector
    with _detector_lock:
        if _vehicle_detector is None:
            _vehicle_detector = VehicleDetector()
        return _vehicle_detector


def get_emergency_detector() -> EmergencyDetector:
    global _emergency_detector
    with _detector_lock:
        if _emergency_detector is None:
            _emergency_detector = EmergencyDetector()
        return _emergency_detector


@app.before_request
def ensure_database() -> None:
    global _db_ready
    if _db_ready:
        return
    with _db_lock:
        if not _db_ready:
            db.init_db(seed_admin=True)
            _db_ready = True


@app.teardown_appcontext
def close_database(_exc) -> None:
    db.close_connection()


@app.context_processor
def inject_globals() -> dict:
    return {
        "app_name": "Smart Traffic AI",
        "current_user": session.get("user"),
        "current_role": session.get("role"),
        "year": datetime.now().year,
    }


def login_required(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login", next=request.path))
        return func(*args, **kwargs)

    return wrapper


def admin_required(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if session.get("role") != "admin":
            flash("Admin access is required.", "warning")
            return redirect(url_for("home"))
        return func(*args, **kwargs)

    return wrapper


def _json_error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


def _safe_int(value, default: int, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _fuel_waste_liters(total_count: int, congestion_level: str) -> float:
    idle_minutes = {"Low": 1, "Medium": 3, "High": 6, "Severe": 9}.get(
        congestion_level,
        1,
    )
    return round(max(0, int(total_count)) * idle_minutes * 0.01, 2)


def _is_browser_feed_stale() -> bool:
    if _live_state.get("mode") != "browser":
        return False
    last_frame_at = _live_state.get("last_frame_at")
    return bool(last_frame_at and time.time() - float(last_frame_at) > 8)


@app.errorhandler(TrafficAIError)
def handle_project_error(exc):
    log.warning("Handled project error: %s", exc)
    if request.path.startswith("/api/"):
        return _json_error(str(exc), 400)
    flash(str(exc), "danger")
    return redirect(url_for("home"))


@app.errorhandler(Exception)
def handle_unexpected(exc):
    if isinstance(exc, HTTPException):
        if request.path.startswith("/api/"):
            return _json_error(exc.description, exc.code or 500)
        return render_template("404.html", error=exc), exc.code or 500
    log.exception("Unhandled error: %s", exc)
    if request.path.startswith("/api/"):
        return _json_error("Unexpected server error", 500)
    flash("Unexpected server error. Check logs for details.", "danger")
    return redirect(url_for("home"))


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = db.get_user(username)
        if user and check_password_hash(user["password_hash"], password):
            session["user"] = user["username"]
            session["role"] = user["role"]
            return redirect(request.args.get("next") or url_for("home"))
        flash("Invalid username or password.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def home():
    return render_template(
        "dashboard.html",
        summary=db.fetch_summary(),
        vehicle_totals=db.fetch_vehicle_type_totals(),
        recent_traffic=db.fetch_recent_traffic(8),
        recent_emergencies=db.fetch_recent_emergencies(5),
        recent_predictions=db.fetch_recent_predictions(9),
    )


@app.route("/live")
@login_required
def live_monitoring():
    return render_template("live.html", cameras=db.fetch_cameras(active_only=True))


@app.route("/alerts")
@login_required
def alerts():
    return render_template("alerts.html", events=db.fetch_recent_emergencies(100))


@app.route("/alerts/<int:event_id>/ack", methods=["POST"])
@login_required
def acknowledge_alert(event_id: int):
    db.acknowledge_emergency(event_id)
    flash("Emergency event acknowledged.", "success")
    return redirect(url_for("alerts"))


@app.route("/traffic")
@login_required
def traffic_analytics():
    return render_template(
        "traffic.html",
        traffic=db.fetch_recent_traffic(200),
        hourly=list(reversed(db.fetch_hourly_traffic(24))),
    )


@app.route("/emissions")
@login_required
def emission_analytics():
    emissions = db.fetch_recent_emissions(120)
    chart_fields = [
        "timestamp", "co2", "co", "nox", "pm25", "pm10", "hc", "voc",
        "so2", "ch4", "n2o", "co2e", "emission_score", "gas_risk",
    ]
    emission_chart_rows = [
        {field: row.get(field) for field in chart_fields}
        for row in reversed(emissions[:60])
    ]
    return render_template(
        "emissions.html",
        emissions=emissions,
        emission_chart_rows=emission_chart_rows,
        summary=db.fetch_emission_summary(),
        vehicle_emissions=db.fetch_vehicle_emission_totals(500),
        pollutant_health=POLLUTANT_HEALTH,
        emission_factors=EMISSION_FACTORS_G_PER_KM,
        source_notes=SOURCE_NOTES,
    )


@app.route("/history")
@login_required
def historical_data():
    table = request.args.get("table", "traffic_data")
    start = request.args.get("start") or None
    end = request.args.get("end") or None
    rows = db.fetch_history(table, start=start, end=end, limit=500)
    return render_template("history.html", rows=rows, table=table, start=start, end=end)


@app.route("/reports")
@login_required
def reports_page():
    reports = sorted(config.REPORTS_DIR.glob("*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return render_template("reports.html", reports=reports)


@app.route("/reports/pdf", methods=["POST"])
@login_required
def export_pdf():
    path = generate_pdf()
    flash(f"Generated {path.name}.", "success")
    return send_file(path, as_attachment=True)


@app.route("/reports/csv", methods=["POST"])
@login_required
def export_csv():
    table = request.form.get("table", "traffic_data")
    path = generate_csv(
        table,
        start=request.form.get("start") or None,
        end=request.form.get("end") or None,
    )
    flash(f"Generated {path.name}.", "success")
    return send_file(path, as_attachment=True)


@app.route("/performance")
@login_required
def model_performance():
    metrics = {}
    for name in ("emergency_yolo_metrics.json", "lstm_metrics.json", "xgboost_metrics.json"):
        path = config.DATA_DIR / name
        if path.exists():
            metrics[name] = json.loads(path.read_text(encoding="utf-8"))
    return render_template(
        "performance.html",
        metrics=metrics,
        vehicle_model_exists=Path(config.VEHICLE_MODEL).exists(),
        vehicle_model_path=config.VEHICLE_MODEL,
        traffic_density_model_path=config.TRAFFIC_DENSITY_MODEL,
        reference_vehicle_models=[
            path for path in (
                config.SMART_TRAFFIC_MODEL,
                config.REFERENCE_TRAFFIC_DENSITY_MODEL,
            )
            if Path(path).exists()
        ],
        emergency_model_exists=Path(config.EMERGENCY_MODEL).exists(),
        emergency_model_path=config.EMERGENCY_MODEL,
        lstm_model_exists=Path(config.LSTM_MODEL_PATH).exists(),
        xgb_models_exist=all(Path(p).exists() for p in config.XGB_MODELS.values()),
    )


@app.route("/models/emergency/install", methods=["POST"])
@login_required
@admin_required
def install_emergency_model_route():
    url = request.form.get("model_url", "").strip() or None
    upload = request.files.get("model_file")
    local_path = None
    if upload and upload.filename:
        filename = secure_filename(upload.filename)
        if not filename.endswith(".pt"):
            flash("Upload a YOLO .pt file.", "danger")
            return redirect(url_for("model_performance"))
        local_path = config.UPLOAD_DIR / filename
        upload.save(local_path)
    if not url and not local_path:
        flash("Provide a trusted model URL or upload a .pt file.", "danger")
        return redirect(url_for("model_performance"))
    try:
        target = install_emergency_model(url=url, source_file=str(local_path) if local_path else None)
        global _emergency_detector
        _emergency_detector = None
        flash(f"Installed emergency model at {target}.", "success")
    except Exception as exc:
        flash(f"Could not install emergency model: {exc}", "danger")
    return redirect(url_for("model_performance"))


@app.route("/models/vehicle/install", methods=["POST"])
@login_required
@admin_required
def install_vehicle_model_route():
    url = request.form.get("model_url", "").strip() or None
    upload = request.files.get("model_file")
    local_path = None
    if upload and upload.filename:
        filename = secure_filename(upload.filename)
        if not filename.lower().endswith(".pt"):
            flash("Upload a YOLO .pt file.", "danger")
            return redirect(url_for("model_performance"))
        local_path = config.UPLOAD_DIR / filename
        upload.save(local_path)
    if not url and not local_path:
        flash("Provide a trusted model URL or upload a .pt file.", "danger")
        return redirect(url_for("model_performance"))
    try:
        target = install_vehicle_model(url=url, source_file=str(local_path) if local_path else None)
        global _vehicle_detector
        _vehicle_detector = None
        flash(f"Installed vehicle model at {target}.", "success")
    except Exception as exc:
        flash(f"Could not install vehicle model: {exc}", "danger")
    return redirect(url_for("model_performance"))


@app.route("/admin/users", methods=["GET", "POST"])
@login_required
@admin_required
def users():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        role = request.form.get("role", "viewer")
        if not username or not password or role not in config.ROLES:
            flash("Provide a username, password and valid role.", "danger")
        else:
            db.add_user(username, generate_password_hash(password), role)
            flash("User created.", "success")
    return render_template("users.html")


def _persist_snapshot(payload: dict, camera_id: int | None) -> None:
    counts = payload["vehicles"]["counts"]
    congestion = payload["congestion"]
    emissions = payload["emissions"]
    ts = _now()
    db.insert_vehicle_counts(counts, camera_id=camera_id, timestamp=ts)
    db.insert_traffic_data(
        total_count=congestion["total_count"],
        density=congestion["density"],
        score=congestion["congestion_score"],
        level=congestion["level"],
        avg_speed=congestion["avg_speed"],
        camera_id=camera_id,
        timestamp=ts,
    )
    db.insert_emission(
        co2=emissions["co2"],
        co=emissions.get("co", 0),
        nox=emissions["nox"],
        pm25=emissions["pm25"],
        pm10=emissions.get("pm10", 0),
        hc=emissions.get("hc", 0),
        voc=emissions.get("voc", 0),
        so2=emissions.get("so2", 0),
        ch4=emissions.get("ch4", 0),
        n2o=emissions.get("n2o", 0),
        co2e=emissions.get("co2e", emissions["co2"]),
        score=emissions["emission_score"],
        category=emissions["category"],
        gas_risk=emissions.get("gas_risk"),
        vehicle_breakdown=emissions.get("vehicle_breakdown"),
        camera_id=camera_id,
        timestamp=ts,
    )
    for event in payload["emergency"]:
        db.insert_emergency_event(
            event["vehicle_type"],
            event["confidence"],
            camera_id=camera_id,
            timestamp=ts,
        )
    for forecast in payload["predictions"]:
        db.insert_prediction(
            horizon_min=forecast["horizon_min"],
            future_congestion=forecast["future_congestion"],
            future_level=forecast["future_level"],
            camera_id=camera_id,
            timestamp=ts,
        )


def analyze_frame(frame, camera_id: int | None = None, persist: bool = True) -> tuple[dict, bytes]:
    vehicle_result = get_vehicle_detector().detect(frame)
    emergency_model_status = "ready"
    try:
        emergency = get_emergency_detector().detect(frame)
    except ModelNotFoundError as exc:
        emergency = []
        emergency_model_status = str(exc)
    congestion = analyzer.analyze(vehicle_result.counts)
    emissions = emission_predictor.predict(
        vehicle_result.counts,
        congestion.density,
        congestion.avg_speed,
    )
    recent = db.fetch_recent_traffic_chronological(config.SEQUENCE_LENGTH)
    predictions = [p.as_dict() for p in traffic_predictor.predict(recent)]

    annotated = get_vehicle_detector().draw(frame, vehicle_result)
    if emergency:
        annotated = get_emergency_detector().draw(annotated, emergency)
    ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise ValidationError("Could not encode frame")

    payload = {
        "timestamp": _now(),
        "vehicles": vehicle_result.as_dict(),
        "emergency": [e.as_dict() for e in emergency],
        "emergency_model_status": emergency_model_status,
        "congestion": congestion.as_dict(),
        "emissions": emissions.as_dict(),
        "predictions": predictions,
    }
    payload["impact"] = {
        "density": congestion.level,
        "co2e_g": emissions.co2e,
        "fuel_waste_l": _fuel_waste_liters(congestion.total_count, congestion.level),
        "left_lane": vehicle_result.roi_density.get("left_lane", {}),
        "right_lane": vehicle_result.roi_density.get("right_lane", {}),
    }
    if persist:
        _persist_snapshot(payload, camera_id)
    return payload, encoded.tobytes()


def analyze_video_file(
    path: Path,
    camera_id: int | None = None,
    every_n_frames: int | None = None,
    max_frames: int | None = None,
) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValidationError(f"Could not open uploaded video: {path.name}")

    every = max(1, every_n_frames or config.VIDEO_SAMPLE_EVERY_N_FRAMES)
    max_items = max_frames or config.VIDEO_MAX_ANALYSIS_FRAMES
    # Use float('inf') for max_items if -1 (all frames)
    if max_items == -1:
        max_items = float("inf")
    
    frame_index = 0
    analyzed = 0
    total_counts: Counter[str] = Counter()
    levels: Counter[str] = Counter()
    emergency_events = []
    density_values = []
    score_values = []
    speed_values = []
    emission_totals: Counter[str] = Counter()
    previews = []
    timeline = []
    fuel_waste_total = 0.0
    roi_lane_totals: Counter[str] = Counter()
    heavy_lane_frames: Counter[str] = Counter()

    # Aggregate per-vehicle emissions if possible for the breakdown
    per_type_emissions = {}
    
    try:
        while analyzed < max_items:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % every != 0:
                frame_index += 1
                continue
            
            # OPTIMIZATION: Call detectors directly to avoid repeated jpeg encoding if not a preview
            is_preview = len(previews) < 12
            vehicle_result = get_vehicle_detector().detect(frame)
            emergency = []
            try:
                emergency = get_emergency_detector().detect(frame)
            except Exception:
                pass
            
            congestion = analyzer.analyze(vehicle_result.counts)
            emissions = emission_predictor.predict(
                vehicle_result.counts,
                congestion.density,
                congestion.avg_speed,
            )
            
            payload = {
                "vehicles": vehicle_result.as_dict(),
                "congestion": congestion.as_dict(),
                "emissions": emissions.as_dict(),
                "emergency": [e.as_dict() for e in emergency],
            }
            
            analyzed += 1
            for vehicle_type, count in payload["vehicles"]["counts"].items():
                total_counts[vehicle_type] += int(count)
                if vehicle_type not in per_type_emissions:
                    per_type_emissions[vehicle_type] = Counter()
                
                # Get breakdown from estimate if available
                v_breakdown = emissions.vehicle_breakdown.get(vehicle_type, {})
                for k, v in v_breakdown.items():
                    if isinstance(v, (int, float)):
                        per_type_emissions[vehicle_type][k] += float(v)

            levels[payload["congestion"]["level"]] += 1
            density_values.append(float(payload["congestion"]["density"]))
            score_values.append(float(payload["congestion"]["congestion_score"]))
            speed_values.append(float(payload["congestion"]["avg_speed"]))
            emergency_events.extend(payload["emergency"])
            
            for pollutant in config.EMISSION_POLLUTANTS:
                emission_totals[pollutant] += float(payload["emissions"].get(pollutant, 0))
            
            if is_preview:
                annotated = get_vehicle_detector().draw(frame, vehicle_result)
                if emergency:
                    annotated = get_emergency_detector().draw(annotated, emergency)
                ok_enc, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok_enc:
                    preview = config.FRAME_DIR / f"video_{path.stem}_{frame_index}.jpg"
                    preview.write_bytes(jpg.tobytes())
                    previews.append(url_for("static", filename=f"frames/{preview.name}"))
            
            roi_density = payload["vehicles"].get("roi_density", {})
            left_lane = roi_density.get("left_lane", {})
            right_lane = roi_density.get("right_lane", {})
            roi_lane_totals["left"] += int(left_lane.get("count", 0))
            roi_lane_totals["right"] += int(right_lane.get("count", 0))
            if left_lane.get("heavy"):
                heavy_lane_frames["left"] += 1
            if right_lane.get("heavy"):
                heavy_lane_frames["right"] += 1
            
            # Fuel waste: total count * idle factor
            fuel_waste_total += _fuel_waste_liters(congestion.total_count, congestion.level)
            
            timeline.append(
                {
                    "frame": frame_index,
                    "total": payload["congestion"]["total_count"],
                    "density": payload["congestion"]["density"],
                    "score": payload["congestion"]["congestion_score"],
                    "co2e": payload["emissions"].get("co2e", payload["emissions"].get("co2", 0)),
                    "left_lane": left_lane.get("count", 0),
                    "right_lane": right_lane.get("count", 0),
                }
            )
            frame_index += 1
    finally:
        cap.release()

    if analyzed == 0:
        raise ValidationError(
            f"OpenCV could open {path.name}, but no frames were decoded. "
            "Try another codec/container or convert the file to H.264 MP4."
        )

    peak_frame = max(timeline, key=lambda row: row["total"], default={})
    
    # Process per-type emissions for the response
    processed_type_emissions = {}
    for vtype, counts in per_type_emissions.items():
        processed_type_emissions[vtype] = {k: round(v, 2) for k, v in counts.items()}

    return {
        "video": path.name,
        "sample_every_n_frames": every,
        "frames_analyzed": analyzed,
        "vehicle_totals": dict(total_counts),
        "avg_density": round(sum(density_values) / max(1, len(density_values)), 2),
        "avg_congestion_score": round(sum(score_values) / max(1, len(score_values)), 2),
        "avg_speed": round(sum(speed_values) / max(1, len(speed_values)), 2),
        "congestion_levels": dict(levels),
        "emergency_events": emergency_events,
        "emission_totals": {k: round(v, 4) for k, v in emission_totals.items()},
        "type_emission_totals": processed_type_emissions,
        "fuel_waste_l": round(fuel_waste_total, 2),
        "roi_density": {
            "avg_left_lane": round(roi_lane_totals["left"] / max(1, analyzed), 2),
            "avg_right_lane": round(roi_lane_totals["right"] / max(1, analyzed), 2),
            "heavy_left_frames": int(heavy_lane_frames["left"]),
            "heavy_right_frames": int(heavy_lane_frames["right"]),
        },
        "timeline": timeline,
        "peak_frame": peak_frame,
        "preview_frames": previews,
    }


def _capture_loop(source, camera_id: int | None = None) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        _live_state["error"] = f"Could not open source: {source}"
        _live_state["running"] = False
        return
    frame_counter = 0
    try:
        while _live_state["running"]:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.1)
                continue
            if frame_counter % 8 == 0:
                payload, jpg = analyze_frame(frame, camera_id=camera_id, persist=True)
                _live_state["latest_payload"] = payload
                _live_state["latest_frame"] = jpg
                _live_state["last_frame_at"] = time.time()
            frame_counter += 1
            time.sleep(0.03)
    except Exception as exc:
        log.exception("Live capture failed: %s", exc)
        _live_state["error"] = str(exc)
    finally:
        cap.release()
        _live_state["running"] = False


@app.route("/api/live/start", methods=["POST"])
@login_required
def api_live_start():
    global _live_thread
    payload = request.get_json(silent=True) if request.is_json else {}
    source = request.form.get("source") or (payload or {}).get("source")
    camera_id = None
    if not source:
        cameras = db.fetch_cameras(active_only=True)
        if not cameras:
            return _json_error("No active camera/source configured", 400)
        camera_id = cameras[0]["id"]
        source = cameras[0]["source"]
    source = normalize_source(source)
    if _live_state["running"]:
        return jsonify({"ok": True, "message": "Live monitoring already running"})
    _live_state.update(
        {
            "running": True,
            "source": str(source),
            "camera_id": camera_id,
            "latest_frame": None,
            "latest_payload": {},
            "error": None,
            "last_frame_at": None,
            "mode": "server",
        }
    )
    _live_thread = threading.Thread(target=_capture_loop, args=(source, camera_id), daemon=True)
    _live_thread.start()
    time.sleep(0.8) # Wait slightly longer for OpenCV to initialize
    if _live_state.get("error"):
        err = _live_state["error"]
        if "Could not open source: 0" in err:
            err = "Server webcam at index 0 not found. If you are on a remote server, please use 'Start Browser Camera' instead."
        return _json_error(err, 400)
    return jsonify({"ok": True, "source": str(source)})


@app.route("/api/live/stop", methods=["POST"])
@login_required
def api_live_stop():
    _live_state["running"] = False
    _live_state["mode"] = None
    return jsonify({"ok": True})


@app.route("/api/live/status")
@login_required
def api_live_status():
    if _is_browser_feed_stale():
        _live_state["running"] = False
    return jsonify(
        {
            "ok": True,
            "running": _live_state["running"],
            "source": _live_state["source"],
            "error": _live_state["error"],
            "mode": _live_state["mode"],
            "last_frame_at": _live_state["last_frame_at"],
            "payload": _live_state["latest_payload"],
        }
    )


@app.route("/api/live/frame")
@login_required
def api_live_frame():
    def generate():
        while True:
            frame = _live_state.get("latest_frame")
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.2)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/analyze-image", methods=["POST"])
@login_required
def api_analyze_image():
    upload = request.files.get("image")
    if not upload:
        return _json_error("Upload an image file with field name 'image'", 400)
    filename = secure_filename(upload.filename or "frame.jpg")
    path = config.UPLOAD_DIR / filename
    upload.save(path)
    frame = cv2.imread(str(path))
    if frame is None:
        return _json_error("Could not read uploaded image", 400)
    payload, jpg = analyze_frame(frame, persist=True)
    preview = config.FRAME_DIR / f"analysis_{int(time.time())}.jpg"
    preview.write_bytes(jpg)
    payload["preview_url"] = url_for("static", filename=f"frames/{preview.name}")
    return jsonify({"ok": True, "result": payload})


@app.route("/api/analyze-video", methods=["POST"])
@login_required
def api_analyze_video():
    upload = request.files.get("video")
    if not upload:
        return _json_error("Upload a video file with field name 'video'", 400)
    filename = secure_filename(upload.filename or "traffic_video.mp4")
    if "." not in filename:
        filename = f"{filename}.mp4"
    path = config.UPLOAD_DIR / filename
    upload.save(path)
    every = _safe_int(
        request.form.get("every_n_frames"),
        config.VIDEO_SAMPLE_EVERY_N_FRAMES,
        minimum=1,
        maximum=300,
    )
    max_frames = _safe_int(
        request.form.get("max_frames"),
        config.VIDEO_MAX_ANALYSIS_FRAMES,
        minimum=1,
        maximum=1000,
    )
    result = analyze_video_file(path, every_n_frames=every, max_frames=max_frames)
    return jsonify({"ok": True, "result": result})


@app.route("/api/analyze-browser-frame", methods=["POST"])
@login_required
def api_analyze_browser_frame():
    upload = request.files.get("frame")
    if not upload:
        return _json_error("Upload a browser camera frame with field name 'frame'", 400)
    data = upload.read()
    import numpy as np

    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return _json_error("Could not decode browser camera frame", 400)
    payload, jpg = analyze_frame(frame, persist=True)
    
    # Update global live state so dashboard charts/status show browser camera data
    _live_state["latest_payload"] = payload
    _live_state["latest_frame"] = jpg
    _live_state["running"] = True
    _live_state["source"] = "browser-camera"
    _live_state["camera_id"] = None
    _live_state["error"] = None
    _live_state["last_frame_at"] = time.time()
    _live_state["mode"] = "browser"
    
    payload["preview_data_url"] = "data:image/jpeg;base64," + __import__("base64").b64encode(jpg).decode("ascii")
    return jsonify({"ok": True, "result": payload})


@app.route("/api/summary")
@login_required
def api_summary():
    return jsonify(
        {
            "summary": db.fetch_summary(),
            "vehicle_totals": db.fetch_vehicle_type_totals(),
            "traffic": db.fetch_recent_traffic_chronological(120),
            "emissions": list(reversed(db.fetch_recent_emissions(120))),
            "emergencies": db.fetch_recent_emergencies(20),
            "predictions": db.fetch_recent_predictions(12),
        }
    )


@app.route("/api/predict/traffic", methods=["POST"])
@login_required
def api_predict_traffic():
    rows = db.fetch_recent_traffic_chronological(config.SEQUENCE_LENGTH)
    forecasts = traffic_predictor.predict(rows)
    for forecast in forecasts:
        db.insert_prediction(forecast.horizon_min, forecast.future_congestion, forecast.future_level)
    return jsonify({"ok": True, "predictions": [f.as_dict() for f in forecasts]})


@app.route("/api/predict/emissions", methods=["POST"])
@login_required
def api_predict_emissions():
    payload = request.get_json(force=True, silent=True) or {}
    counts = payload.get("counts") or {}
    density = float(payload.get("density", 0))
    avg_speed = float(payload.get("avg_speed", 40))
    estimate = emission_predictor.predict(counts, density, avg_speed)
    db.insert_emission(
        estimate.co2,
        estimate.nox,
        estimate.pm25,
        estimate.emission_score,
        estimate.category,
        co=estimate.co,
        pm10=estimate.pm10,
        hc=estimate.hc,
        voc=estimate.voc,
        so2=estimate.so2,
        ch4=estimate.ch4,
        n2o=estimate.n2o,
        co2e=estimate.co2e,
        gas_risk=estimate.gas_risk,
        vehicle_breakdown=estimate.vehicle_breakdown,
    )
    return jsonify({"ok": True, "emissions": estimate.as_dict()})


@app.route("/api/history/<table>")
@login_required
def api_history(table: str):
    return jsonify(
        {
            "ok": True,
            "rows": db.fetch_history(
                table,
                start=request.args.get("start") or None,
                end=request.args.get("end") or None,
                limit=int(request.args.get("limit", 500)),
            ),
        }
    )


@app.route("/api/cameras", methods=["GET", "POST"])
@login_required
def api_cameras():
    if request.method == "POST":
        payload = request.get_json(force=True, silent=True) or request.form
        camera_id = db.add_camera(
            name=payload.get("name", "Camera"),
            source=payload.get("source", "0"),
            location=payload.get("location"),
            active=bool(payload.get("active", True)),
        )
        return jsonify({"ok": True, "id": camera_id})
    return jsonify({"ok": True, "cameras": db.fetch_cameras()})


def create_app() -> Flask:
    db.init_db(seed_admin=True)
    return app


if __name__ == "__main__":
    db.init_db(seed_admin=True)
    log.info("Starting dashboard on http://%s:%s", config.HOST, config.PORT)
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
