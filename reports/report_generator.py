"""PDF and CSV report generation."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import config
from database import db


def _report_name(prefix: str, ext: str) -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return config.REPORTS_DIR / f"{prefix}_{stamp}.{ext}"


def generate_csv(table: str, start: str | None = None,
                 end: str | None = None) -> Path:
    rows = db.fetch_history(table, start=start, end=end, limit=10000)
    path = _report_name(table, "csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            handle.write("message\nNo rows found\n")
    return path


def generate_pdf() -> Path:
    """Create a compact traffic, emergency, emissions and prediction report."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        return _generate_basic_pdf()

    path = _report_name("traffic_report", "pdf")
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=36, leftMargin=36)
    story = [
        Paragraph("AI-Powered Smart Traffic System Report", styles["Title"]),
        Paragraph(datetime.utcnow().strftime("Generated on %Y-%m-%d %H:%M UTC"), styles["Normal"]),
        Spacer(1, 14),
    ]

    summary = db.fetch_summary()
    current = summary.get("current_congestion") or {}
    emission = summary.get("current_emission") or {}
    summary_rows = [
        ["Metric", "Value"],
        ["Total vehicle records", summary["total_vehicles"]],
        ["Emergency events", summary["total_emergencies"]],
        ["Current congestion", current.get("congestion_level", "N/A")],
        ["Current density", f"{current.get('density', 0):.2f}%" if current else "N/A"],
        ["Emission category", emission.get("category", "N/A")],
        ["CO2e", emission.get("co2e", emission.get("co2", "N/A"))],
        ["CO2", emission.get("co2", "N/A")],
        ["CO", emission.get("co", "N/A")],
        ["NOx", emission.get("nox", "N/A")],
        ["PM2.5", emission.get("pm25", "N/A")],
        ["PM10", emission.get("pm10", "N/A")],
        ["VOC", emission.get("voc", "N/A")],
    ]
    story.append(_table(summary_rows))
    story.append(Spacer(1, 14))

    sections = [
        ("Recent Congestion", db.fetch_recent_traffic(10), ["timestamp", "density", "congestion_score", "congestion_level"]),
        ("Emergency Detections", db.fetch_recent_emergencies(10), ["timestamp", "vehicle_type", "confidence", "acknowledged"]),
        ("Emission Analysis", db.fetch_recent_emissions(10), ["timestamp", "co2e", "co2", "co", "nox", "pm25", "pm10", "category"]),
        ("Forecasts", db.fetch_recent_predictions(10), ["timestamp", "horizon_min", "future_congestion", "future_level"]),
    ]
    for title, rows, columns in sections:
        story.append(Paragraph(title, styles["Heading2"]))
        if rows:
            story.append(_table([columns] + [[row.get(col, "") for col in columns] for row in rows]))
        else:
            story.append(Paragraph("No data available.", styles["Normal"]))
        story.append(Spacer(1, 12))

    doc.build(story)
    return path


def _table(rows):
    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _generate_basic_pdf() -> Path:
    """Dependency-free fallback PDF used when reportlab is not installed yet."""
    path = _report_name("traffic_report_basic", "pdf")
    summary = db.fetch_summary()
    current = summary.get("current_congestion") or {}
    emission = summary.get("current_emission") or {}
    lines = [
        "AI-Powered Smart Traffic System Report",
        datetime.utcnow().strftime("Generated on %Y-%m-%d %H:%M UTC"),
        "",
        f"Total vehicle records: {summary['total_vehicles']}",
        f"Emergency events: {summary['total_emergencies']}",
        f"Current congestion: {current.get('congestion_level', 'N/A')}",
        f"Current density: {current.get('density', 'N/A')}",
        f"Emission category: {emission.get('category', 'N/A')}",
        f"CO2e: {emission.get('co2e', emission.get('co2', 'N/A'))}",
        f"CO2: {emission.get('co2', 'N/A')}",
        f"CO: {emission.get('co', 'N/A')}",
        f"NOx: {emission.get('nox', 'N/A')}",
        f"PM2.5: {emission.get('pm25', 'N/A')}",
        f"PM10: {emission.get('pm10', 'N/A')}",
        "",
        "Recent traffic:",
    ]
    for row in db.fetch_recent_traffic(12):
        lines.append(
            f"{row['timestamp']} | total={row['total_count']} | "
            f"density={row['density']} | level={row['congestion_level']}"
        )
    lines.append("")
    lines.append("Recent emissions:")
    for row in db.fetch_recent_emissions(12):
        lines.append(
            f"{row['timestamp']} | CO2e={row.get('co2e', row.get('co2'))} | "
            f"NOx={row.get('nox')} | PM2.5={row.get('pm25')} | {row.get('category')}"
        )
    _write_simple_pdf(path, lines)
    return path


def _write_simple_pdf(path: Path, lines: list[str]) -> None:
    def esc(text: str) -> str:
        return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    text_ops = ["BT", "/F1 11 Tf", "50 800 Td", "14 TL"]
    for line in lines[:48]:
        text_ops.append(f"({esc(line)}) Tj")
        text_ops.append("T*")
    text_ops.append("ET")
    stream = "\n".join(text_ops).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{idx} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        output.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(output)
