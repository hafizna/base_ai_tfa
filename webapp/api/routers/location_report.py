"""Fault-location reports: TWS (traveling wave, both ends) and double-ended
impedance FL, on the template every report shares (report.py).

These distances come from a different process than the distance engine, and
the report says so before anything else:
- the method's accent colour, drawn above the header on every page;
- a method band that names the method and prints the process path the
  distance came from;
- a field-confirmation form for the patrol.

Page 1 carries the location (distance from both ends, line diagram), what it
was computed from, and whether it can be trusted. Page 2 carries the
attachments: the waveforms (TWS) or the distance histogram and the inputs
(DE-FL).
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    Image,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from core.line_selection import (
    LINE_ALSO_OPERATED,
    LINE_BREAKER_OR_RECLOSE_ONLY,
    LINE_DE_ENERGIZED,
    LINE_IMPACTED,
    LINE_QUIET,
    scope_payload_with_selection,
)
from ..schemas import DoubleEndedComputeRequest
from ..storage import load_analysis
from .relay_21_de import GROUND_LOOPS, _run_compute
from .report import (
    BRAND_BORDER,
    BRAND_MUTED,
    BRAND_NAVY,
    BRAND_SLATE,
    MARGIN,
    PAGE_H,
    PAGE_W,
    ChartImage,
    _build_styles,
    _format_datetime,
    _HeaderFooter,
    _safe_text,
)

router = APIRouter(prefix="/api/location-report", tags=["location-report"])

FRAME_W = PAGE_W - 2 * MARGIN
RULE = colors.HexColor("#dde2e8")
FORM_LINE = colors.HexColor("#9aa4b1")

# Pass/fail colours match the other reports' severity palette.
VERDICT_STYLE = {
    "pass": ("Lolos", "#14532d", colors.HexColor("#f0fdf4")),
    "fail": ("Tidak lolos", "#991b1b", colors.HexColor("#fef2f2")),
    "pending": ("Belum", "#92400e", colors.HexColor("#fffbeb")),
    "info": ("Info", "#475569", colors.HexColor("#f8fafc")),
}


@dataclass(frozen=True)
class Method:
    """How a location was computed, and the accent that marks it on paper."""
    title: str
    subtitle: str
    basis: str
    accent_hex: str
    tint_hex: str
    footer: str

    @property
    def accent(self) -> colors.Color:
        return colors.HexColor(self.accent_hex)

    @property
    def tint(self) -> colors.Color:
        return colors.HexColor(self.tint_hex)


TWS = Method(
    title="Traveling wave dua ujung (Type D)",
    subtitle="Traveling wave · dua ujung (Type D)",
    basis=(
        "Dari selisih waktu tiba gelombang berjalan di kedua ujung (jam GPS), bukan dari impedansi satu "
        "ujung seperti engine distance."
    ),
    accent_hex="#0e7490",
    tint_hex="#ecfeff",
    footer="TFA · lokasi traveling wave",
)
DOUBLE_ENDED = Method(
    title="Impedansi dua ujung (hukum tegangan Kirchhoff)",
    subtitle="Impedansi · dua ujung",
    basis=(
        "Dari tegangan dan arus kedua ujung yang disinkronkan, bukan dari impedansi satu ujung seperti "
        "engine distance."
    ),
    accent_hex="#6d28d9",
    tint_hex="#f5f3ff",
    footer="TFA · lokasi dua ujung",
)

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
_PHASE_LABEL = {"A": "R", "B": "S", "C": "T"}
_LOOP_LABEL = {"ZA": "R-N", "ZB": "S-N", "ZC": "T-N", "ZAB": "R-S", "ZBC": "S-T", "ZCA": "T-R"}
_SHIFT_SOURCE = {
    "estimate": "estimasi dari inception gangguan",
    "residual_search": "saran pencarian residual",
    "manual": "diatur manual",
}
_SELECTION_METHOD = {
    "status_protection": "proteksinya operate",
    "status_breaker": "PMT-nya operate",
    "status_reclose": "reclose-nya operate",
    "current_interruption": "arusnya terputus",
    "superimposed_current": "arus gangguannya terbesar",
}
_LINE_STATE = {
    LINE_DE_ENERGIZED: "tidak bertegangan",
    LINE_QUIET: "tidak terganggu",
    LINE_IMPACTED: "ikut dialiri arus gangguan",
    LINE_BREAKER_OR_RECLOSE_ONLY: "hanya PMT/reclose yang bekerja",
    LINE_ALSO_OPERATED: "ikut operate",
}


# ---------------------------------------------------------------------------
# Formatting (Indonesian: decimal comma, minus sign)
# ---------------------------------------------------------------------------

def _num(value, digits: int = 2, unit: str = "", signed: bool = False) -> str:
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "—"
    text = f"{abs(value):,.{digits}f}".replace(",", " ").replace(".", ",")
    if round(abs(value), digits) == 0:
        sign = ""
    elif value < 0:
        sign = "−"
    else:
        sign = "+" if signed else ""
    return f"{sign}{text}{' ' + unit if unit else ''}"


def _event_time(moment: Optional[datetime]) -> str:
    if moment is None:
        return "—"
    millis = f"{moment.microsecond // 1000:03d}"
    return f"{moment.day} {_MONTHS[moment.month - 1]} {moment.year} · {moment:%H:%M:%S},{millis}"


def _parse_iso(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _short_id(analysis_id: str) -> str:
    return (analysis_id or "")[:8]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _styles(method: Method) -> dict[str, ParagraphStyle]:
    styles = _build_styles()
    body = styles["body"]
    styles["section_kicker"] = ParagraphStyle("lr_section_kicker", parent=styles["section_kicker"], textColor=method.accent)
    styles["label"] = ParagraphStyle("lr_label", parent=body, fontSize=7.5, leading=9.5, textColor=BRAND_MUTED)
    styles["value"] = ParagraphStyle("lr_value", parent=body, fontSize=9.5, leading=12)
    styles["mono"] = ParagraphStyle("lr_mono", parent=styles["value"], fontName="Courier")
    styles["note"] = ParagraphStyle("lr_note", parent=body, fontSize=8.5, leading=12, textColor=BRAND_SLATE)
    styles["step"] = ParagraphStyle("lr_step", parent=body, fontSize=7, leading=8.8)
    styles["arrow"] = ParagraphStyle("lr_arrow", parent=body, fontSize=10, leading=12, textColor=method.accent, alignment=1)
    styles["band_kicker"] = ParagraphStyle("lr_band_kicker", parent=body, fontName="Helvetica-Bold", fontSize=7, leading=9, textColor=method.accent)
    styles["band_title"] = ParagraphStyle("lr_band_title", parent=body, fontName="Helvetica-Bold", fontSize=11, leading=14)
    styles["band_body"] = ParagraphStyle("lr_band_body", parent=body, fontSize=8.5, leading=11.5, textColor=BRAND_SLATE)
    styles["hero"] = ParagraphStyle("lr_hero", parent=body, fontName="Helvetica-Bold", fontSize=18, leading=21)
    styles["hero_right"] = ParagraphStyle("lr_hero_right", parent=styles["hero"], alignment=2)
    styles["hero_sub"] = ParagraphStyle("lr_hero_sub", parent=body, fontSize=9.5, leading=12)
    styles["hero_sub_right"] = ParagraphStyle("lr_hero_sub_right", parent=styles["hero_sub"], alignment=2)
    styles["kicker_row"] = ParagraphStyle("lr_kicker_row", parent=body, fontName="Helvetica-Bold", fontSize=7.5, leading=10, textColor=BRAND_MUTED)
    styles["th"] = ParagraphStyle("lr_th", parent=body, fontName="Helvetica-Bold", fontSize=7.5, leading=9.5, textColor=BRAND_MUTED)
    styles["td"] = ParagraphStyle("lr_td", parent=body, fontSize=9, leading=11.5)
    styles["td_right"] = ParagraphStyle("lr_td_right", parent=styles["td"], alignment=2)
    styles["th_right"] = ParagraphStyle("lr_th_right", parent=styles["th"], alignment=2)
    return styles


def _section(styles: dict, kicker: str, title: str, aside: Optional[str] = None) -> list:
    head = Paragraph(title, styles["section"])
    if aside:
        head = Table(
            [[Paragraph(title, styles["section"]), Paragraph(aside, ParagraphStyle(
                "lr_aside", parent=styles["label"], alignment=2))]],
            colWidths=[FRAME_W * 0.5, FRAME_W * 0.5],
        )
        head.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
    return [Paragraph(kicker, styles["section_kicker"]), head]


def _method_band(styles: dict, method: Method, steps: list[tuple[str, str]]) -> Table:
    """The method and the path its distance came from, step by step."""
    gap = 13.0
    inner_w = FRAME_W - 2 * 10
    step_w = (inner_w - gap * (len(steps) - 1)) / len(steps)
    cells: list = []
    widths: list[float] = []
    for i, (title, text) in enumerate(steps):
        if i:
            cells.append(Paragraph("→", styles["arrow"]))
            widths.append(gap)
        cells.append(Paragraph(f"<b>{_safe_text(title)}</b><br/>{text}", styles["step"]))
        widths.append(step_w)
    path = Table([cells], colWidths=widths)
    path.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        *[("BACKGROUND", (col, 0), (col, 0), colors.white) for col in range(0, len(cells), 2)],
        *[("BOX", (col, 0), (col, 0), 0.5, method.accent) for col in range(0, len(cells), 2)],
        *[("LEFTPADDING", (col, 0), (col, 0), 0) for col in range(1, len(cells), 2)],
        *[("RIGHTPADDING", (col, 0), (col, 0), 0) for col in range(1, len(cells), 2)],
    ]))
    band = Table(
        [
            [Paragraph("METODE · BERBEDA DARI ENGINE DISTANCE", styles["band_kicker"])],
            [Paragraph(_safe_text(method.title), styles["band_title"])],
            [Paragraph(_safe_text(method.basis), styles["band_body"])],
            [Paragraph("JALUR PROSES", styles["band_kicker"])],
            [path],
        ],
        colWidths=[FRAME_W],
    )
    band.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), method.tint),
        ("LINEBEFORE", (0, 0), (0, -1), 3, method.accent),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("TOPPADDING", (0, 0), (0, 0), 7),
        ("TOPPADDING", (0, 3), (0, 3), 5),
        ("BOTTOMPADDING", (0, -1), (0, -1), 8),
    ]))
    return band


def _clip(text: str, limit: int = 26) -> str:
    """A long record file name, shortened for a narrow cell; the full name
    is in the inputs table. A C37.232 name (date,time,offset,station,device,
    company) shortens to its last field."""
    text = str(text)
    fields = [field.strip() for field in text.split(",")]
    if len(fields) >= 5 and fields[-1]:
        text = fields[-1]
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _tick_step(line_km: float) -> float:
    for step in (1, 2, 5, 10, 20, 25, 50, 100):
        if line_km / step <= 8:
            return float(step)
    return 200.0


def _line_diagram(
    line_km: float,
    left: tuple[str, str, str],
    right: tuple[str, str, str],
    markers: list[dict],
) -> Drawing:
    """The line from end to end, with km ticks and the located point(s).
    A marker is {km, label, color, primary, dashed}."""
    width, height = FRAME_W, 86.0
    drawing = Drawing(width, height)
    x0, x1, y = 13.0, width - 13.0, 50.0
    ink = BRAND_NAVY
    drawing.add(Line(x0, y, x1, y, strokeColor=ink, strokeWidth=2.2))

    def at(km: float) -> float:
        return x0 + (x1 - x0) * min(max(km / line_km, 0.0), 1.0) if line_km > 0 else x0

    step = _tick_step(line_km) if line_km > 0 else 0
    ticks = []
    km = step
    while step and km < line_km - step * 0.3:
        ticks.append(km)
        km += step
    marked = [at(float(m["km"])) for m in markers if m.get("km") is not None and math.isfinite(float(m["km"]))]
    for i, km in enumerate(ticks):
        x = at(km)
        drawing.add(Line(x, y - 3, x, y - 7, strokeColor=FORM_LINE, strokeWidth=0.6))
        if any(abs(x - mx) < 14 for mx in marked):
            continue  # the marker's own line would cross the label
        label = _num(km, 0) + (" km" if i == len(ticks) - 1 else "")
        drawing.add(String(x, y - 16, label, fontName="Helvetica", fontSize=7, fillColor=BRAND_MUTED, textAnchor="middle"))

    for (code, station, sub), x, anchor, tx in ((left, x0, "start", x0 - 11), (right, x1, "end", x1 + 11)):
        drawing.add(Rect(x - 11, y - 11, 22, 22, rx=3, ry=3, strokeColor=ink, strokeWidth=1.4, fillColor=colors.white))
        drawing.add(String(x, y - 3.6, code, fontName="Helvetica-Bold", fontSize=10, fillColor=ink, textAnchor="middle"))
        drawing.add(String(tx, y - 30, station, fontName="Helvetica-Bold", fontSize=7.5, fillColor=ink, textAnchor=anchor))
        if sub:
            drawing.add(String(tx, y - 40, sub, fontName="Helvetica", fontSize=7, fillColor=BRAND_MUTED, textAnchor=anchor))

    for marker in markers:
        if marker.get("km") is None or not math.isfinite(float(marker["km"])):
            continue
        x = at(float(marker["km"]))
        color = marker["color"]
        label_anchor = "middle"
        if x - x0 < 40:
            label_anchor = "start"
        elif x1 - x < 40:
            label_anchor = "end"
        if marker.get("primary"):
            if marker.get("dashed"):
                drawing.add(Line(x, y - 16, x, y + 16, strokeColor=color, strokeWidth=2.4, strokeDashArray=[3, 2]))
            else:
                drawing.add(Rect(x - 1.6, y - 16, 3.2, 32, rx=1.2, ry=1.2, strokeColor=None, fillColor=color))
            drawing.add(String(x, y + 22, marker["label"], fontName="Helvetica-Bold", fontSize=7.5,
                               fillColor=color, textAnchor=label_anchor))
        else:
            drawing.add(Line(x, y - 10, x, y + 10, strokeColor=color, strokeWidth=1.2))
            drawing.add(String(x, y + 22 if marker.get("above") else y - 27, marker["label"], fontName="Helvetica",
                               fontSize=7, fillColor=color, textAnchor=label_anchor))
    return drawing


def _hero(
    styles: dict, kicker: str, event: str, left: tuple[str, str], right: tuple[str, str], color: colors.Color,
) -> list:
    """The distance from both ends, the headline of page 1, under the event
    time."""
    hex_color = color.hexval().replace("0x", "#")
    head = Table(
        [[Paragraph(kicker, styles["kicker_row"]),
          Paragraph(event, ParagraphStyle("lr_event", parent=styles["note"], alignment=2))]],
        colWidths=[FRAME_W * 0.35, FRAME_W * 0.65],
    )
    head.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    table = Table(
        [[
            [Paragraph(f"<font color='{hex_color}'>{_safe_text(left[0])}</font>", styles["hero"]),
             Paragraph(_safe_text(left[1]), styles["hero_sub"])],
            [Paragraph(_safe_text(right[0]), styles["hero_right"]),
             Paragraph(_safe_text(right[1]), styles["hero_sub_right"])],
        ]],
        colWidths=[FRAME_W / 2, FRAME_W / 2],
    )
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return [head, Spacer(1, 4), table]


def _meta_line(styles: dict, items: list[str]) -> Paragraph:
    return Paragraph("    ·    ".join(_safe_text(item) for item in items if item), styles["note"])


def _fact_grid(styles: dict, facts: list[tuple[str, str, bool]]) -> Table:
    """Label over value, two columns. (label, value, monospace)."""
    cells = [
        [Paragraph(_safe_text(label), styles["label"]), Paragraph(value, styles["mono" if mono else "value"])]
        for label, value, mono in facts
    ]
    if len(cells) % 2:
        cells.append("")
    rows = [[cells[i], "", cells[i + 1]] for i in range(0, len(cells), 2)]
    table = Table(rows, colWidths=[FRAME_W * 0.48, FRAME_W * 0.04, FRAME_W * 0.48])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _rule_table(rows: list[list], col_widths: list[float], numeric_cols: tuple[int, ...] = ()) -> Table:
    """Header row, then rows separated by hairlines — the reports' plain table."""
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, BRAND_BORDER),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.3, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, BRAND_BORDER),
    ]
    table.setStyle(TableStyle(style))
    return table


def _field_form(styles: dict, aside: str, from_station: str) -> list:
    """The patrol fills this in; the report then goes to the fault archive."""
    labels = [
        "Nomor tower / span",
        f"Jarak terukur dari {from_station} (km)",
        "Selisih terhadap hasil laporan ini (m)",
        "Jenis kerusakan",
        "Dugaan penyebab",
        "Foto / berita acara",
        "Petugas patroli",
        "Tanggal dan tanda tangan",
    ]
    # Room to write by hand.
    cells = [[Paragraph(_safe_text(label), styles["label"]), Spacer(1, 17)] for label in labels]
    rows = [[cells[i], "", cells[i + 1]] for i in range(0, len(cells), 2)]
    table = Table(rows, colWidths=[FRAME_W * 0.48, FRAME_W * 0.04, FRAME_W * 0.48])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("LINEBELOW", (0, 0), (0, -1), 0.6, FORM_LINE),
        ("LINEBELOW", (2, 0), (2, -1), 0.6, FORM_LINE),
    ]))
    return [KeepTogether([*_section(styles, "LAPANGAN", "Konfirmasi lapangan", aside), Spacer(1, 2), table])]


def _verdict_cell(styles: dict, verdict: str) -> Paragraph:
    label, hex_color, _bg = VERDICT_STYLE[verdict]
    return Paragraph(f"<font color='{hex_color}'><b>{label}</b></font>", styles["td"])


def _chart_image(chart: ChartImage, max_w: float, max_h: float) -> Optional[Image]:
    try:
        raw = base64.b64decode(chart.image_b64)
    except Exception:
        return None
    image = Image(io.BytesIO(raw))
    aspect = image.imageHeight / image.imageWidth if image.imageWidth else 1.0
    width, height = max_w, max_w * aspect
    if height > max_h:
        height, width = max_h, max_h / aspect if aspect else max_w
    image.drawWidth, image.drawHeight = width, height
    image.hAlign = "CENTER"
    return image


def _figure(styles: dict, chart: Optional[ChartImage], title: str, aside: str, caption: Optional[str], max_h: float) -> list:
    if chart is None:
        return []
    image = _chart_image(chart, FRAME_W, max_h)
    if image is None:
        return [Paragraph(f"<i>Grafik {_safe_text(title)} tidak dapat dimuat.</i>", styles["body_muted"])]
    head = Table(
        [[Paragraph(f"<b>{_safe_text(title)}</b>", styles["value"]),
          Paragraph(_safe_text(aside), ParagraphStyle("lr_fig_aside", parent=styles["label"], alignment=2))]],
        colWidths=[FRAME_W * 0.6, FRAME_W * 0.4],
    )
    head.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    block = [head, image]
    if caption:
        block += [Spacer(1, 3), Paragraph(caption, styles["note"])]
    return [KeepTogether(block)]


def _document(buf: io.BytesIO, method: Method, header: _HeaderFooter, title: str) -> BaseDocTemplate:
    header_clearance = 22 * mm + 4 * mm
    footer_clearance = 10 * mm
    frame = Frame(
        MARGIN,
        MARGIN + footer_clearance,
        FRAME_W,
        PAGE_H - 2 * MARGIN - header_clearance - footer_clearance,
        leftPadding=0,
        rightPadding=0,
        showBoundary=0,
    )
    doc = BaseDocTemplate(
        buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
        title=title, author="TFA Analisis", subject=method.title,
    )
    doc.addPageTemplates([PageTemplate(id="location", frames=[frame], onPage=header.on_page)])
    return doc


# ---------------------------------------------------------------------------
# TWS
# ---------------------------------------------------------------------------

class LocationChartsRequest(BaseModel):
    charts: list[ChartImage] = []


def _tws_endpoints(result: dict) -> tuple[dict, dict]:
    endpoints = result.get("endpoints") or []
    x = next((ep for ep in endpoints if ep.get("role") == "X"), endpoints[0] if endpoints else {})
    y = next((ep for ep in endpoints if ep.get("role") == "Y"), endpoints[1] if len(endpoints) > 1 else {})
    return x, y


def _station(ep: dict, fallback: str) -> str:
    return ep.get("station_display_name") or ep.get("station_name") or fallback


def build_tws_pdf(payload: dict, charts: list[ChartImage], analysis_id: str) -> bytes:
    method = TWS
    styles = _styles(method)
    result = (payload.get("results") or [{}])[0]
    x_ep, y_ep = _tws_endpoints(result)
    x_name, y_name = _station(x_ep, "Terminal X"), _station(y_ep, "Terminal Y")
    line_km = float(result.get("line_length_km") or 0.0)
    circuit = result.get("circuit_name") or result.get("segment_name") or payload.get("station_name") or "—"
    sel = result.get("sel_type_d") or {}
    x_km, y_km = x_ep.get("fault_distance_km"), y_ep.get("fault_distance_km")
    local = result.get("result_time_local")
    moment = datetime.fromtimestamp(float(local), tz=timezone.utc) if local else None
    both_locked = bool(x_ep.get("gps_locked")) and bool(y_ep.get("gps_locked"))
    sample_rate = x_ep.get("sample_rate_hz")

    buf = io.BytesIO()
    header = _HeaderFooter(
        circuit.replace("_", " "), "TWS FL", _short_id(analysis_id), method.subtitle, _format_datetime(),
        kicker="LAPORAN LOKASI GANGGUAN", accent=method.accent,
        footer_left=f"{method.footer} · {payload.get('source_file') or 'TWS'} · analisa {_short_id(analysis_id)}",
    )
    doc = _document(buf, method, header, f"Laporan lokasi gangguan — {circuit}")

    delta_t = sel.get("delta_t_us")
    steps = [
        ("Rekaman X", f"{_safe_text(x_name)} · #{_safe_text(x_ep.get('record_number') or '—')}"
                      f"{' · GPS terkunci' if x_ep.get('gps_locked') else ' · GPS tidak terkunci'}"),
        ("Rekaman Y", f"{_safe_text(y_name)} · #{_safe_text(y_ep.get('record_number') or '—')}"
                      f"{' · GPS terkunci' if y_ep.get('gps_locked') else ' · GPS tidak terkunci'}"),
        ("Waktu tiba", f"gelombang pertama di tiap ujung; tX − tY = {_num(delta_t, 2, 'µs')}"),
        ("Type D", "m = ½ · (L + (tX − tY) · v)"),
        ("Hasil", f"{_num(x_km, 2, 'km')} dari X (Qualitrol)"),
    ]

    story: list = [_method_band(styles, method, steps), Spacer(1, 10)]
    story += _hero(
        styles, "LOKASI GANGGUAN",
        f"<b>Kejadian</b> {_safe_text(_event_time(moment))} WIB · "
        f"{'waktu GPS, kedua ujung terkunci' if both_locked else 'GPS tidak terkunci di salah satu ujung'}",
        (_num(x_km, 2, "km"), f"dari {x_name} (X)"),
        (_num(y_km, 2, "km"), f"dari {y_name} (Y)"),
        method.accent,
    )
    story.append(Spacer(1, 4))
    markers = [{"km": x_km, "label": f"Qualitrol {_num(x_km, 2)}", "color": method.accent, "primary": True}]
    if sel.get("m_from_x_km") is not None:
        markers.append({"km": sel["m_from_x_km"], "label": f"Hitungan TFA {_num(sel['m_from_x_km'], 2)}",
                        "color": BRAND_SLATE})
    story.append(_line_diagram(
        line_km,
        ("X", x_name, f"bay {x_ep.get('feeder_display_name') or x_ep.get('feeder_name') or '—'}"),
        ("Y", y_name, f"bay {y_ep.get('feeder_display_name') or y_ep.get('feeder_name') or '—'}"),
        markers,
    ))
    story.append(_meta_line(styles, [
        f"Panjang line {_num(line_km, 2, 'km')}",
        f"Resolusi 1 sampel = {_num(result.get('sample_distance_km'), 2, 'km')}",
        "GPS terkunci di X dan Y" if both_locked else "GPS tidak terkunci di salah satu ujung",
    ]))
    story.append(Spacer(1, 10))

    velocity = sel.get("velocity_km_s") or result.get("velocity_km_s")
    story += _section(styles, "PERHITUNGAN", "Data perhitungan")
    story.append(_fact_grid(styles, [
        ("Rumus (Type D, jarak dari X)", "m = ½ · (L + (tX − tY) · v)", False),
        ("Selisih waktu tiba tX − tY", _num(delta_t, 2, "µs"), False),
        ("Kecepatan rambat v", f"{_num(velocity, 0, 'km/s')} (velocity factor {_num(result.get('velocity_factor'), 2, '%')})", False),
        ("Panjang line L", _num(line_km, 2, "km"), False),
    ]))
    story.append(Spacer(1, 10))
    story += _field_form(styles, "Diisi tim patroli, lalu dilampirkan ke arsip gangguan", x_name)

    # Attachments: a page of their own unless the form already moved to one.
    waveform = {chart.id: chart for chart in charts}
    has_waveform = any(chart_id in waveform for chart_id in ("tws_waveform_x", "tws_waveform_y"))
    if sel or has_waveform:
        story.append(CondPageBreak(PAGE_H * 0.45))
        story.append(Spacer(1, 6))
        story += _section(styles, "LAMPIRAN", "Perbandingan dan waveform" if sel and has_waveform
                          else "Perbandingan dua perhitungan" if sel else "Waveform traveling wave")

    if sel:
        story.append(Paragraph("<b>Perbandingan dua perhitungan</b>", styles["value"]))
        story.append(Spacer(1, 3))
        rows = [[
            Paragraph("Terminal", styles["th"]), Paragraph("Qualitrol", styles["th_right"]),
            Paragraph("Hitungan TFA", styles["th_right"]), Paragraph("Selisih", styles["th_right"]),
        ]]
        for code, name, device_km, own_km, delta_km in (
            ("X", x_name, sel.get("qualitrol_x_km"), sel.get("m_from_x_km"), sel.get("delta_x_km")),
            ("Y", y_name, sel.get("qualitrol_y_km"), sel.get("m_from_y_km"), sel.get("delta_y_km")),
        ):
            rows.append([
                Paragraph(f"{code} · {_safe_text(name)}", styles["td"]),
                Paragraph(_num(device_km, 3, "km"), styles["td_right"]),
                Paragraph(_num(own_km, 3, "km"), styles["td_right"]),
                Paragraph(_num(delta_km * 1000 if delta_km is not None else None, 0, "m", signed=True), styles["td_right"]),
            ])
        story.append(_rule_table(rows, [FRAME_W * 0.4, FRAME_W * 0.2, FRAME_W * 0.2, FRAME_W * 0.2]))
        spread_m = abs(sel.get("delta_x_km") or 0.0) * 1000
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            f"Kedua perhitungan berselisih {_num(spread_m, 0, 'm')}. Akurasi bawaan traveling wave sekitar "
            "±0,2 µs (±60 m); andongan, beda ketinggian, dan struktur tower menambah selisih. "
            "Periksa rentang di antara kedua titik.",
            styles["note"],
        ))
        story.append(Spacer(1, 10))

    if has_waveform:
        story.append(Paragraph(
            "Puncak pertama setelah garis penanda M adalah gelombang yang tiba dari titik gangguan. "
            "Sumbu mendatar menunjukkan jarak dari penanda trigger terkoreksi (km). "
            f"Jejak merah, hijau, dan biru adalah fasa R, S, dan T{'; laju sampel ' + _num(sample_rate / 1e6, 2, 'MHz') if sample_rate else ''}.",
            styles["note"],
        ))
        story.append(Spacer(1, 6))
        for code, ep, chart_id in (("X", x_ep, "tws_waveform_x"), ("Y", y_ep, "tws_waveform_y")):
            rate = ep.get("sample_rate_hz")
            story += _figure(
                styles, waveform.get(chart_id), f"Terminal {code} · {_station(ep, code)}",
                f"rekaman #{ep.get('record_number') or '—'} · {_num((rate or 0) / 1e6, 2, 'MHz') if rate else '—'}",
                None, 74 * mm,
            )
            story.append(Spacer(1, 6))

    doc.build(story)
    return buf.getvalue()


@router.post("/tws/{analysis_id}")
async def tws_report(analysis_id: str, body: LocationChartsRequest):
    payload = load_analysis(analysis_id)
    if payload is None or payload.get("source_type") != "tws_cdb":
        raise HTTPException(status_code=404, detail="TWS analysis not found or expired.")
    loop = asyncio.get_event_loop()
    pdf = await loop.run_in_executor(None, build_tws_pdf, payload, body.charts, analysis_id)
    return _pdf_response(pdf, f"laporan_tws_{_short_id(analysis_id)}.pdf")


# ---------------------------------------------------------------------------
# Double-ended FL
# ---------------------------------------------------------------------------

class DoubleEndedReportRequest(DoubleEndedComputeRequest):
    """The DE-FL page's inputs, recomputed server-side for the report, plus
    what only the page knows: the record names, where the shift came from,
    whether the user confirmed it on the overlay, and the loop step 4 named."""
    record_name_a: Optional[str] = None
    record_name_b: Optional[str] = None
    shift_source: str = "manual"            # estimate | residual_search | manual
    sync_confirmed: bool = False
    suggested_loop: Optional[str] = None
    charts: list[ChartImage] = []


_RESIDUAL_LIMIT = 0.15  # the page's own reliability threshold (Im(m), KVL)
_FAULT_POINT_LIMIT = 0.2  # F7.4


def _line_text(selection) -> Optional[str]:
    if selection is None:
        return None
    why = _SELECTION_METHOD.get(selection.method, selection.method)
    others = "; ".join(
        f"{line.key} {_LINE_STATE.get(line.state, line.state.lower())}"
        for line in selection.lines if line.key != selection.selected
    )
    return f"line {selection.selected} (dipilih otomatis: {why}{'; ' + others if others else ''})"


def _loop_text(loop: str) -> str:
    label = _LOOP_LABEL.get(loop, loop)
    basis = "urutan negatif" if loop in GROUND_LOOPS else "fasa-fasa"
    return f"{label} ({loop}, {basis})"


def _defl_checks(result: dict, body: DoubleEndedReportRequest) -> list[tuple[str, str, str, str]]:
    """(check, value, requirement, verdict) — every row a rule the result must meet."""
    kvl = float(result.get("kvl_residual") or 0.0)
    im = abs(float(result.get("m_residual_imag") or 0.0))
    ratio = result.get("fault_point_ratio")
    rows = [
        ("Residual KVL", _num(kvl, 3), f"≤ {_num(_RESIDUAL_LIMIT, 2)}", "pass" if kvl <= _RESIDUAL_LIMIT else "fail"),
        ("Residual Im(m)", _num(im, 3), f"≤ {_num(_RESIDUAL_LIMIT, 2)}", "pass" if im <= _RESIDUAL_LIMIT else "fail"),
        (
            "Arus gangguan pada besaran yang dipakai (F7.4)",
            f"{_num(ratio, 2)}× arus urutan positif di titik gangguan" if ratio is not None else "tidak bisa diperiksa",
            f"≥ {_num(_FAULT_POINT_LIMIT, 1)}×",
            "pending" if ratio is None else ("pass" if ratio >= _FAULT_POINT_LIMIT else "fail"),
        ),
        (
            "Sinkronisasi waktu (F7.4)",
            f"geser B {_num(body.manual_shift_ms, 1, 'ms')}, {_SHIFT_SOURCE.get(body.shift_source, body.shift_source)}",
            "dikonfirmasi di grafik overlay",
            "pass" if body.sync_confirmed else "pending",
        ),
    ]
    return rows


def _defl_reasons(checks: list[tuple[str, str, str, str]], result: dict) -> list[str]:
    reasons = []
    residuals = [(name, value) for name, value, _req, verdict in checks if name.startswith("Residual") and verdict != "pass"]
    if residuals:
        listed = " dan ".join(f"{name.removeprefix('Residual ')} {value}" for name, value in residuals)
        reasons.append(f"Residual {listed} (syarat ≤ {_num(_RESIDUAL_LIMIT, 2)}): persamaan kedua ujung belum "
                       "konsisten. Penyebab paling umum adalah waktu kedua rekaman yang belum sinkron.")
    for name, value, requirement, verdict in checks:
        if verdict == "pass" or name.startswith("Residual"):
            continue
        if name.startswith("Arus gangguan"):
            reasons.append("Gangguan hampir tidak menarik arus pada loop ini, sehingga jarak berapa pun memenuhi "
                           "persamaan. Pilih loop fasa yang terganggu." if verdict == "fail"
                           else "Kecukupan arus gangguan pada loop ini tidak bisa diperiksa.")
        elif name.startswith("Sinkronisasi"):
            reasons.append("Geser waktu rekaman B belum dikonfirmasi di grafik overlay.")
    if result.get("distance_pct") is not None and not 0 <= float(result["distance_pct"]) <= 100:
        reasons.append("Jarak hasil berada di luar line: periksa sinkronisasi, rasio CT/PT, urutan fasa, dan loop.")
    return reasons


def build_defl_pdf(payload_a: dict, payload_b: dict, body: DoubleEndedReportRequest) -> bytes:
    method = DOUBLE_ENDED
    styles = _styles(method)
    result = _run_compute(payload_a, payload_b, body)
    checks = _defl_checks(result, body)
    reasons = _defl_reasons(checks, result)
    reliable = not reasons

    _scoped_a, selection_a = scope_payload_with_selection(payload_a)
    _scoped_b, selection_b = scope_payload_with_selection(payload_b)
    station_a = payload_a.get("station_name") or "Terminal A"
    station_b = payload_b.get("station_name") or "Terminal B"
    record_a = body.record_name_a or payload_a.get("rec_dev_id") or "—"
    record_b = body.record_name_b or payload_b.get("rec_dev_id") or "—"
    line_a = selection_a.selected if selection_a else None
    line_b = selection_b.selected if selection_b else None
    line_km = float(body.line_len_km)
    distance = float(result["distance_km"])
    loop = body.loop
    loop_label = _LOOP_LABEL.get(loop, loop)
    windows = int(result.get("selected_window_count") or 0)
    weak = list(result.get("weak_infeed_terminals") or [])
    moment = _parse_iso(payload_a.get("trigger_time_iso"))

    buf = io.BytesIO()
    title = f"{station_a} – {station_b}"
    header = _HeaderFooter(
        title, f"loop {loop_label}", f"A {_short_id(body.analysis_id_a)} · B {_short_id(body.analysis_id_b)}",
        method.subtitle, _format_datetime(),
        kicker="LAPORAN LOKASI GANGGUAN", accent=method.accent,
        footer_left=f"{method.footer} · {_clip(record_a)} + {_clip(record_b)}",
    )
    doc = _document(buf, method, header, f"Laporan lokasi gangguan — {title}")

    if body.suggested_loop and body.suggested_loop == loop:
        loop_source = "dari fasa terganggu (langkah 4)"
    elif body.suggested_loop:
        loop_source = f"dipilih pengguna; langkah 4 menyarankan {_LOOP_LABEL.get(body.suggested_loop, body.suggested_loop)}"
    else:
        loop_source = "dipilih pengguna"
    steps = [
        ("Rekaman A", f"{_safe_text(station_a)} · {_safe_text(_clip(record_a))}{' · line ' + _safe_text(line_a) if line_a else ''}"),
        ("Rekaman B", f"{_safe_text(station_b)} · {_safe_text(_clip(record_b))}{' · line ' + _safe_text(line_b) if line_b else ''}"),
        ("Sinkronisasi", f"geser B {_num(body.manual_shift_ms, 1, 'ms')} · "
                         f"{'terkonfirmasi' if body.sync_confirmed else 'belum dikonfirmasi'}"),
        ("Loop", f"{_safe_text(loop_label)} · {_safe_text(loop_source)}"),
        ("KVL", f"{windows} jendela simultan selama gangguan"),
        ("Hasil", f"{_num(distance, 2, 'km')} dari A{'' if reliable else ' (belum andal)'}"),
    ]
    story: list = [_method_band(styles, method, steps), Spacer(1, 10)]

    if not reliable:
        banner = Table(
            [[Paragraph("<font color='#991b1b'><b>Hasil belum andal. Jangan dipakai untuk patroli.</b></font>",
                        styles["value"])]]
            + [[Paragraph(_safe_text(reason), styles["note"])] for reason in reasons],
            colWidths=[FRAME_W],
        )
        banner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#fca5a5")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (0, 0), 7),
            ("BOTTOMPADDING", (0, -1), (0, -1), 7),
        ]))
        story += [banner, Spacer(1, 10)]

    number_color = method.accent if reliable else colors.HexColor("#6b7280")
    story += _hero(
        styles, "LOKASI GANGGUAN" if reliable else "LOKASI GANGGUAN (BELUM ANDAL)",
        f"<b>Kejadian</b> {_safe_text(_event_time(moment))} · jam DFR {_safe_text(station_a)}",
        (_num(distance, 2, "km"), f"dari {station_a} (A)"),
        (_num(line_km - distance, 2, "km"), f"dari {station_b} (B)"),
        number_color,
    )
    story.append(Spacer(1, 4))
    story.append(_line_diagram(
        line_km,
        ("A", station_a, f"{'line ' + line_a + ' · ' if line_a else ''}rekaman {_clip(record_a)}"),
        ("B", station_b, f"{'line ' + line_b + ' · ' if line_b else ''}rekaman {_clip(record_b)}"),
        [{"km": distance, "label": f"Dua ujung {_num(distance, 2)}", "color": number_color,
          "primary": True, "dashed": not reliable}],
    ))
    current_label = (
        f"Arus loop {loop_label} di titik gangguan" if loop not in GROUND_LOOPS
        else f"Arus fasa {_PHASE_LABEL.get(loop[1], loop[1])} di titik gangguan"
    )
    story.append(_meta_line(styles, [
        f"Panjang line {_num(line_km, 2, 'km')}",
        f"{_num(result.get('distance_pct'), 1, '%')} dari A",
        f"{current_label} {_num((result.get('fault_current_a') or 0) / 1000, 2, 'kA')}",
        f"Loop {_loop_text(loop)}",
    ]))
    story.append(Spacer(1, 10))

    rows = [[Paragraph(text, styles["th"]) for text in ("Pemeriksaan", "Nilai", "Syarat", "Hasil")]]
    for name, value, requirement, verdict in checks:
        rows.append([
            Paragraph(_safe_text(name), styles["td"]), Paragraph(_safe_text(value), styles["td"]),
            Paragraph(_safe_text(requirement), styles["td"]), _verdict_cell(styles, verdict),
        ])
    notes = [
        f"Sebaran jarak antarjendela {_num(result.get('distance_spread_km'), 2, 'km')} dari {windows} jendela "
        "(informasi: kelompok yang rapat berarti hasilnya stabil sepanjang gangguan)."
    ]
    for terminal in weak:
        ratio = result.get(f"fault_contribution_ratio_{terminal.lower()}")
        notes.append(
            f"Ujung {terminal} weak infeed: kontribusi arus gangguannya {_num(ratio, 2)}× arus beban. Lokasi dua "
            f"ujung tetap sah karena memakai tegangan ujung {terminal}."
        )
    story.append(KeepTogether([
        *_section(styles, "KUALITAS", "Kualitas hasil"),
        _rule_table(rows, [FRAME_W * 0.3, FRAME_W * 0.33, FRAME_W * 0.24, FRAME_W * 0.13]),
        Spacer(1, 3),
        *[Paragraph(_safe_text(note), styles["note"]) for note in notes],
    ]))
    story.append(Spacer(1, 10))

    story += _field_form(
        styles,
        "Diisi tim patroli, lalu dilampirkan ke arsip gangguan" if reliable else "Isi setelah hasil andal",
        station_a,
    )

    # Attachments: a page of their own unless the form already moved to one.
    story.append(CondPageBreak(PAGE_H * 0.45))
    story.append(Spacer(1, 6))
    story += _section(styles, "LAMPIRAN", "Sebaran jarak dan data masukan")
    by_id = {chart.id: chart for chart in body.charts}
    story += _figure(
        styles, by_id.get("defl_histogram"), "Jarak dua ujung per jendela evaluasi", f"loop {loop_label}",
        "Setiap batang adalah hasil satu jendela waktu selama gangguan. Kelompok yang rapat berarti hasilnya "
        "stabil sepanjang gangguan; ini bukan skor keyakinan.",
        62 * mm,
    )
    story.append(Spacer(1, 8))

    shift_text = (
        f"{_num(body.manual_shift_ms, 1, 'ms')} ({_SHIFT_SOURCE.get(body.shift_source, body.shift_source)}, "
        f"{'dikonfirmasi di grafik' if body.sync_confirmed else 'belum dikonfirmasi'})"
    )
    inputs = [
        ("Terminal A", f"{station_a} · rekaman {record_a}" + (f" · {_line_text(selection_a)}" if selection_a else "")),
        ("Terminal B", f"{station_b} · rekaman {record_b}" + (f" · {_line_text(selection_b)}" if selection_b else "")),
        ("Line", f"{_num(line_km, 2, 'km')} · R1 {_num(body.r1_ohm_per_km, 3, 'Ω/km')} · "
                 f"X1 {_num(body.x1_ohm_per_km, 3, 'Ω/km')}"),
        ("Loop", f"{_loop_text(loop)}, {loop_source}"),
        ("CT/PT dan polaritas", f"rasio sesuai CFG kedua rekaman; {_inversions(body)}"),
        ("Geser waktu rekaman B", shift_text),
        ("Metode", "KVL per jendela selama kedua ujung mengalirkan arus gangguan, digabung kuadrat terkecil; "
                   "loop tanah memakai urutan negatif; tahanan gangguan dan K0 tidak ikut"),
    ]
    rows = [[Paragraph("Masukan", styles["th"]), Paragraph("Nilai", styles["th"])]]
    rows += [[Paragraph(_safe_text(label), styles["label"]), Paragraph(_safe_text(value), styles["td"])]
             for label, value in inputs]
    story += _section(styles, "MASUKAN", "Data masukan")
    story.append(_rule_table(rows, [FRAME_W * 0.24, FRAME_W * 0.76]))

    doc.build(story)
    return buf.getvalue()


def _inversions(body: DoubleEndedReportRequest) -> str:
    changes = [
        text for flag, text in (
            (body.invert_i_a, "arus A dibalik"),
            (body.invert_i_b, "arus B dibalik"),
            (body.invert_phase_sequence_a, "urutan fasa A dibalik"),
            (body.invert_phase_sequence_b, "urutan fasa B dibalik"),
        ) if flag
    ]
    return ", ".join(changes) if changes else "normal, tidak dibalik"


@router.post("/de-fl")
async def de_fl_report(body: DoubleEndedReportRequest):
    payload_a = load_analysis(body.analysis_id_a)
    payload_b = load_analysis(body.analysis_id_b)
    if payload_a is None or payload_b is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")
    loop = asyncio.get_event_loop()
    pdf = await loop.run_in_executor(None, build_defl_pdf, payload_a, payload_b, body)
    return _pdf_response(pdf, f"laporan_defl_{_short_id(body.analysis_id_a)}_{_short_id(body.analysis_id_b)}.pdf")


def _pdf_response(pdf: bytes, filename: str) -> StreamingResponse:
    return StreamingResponse(
        io.BytesIO(pdf),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
