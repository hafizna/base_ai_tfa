"""PDF report generator — structured A4 PDF for fault analysis (Opsi A2).

Backend assembles metadata, fault classification, electrical params, and AI
fault analysis into a ReportLab PDF. Charts are sent by the frontend as base64
PNGs (already-rendered Plotly figures) and embedded directly — avoids
duplicating chart logic in matplotlib.

Layout (relay 21 — Distance):
    Header bar (PLN logo + title + metadata)
    KONKLUSI callout (fault summary)
    Section 1 — Metadata Recording (table)
    Section 2 — Parameter Elektrikal (table)
    Section 3 — Impedance Locus (chart)
    Section 4 — Waveform + Locus Events (chart)
    Section 5 — Digital Status Snapshot (chart, optional)
    Footer (analysis ID + page N of M)
"""

from __future__ import annotations

import asyncio
import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from ..storage import load_analysis
from .relay_21 import _compute_electrical_params, _compute_fault_classification

router = APIRouter(prefix="/api/report", tags=["report"])

ASSETS_DIR = Path(__file__).parent.parent / "assets"
LOGO_PATH = ASSETS_DIR / "pln-1429x512.png"

BRAND_NAVY = colors.HexColor("#0f172a")
BRAND_BLUE = colors.HexColor("#2563eb")
BRAND_SLATE = colors.HexColor("#475569")
BRAND_MUTED = colors.HexColor("#64748b")
BRAND_BORDER = colors.HexColor("#cbd5e1")
BRAND_BG_SOFT = colors.HexColor("#f1f5f9")
BRAND_BG_HIGHLIGHT = colors.HexColor("#eff6ff")

SEVERITY_HEX = {
    "verdict":  "#1d4ed8",
    "critical": "#b91c1c",
    "warning":  "#b45309",
    "notable":  "#0f766e",
    "info":     "#475569",
}
SEVERITY_BG = {
    "verdict":  colors.HexColor("#eff6ff"),
    "critical": colors.HexColor("#fef2f2"),
    "warning":  colors.HexColor("#fffbeb"),
    "notable":  colors.HexColor("#f0fdfa"),
    "info":     colors.HexColor("#f8fafc"),
}

PAGE_W, PAGE_H = A4
MARGIN = 15 * mm


class ChartImage(BaseModel):
    id: str
    title: str
    image_b64: str  # base64 PNG (no data: prefix)


class SoeEvent(BaseModel):
    time_ms: float
    rel_ms: Optional[float] = None
    channel: str
    state: int
    category: Optional[str] = None
    label: Optional[str] = None


class ReportRequest(BaseModel):
    relay_type: str
    ai_analysis: Optional[dict] = None
    charts: list[ChartImage] = []
    soe_events: list[SoeEvent] = []


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {}

    styles["kicker"] = ParagraphStyle(
        "kicker",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        textColor=BRAND_BLUE,
        leading=10,
        spaceAfter=2,
    )
    styles["title"] = ParagraphStyle(
        "title",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        textColor=BRAND_NAVY,
        leading=18,
        spaceAfter=2,
    )
    styles["subtitle"] = ParagraphStyle(
        "subtitle",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        textColor=BRAND_SLATE,
        leading=12,
    )
    styles["section"] = ParagraphStyle(
        "section",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=BRAND_NAVY,
        leading=12,
        spaceBefore=4,
        spaceAfter=4,
    )
    styles["section_kicker"] = ParagraphStyle(
        "section_kicker",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7,
        textColor=BRAND_BLUE,
        leading=9,
        spaceAfter=1,
    )
    styles["body"] = ParagraphStyle(
        "body",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=BRAND_NAVY,
        leading=12,
        alignment=TA_LEFT,
    )
    styles["body_muted"] = ParagraphStyle(
        "body_muted",
        parent=styles["body"],
        textColor=BRAND_MUTED,
        fontSize=8.5,
    )
    styles["conclusion_kicker"] = ParagraphStyle(
        "conclusion_kicker",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=BRAND_BLUE,
        leading=10,
        spaceAfter=2,
    )
    styles["conclusion_head"] = ParagraphStyle(
        "conclusion_head",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=BRAND_NAVY,
        leading=16,
        spaceAfter=4,
    )
    styles["conclusion_body"] = ParagraphStyle(
        "conclusion_body",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        textColor=BRAND_NAVY,
        leading=13,
    )
    return styles


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_datetime() -> str:
    months_id = [
        "Januari", "Februari", "Maret", "April", "Mei", "Juni",
        "Juli", "Agustus", "September", "Oktober", "November", "Desember",
    ]
    now = datetime.now()
    return f"{now.day} {months_id[now.month - 1]} {now.year}, {now.strftime('%H:%M')} WIB"


def _format_number(value, suffix: str = "", digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _format_duration_ms(time_arr: list) -> str:
    if not time_arr or len(time_arr) < 2:
        return "—"
    return f"{(time_arr[-1] - time_arr[0]) * 1000:.1f} ms"


RELAY_LABELS = {
    "21": "21 — Distance Protection",
    "87L": "87L — Line Differential",
    "CCP": "CCP / Stub Differential",
    "87T": "87T — Transformer Differential",
    "OCR": "50/51 — Overcurrent",
    "REF": "REF / GFR / SBEF",
    "SBEF": "SBEF / Ground Fault",
}


# ---------------------------------------------------------------------------
# Header / footer painter (drawn on every page via PageTemplate)
# ---------------------------------------------------------------------------

class _HeaderFooter:
    def __init__(self, station: str, device: str, analysis_id: str, relay_label: str, timestamp: str):
        self.station = station or "—"
        self.device = device or "—"
        self.analysis_id = analysis_id
        self.relay_label = relay_label
        self.timestamp = timestamp

    def on_page(self, canvas: Canvas, _doc):
        canvas.saveState()

        # Header band: logo (left) + title block (right)
        header_top = PAGE_H - MARGIN
        header_height = 22 * mm

        if LOGO_PATH.exists():
            logo_h = 12 * mm
            logo_w = logo_h * (1429 / 512)  # preserve aspect ratio
            canvas.drawImage(
                str(LOGO_PATH),
                MARGIN,
                header_top - logo_h - 1 * mm,
                width=logo_w,
                height=logo_h,
                preserveAspectRatio=True,
                mask="auto",
            )
            text_left = MARGIN + logo_w + 5 * mm
        else:
            text_left = MARGIN

        canvas.setFillColor(BRAND_BLUE)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.drawString(text_left, header_top - 3.5 * mm, "LAPORAN ANALISIS GANGGUAN COMTRADE")

        canvas.setFillColor(BRAND_NAVY)
        canvas.setFont("Helvetica-Bold", 12)
        canvas.drawString(text_left, header_top - 8 * mm, self.station)

        canvas.setFillColor(BRAND_SLATE)
        canvas.setFont("Helvetica", 8.5)
        canvas.drawString(text_left, header_top - 12 * mm, f"{self.relay_label}  |  {self.device}")

        # Right meta block (analysis ID + timestamp)
        canvas.setFillColor(BRAND_MUTED)
        canvas.setFont("Helvetica-Bold", 6.5)
        canvas.drawRightString(PAGE_W - MARGIN, header_top - 3.5 * mm, "ANALYSIS ID")
        canvas.setFillColor(BRAND_NAVY)
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(PAGE_W - MARGIN, header_top - 6 * mm, self.analysis_id)

        canvas.setFillColor(BRAND_MUTED)
        canvas.setFont("Helvetica-Bold", 6.5)
        canvas.drawRightString(PAGE_W - MARGIN, header_top - 10 * mm, "DICETAK")
        canvas.setFillColor(BRAND_NAVY)
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(PAGE_W - MARGIN, header_top - 12.5 * mm, self.timestamp)

        # Divider under header
        canvas.setStrokeColor(BRAND_BORDER)
        canvas.setLineWidth(0.6)
        divider_y = header_top - header_height
        canvas.line(MARGIN, divider_y, PAGE_W - MARGIN, divider_y)

        # Footer
        footer_y = MARGIN - 5 * mm
        canvas.setStrokeColor(BRAND_BORDER)
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN, footer_y + 4 * mm, PAGE_W - MARGIN, footer_y + 4 * mm)

        canvas.setFillColor(BRAND_MUTED)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(MARGIN, footer_y, f"Analysis ID: {self.analysis_id}")
        canvas.drawRightString(
            PAGE_W - MARGIN,
            footer_y,
            f"Halaman {canvas.getPageNumber()}",
        )
        canvas.drawCentredString(PAGE_W / 2, footer_y, "PLN — UIT JBT")

        canvas.restoreState()


# ---------------------------------------------------------------------------
# Conclusion (KONKLUSI) callout — the headline summary at the top
# ---------------------------------------------------------------------------

def _build_conclusion(
    styles: dict,
    fault_class: dict,
    ai_analysis: Optional[dict],
    elec: dict,
) -> Table:
    fault_code = fault_class.get("fault_code", "—")
    phases_label = fault_class.get("phases_label", "—")
    zone = fault_class.get("zone") or "—"
    trip_type = fault_class.get("trip_type") or "—"
    fault_ms = fault_class.get("fault_ms", 0.0)
    ar_status = fault_class.get("ar_status")
    ar_label = {
        "successful": "A/R BERHASIL",
        "failed": "A/R GAGAL",
        None: "A/R N/A",
    }.get(ar_status, "—")

    # Build narrative paragraph from AI analysis if available.
    # Shape comes from AIFaultResult (schemas.py): cause_ranking, fault_type,
    # overall_confidence, evidence. Legacy keys kept as fallback.
    narrative_lines = []
    if ai_analysis:
        cause_label = None
        cause_code = None
        confidence = None

        cause_ranking = ai_analysis.get("cause_ranking") or []
        if isinstance(cause_ranking, list) and cause_ranking:
            top = cause_ranking[0] or {}
            cause_label = top.get("label") or top.get("cause")
            cause_code = top.get("cause")
            confidence = top.get("confidence")

        # Legacy fallbacks
        if cause_label is None:
            cause_label = (
                ai_analysis.get("predicted_cause")
                or ai_analysis.get("cause")
                or ai_analysis.get("fault_cause")
            )
        if confidence is None:
            confidence = ai_analysis.get("overall_confidence") or ai_analysis.get("confidence")

        if cause_label:
            line = f"<b>Penyebab (AI):</b> {cause_label}"
            if cause_code and cause_code != cause_label:
                line += f" <font color='#64748b'>[{cause_code}]</font>"
            if isinstance(confidence, (int, float)):
                pct = confidence * 100 if confidence <= 1 else confidence
                line += f" <font color='#64748b'>(confidence {pct:.0f}%)</font>"
            narrative_lines.append(line)

        fault_type = ai_analysis.get("fault_type")
        if isinstance(fault_type, str) and fault_type.strip():
            ft_label = {
                "transient": "Transient (sementara)",
                "permanent": "Permanent (menetap)",
            }.get(fault_type.lower(), fault_type)
            narrative_lines.append(f"<b>Jenis kejadian:</b> {ft_label}")

        # Verdict-level evidence promoted to KONKLUSI summary
        evidence = ai_analysis.get("evidence") or []
        verdict_text = None
        if isinstance(evidence, list):
            for item in evidence:
                if isinstance(item, dict) and item.get("severity") == "verdict":
                    txt = (item.get("text") or "").strip()
                    if txt:
                        verdict_text = txt
                        break
        if verdict_text:
            narrative_lines.append(verdict_text)
        else:
            reasoning = (
                ai_analysis.get("reasoning")
                or ai_analysis.get("explanation")
                or ai_analysis.get("narrative")
            )
            if isinstance(reasoning, str) and reasoning.strip():
                narrative_lines.append(reasoning.strip())

    z_inception = elec.get("z_at_inception_ohm")
    z_angle = elec.get("z_angle_deg")
    if z_inception is not None:
        z_line = f"<b>Impedansi saat inception:</b> {z_inception:.2f} Ω"
        if z_angle is not None:
            z_line += f" ∠ {z_angle:.1f}°"
        narrative_lines.append(z_line)

    # Right-side chip column: vertical [label, value] pairs to fit the
    # narrow ~77mm slot inside the conclusion box without clipping.
    summary_chips = [
        ("JENIS GANGGUAN", fault_code),
        ("FASA", phases_label),
        ("ZONE", str(zone)),
        ("TRIP TYPE", str(trip_type)),
        ("DURASI GANGGUAN", f"{fault_ms:.1f} ms"),
        ("AUTORECLOSE", ar_label),
    ]

    chips_data = []
    for label, val in summary_chips:
        chips_data.append([
            Paragraph(f"<font size=6.5 color='#64748b'><b>{label}</b></font>", styles["body"]),
            Paragraph(f"<font size=9.5 color='#0f172a'><b>{val}</b></font>", styles["body"]),
        ])

    chips_table = Table(
        chips_data,
        colWidths=[30 * mm, 40 * mm],
        hAlign="LEFT",
    )
    chips_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, BRAND_BORDER),
    ]))

    narrative_cells = [
        Paragraph("KONKLUSI", styles["conclusion_kicker"]),
        Paragraph(f"{fault_code} — {phases_label}", styles["conclusion_head"]),
    ]
    for line in narrative_lines:
        narrative_cells.append(Paragraph(line, styles["conclusion_body"]))
        narrative_cells.append(Spacer(1, 2))

    inner = Table(
        [[narrative_cells, chips_table]],
        colWidths=[(PAGE_W - 2 * MARGIN) * 0.55 - 4, (PAGE_W - 2 * MARGIN) * 0.45 - 4],
    )
    inner.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_BG_HIGHLIGHT),
        ("BOX", (0, 0), (-1, -1), 0.6, BRAND_BLUE),
    ]))
    return inner


# ---------------------------------------------------------------------------
# Section tables
# ---------------------------------------------------------------------------

def _section_header(styles: dict, kicker: str, title: str) -> list:
    return [
        Paragraph(kicker, styles["section_kicker"]),
        Paragraph(title, styles["section"]),
    ]


def _kv_table(rows: list[tuple[str, str]], col_widths: tuple[float, float] = (45 * mm, 50 * mm)) -> Table:
    data = []
    for label, value in rows:
        data.append([
            label,
            value if value not in (None, "") else "—",
        ])
    table = Table(data, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 8),
        ("TEXTCOLOR", (0, 0), (0, -1), BRAND_MUTED),
        ("TEXTCOLOR", (1, 0), (1, -1), BRAND_NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, BRAND_BORDER),
        ("BOX", (0, 0), (-1, -1), 0.4, BRAND_BORDER),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, BRAND_BG_SOFT]),
    ]))
    return table


def _build_metadata_section(styles: dict, payload: dict) -> list:
    duration = _format_duration_ms(payload.get("time", []))
    sampling = payload.get("sampling_rates") or [[0]]
    sr = sampling[0][0] if sampling and sampling[0] else "—"
    freq = payload.get("frequency", "—")

    rows = [
        ("Station", payload.get("station_name") or "—"),
        ("Device ID", payload.get("rec_dev_id") or "—"),
        ("Durasi rekaman", duration),
        ("Total sample", f"{payload.get('total_samples', 0):,}".replace(",", ".")),
        ("Sampling rate", f"{sr} Hz"),
        ("Frekuensi nominal", f"{freq} Hz"),
        ("Kanal analog", f"{len(payload.get('analog_channels', []))} kanal"),
        ("Kanal digital", f"{len(payload.get('status_channels', []))} kanal"),
    ]
    return _section_header(styles, "SECTION 1", "Metadata Rekaman") + [_kv_table(rows)]


def _build_electrical_section(styles: dict, elec: dict) -> list:
    rows = [
        ("Waktu inception", f"{elec['inception_time_ms']:.1f} ms" if elec.get("inception_time_ms") is not None else "—"),
        ("Durasi gangguan", f"{elec['fault_duration_ms']:.1f} ms" if elec.get("fault_duration_ms") is not None else "—"),
        ("Trip time", f"{elec['trip_time_ms']:.1f} ms ({elec.get('trip_time_source', '—')})" if elec.get("trip_time_ms") is not None else "—"),
        ("A/R dead time", f"{elec['ar_dead_time_ms']:.1f} ms" if elec.get("ar_dead_time_ms") is not None else "—"),
        ("I peak fasa A", _format_number(elec.get("i_peak_ia_a"), " A")),
        ("I peak fasa B", _format_number(elec.get("i_peak_ib_a"), " A")),
        ("I peak fasa C", _format_number(elec.get("i_peak_ic_a"), " A")),
        ("V sag", _format_number(elec.get("v_sag_pct"), " %", 1)),
        ("I positive sequence", _format_number(elec.get("i_pos_seq_a"), " A")),
        ("I negative sequence", _format_number(elec.get("i_neg_seq_a"), " A")),
        ("I zero sequence", _format_number(elec.get("i_zero_seq_a"), " A")),
        ("|Z| at inception", _format_number(elec.get("z_at_inception_ohm"), " Ω")),
        ("|Z| minimum", _format_number(elec.get("z_min_ohm"), " Ω")),
        ("R at fault", _format_number(elec.get("r_at_fault_ohm"), " Ω")),
        ("X at fault", _format_number(elec.get("x_at_fault_ohm"), " Ω")),
        ("Z angle", _format_number(elec.get("z_angle_deg"), " °", 1)),
    ]

    # Two-column layout for compactness
    half = (len(rows) + 1) // 2
    left_rows = rows[:half]
    right_rows = rows[half:]
    while len(right_rows) < len(left_rows):
        right_rows.append(("", ""))

    pair_data = []
    col_w = 50 * mm
    val_w = 35 * mm
    for (ll, lv), (rl, rv) in zip(left_rows, right_rows):
        pair_data.append([ll, lv, rl, rv])

    pair_table = Table(
        pair_data,
        colWidths=[col_w, val_w, col_w, val_w],
        hAlign="LEFT",
    )
    pair_table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 8),
        ("FONT", (2, 0), (2, -1), "Helvetica-Bold", 8),
        ("TEXTCOLOR", (0, 0), (0, -1), BRAND_MUTED),
        ("TEXTCOLOR", (2, 0), (2, -1), BRAND_MUTED),
        ("TEXTCOLOR", (1, 0), (1, -1), BRAND_NAVY),
        ("TEXTCOLOR", (3, 0), (3, -1), BRAND_NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, BRAND_BORDER),
        ("LINEAFTER", (1, 0), (1, -1), 0.4, BRAND_BORDER),
        ("BOX", (0, 0), (-1, -1), 0.4, BRAND_BORDER),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, BRAND_BG_SOFT]),
    ]))
    return _section_header(styles, "SECTION 2", "Parameter Elektrikal") + [pair_table]


def _build_ai_analysis_section(styles: dict, ai_analysis: Optional[dict]) -> list:
    """Render the AI fault-cause analysis as its own PDF section.

    Includes top-N cause ranking (with confidence bars) and the structured
    evidence list (color-coded by severity). Returns [] when no AI payload.
    """
    if not ai_analysis:
        return []

    cause_ranking = ai_analysis.get("cause_ranking") or []
    evidence = ai_analysis.get("evidence") or []
    fault_type = ai_analysis.get("fault_type")
    overall_conf = ai_analysis.get("overall_confidence")

    if not cause_ranking and not evidence and not fault_type:
        return []

    flowables: list = []
    flowables.extend(_section_header(styles, "SECTION 3", "Analisis AI — Fault Cause"))

    # Header summary line (fault type + overall confidence)
    summary_bits = []
    if isinstance(fault_type, str) and fault_type.strip():
        ft_label = {
            "transient": "Transient (sementara)",
            "permanent": "Permanent (menetap)",
        }.get(fault_type.lower(), fault_type)
        summary_bits.append(f"<b>Jenis kejadian:</b> {ft_label}")
    if isinstance(overall_conf, (int, float)):
        pct = overall_conf * 100 if overall_conf <= 1 else overall_conf
        summary_bits.append(f"<b>Confidence overall:</b> {pct:.0f}%")
    if summary_bits:
        flowables.append(Paragraph(" &nbsp;·&nbsp; ".join(summary_bits), styles["body"]))
        flowables.append(Spacer(1, 4))

    # Cause ranking table
    if isinstance(cause_ranking, list) and cause_ranking:
        top_n = cause_ranking[:5]
        header = [
            Paragraph("<b>#</b>", styles["body_muted"]),
            Paragraph("<b>Cause</b>", styles["body_muted"]),
            Paragraph("<b>Label</b>", styles["body_muted"]),
            Paragraph("<b>Confidence</b>", styles["body_muted"]),
        ]
        data = [header]
        for idx, item in enumerate(top_n, start=1):
            if not isinstance(item, dict):
                continue
            cause = str(item.get("cause") or "—")
            label = str(item.get("label") or cause)
            conf = item.get("confidence")
            if isinstance(conf, (int, float)):
                pct = conf * 100 if conf <= 1 else conf
                conf_str = f"{pct:.1f}%"
            else:
                conf_str = "—"
            data.append([
                Paragraph(str(idx), styles["body"]),
                Paragraph(cause, styles["body"]),
                Paragraph(label, styles["body"]),
                Paragraph(f"<b>{conf_str}</b>", styles["body"]),
            ])

        rank_table = Table(
            data,
            colWidths=[10 * mm, 35 * mm, (PAGE_W - 2 * MARGIN) - 10 * mm - 35 * mm - 28 * mm, 28 * mm],
            hAlign="LEFT",
        )
        rank_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_BG_SOFT),
            ("LINEBELOW", (0, 0), (-1, -2), 0.25, BRAND_BORDER),
            ("BOX", (0, 0), (-1, -1), 0.4, BRAND_BORDER),
            # Highlight the top row
            ("BACKGROUND", (0, 1), (-1, 1), BRAND_BG_HIGHLIGHT),
            ("FONT", (0, 1), (-1, 1), "Helvetica-Bold", 9),
        ]))
        flowables.append(rank_table)
        flowables.append(Spacer(1, 6))

    # Evidence list — structured items with severity coloring
    if isinstance(evidence, list) and evidence:
        flowables.append(Paragraph("<b>Bukti pendukung (evidence)</b>", styles["body"]))
        flowables.append(Spacer(1, 2))

        rows = []
        for item in evidence:
            if isinstance(item, dict):
                text = (item.get("text") or "").strip()
                severity = (item.get("severity") or "info").lower()
                kind = item.get("kind")
            elif isinstance(item, str):
                text = item.strip()
                severity = "info"
                kind = None
            else:
                continue
            if not text:
                continue

            sev_hex = SEVERITY_HEX.get(severity, SEVERITY_HEX["info"])
            sev_bg = SEVERITY_BG.get(severity, SEVERITY_BG["info"])
            sev_chip = Paragraph(
                f"<font color='{sev_hex}'><b>{severity.upper()}</b></font>",
                styles["body_muted"],
            )
            body_text = text
            if kind:
                body_text += f" <font color='#64748b'>· {kind}</font>"
            rows.append((sev_chip, Paragraph(body_text, styles["body"]), sev_bg))

        if rows:
            ev_data = [[chip, body] for chip, body, _ in rows]
            ev_table = Table(
                ev_data,
                colWidths=[20 * mm, (PAGE_W - 2 * MARGIN) - 20 * mm],
                hAlign="LEFT",
            )
            style_cmds = [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("BOX", (0, 0), (-1, -1), 0.4, BRAND_BORDER),
                ("LINEBELOW", (0, 0), (-1, -2), 0.25, BRAND_BORDER),
            ]
            for i, (_, _, bg) in enumerate(rows):
                style_cmds.append(("BACKGROUND", (0, i), (-1, i), bg))
            ev_table.setStyle(TableStyle(style_cmds))
            flowables.append(ev_table)

    return flowables


CATEGORY_HEX = {
    "trip":    "#dc2626",
    "zone":    "#7c3aed",
    "reclose": "#0891b2",
    "breaker": "#ea580c",
    "comms":   "#16a34a",
    "other":   "#64748b",
}
CATEGORY_LABEL = {
    "trip":    "Trip",
    "zone":    "Zona",
    "reclose": "Auto-Reclose",
    "breaker": "PMT / CB",
    "comms":   "Teleproteksi",
    "other":   "Lain",
}


def _build_soe_section(styles: dict, events: list) -> list:
    """Render Sequence of Events as a tabular section.

    Replaces the digital-status strip plot — table form is more readable in a
    static PDF and preserves precise timestamps.
    """
    if not events:
        return []

    flowables: list = []
    flowables.extend(_section_header(styles, "SECTION 4", "Sequence of Events (SOE)"))

    header = [
        Paragraph("<b>Time (ms)</b>", styles["body_muted"]),
        Paragraph("<b>Δt (ms)</b>", styles["body_muted"]),
        Paragraph("<b>Kanal</b>", styles["body_muted"]),
        Paragraph("<b>State</b>", styles["body_muted"]),
        Paragraph("<b>Kategori</b>", styles["body_muted"]),
        Paragraph("<b>Keterangan</b>", styles["body_muted"]),
    ]
    data = [header]

    for ev in events:
        if hasattr(ev, "model_dump"):
            ev = ev.model_dump()
        elif hasattr(ev, "dict"):
            ev = ev.dict()
        if not isinstance(ev, dict):
            continue

        time_ms = ev.get("time_ms")
        rel_ms = ev.get("rel_ms")
        channel = str(ev.get("channel") or "—")
        state = ev.get("state")
        category = (ev.get("category") or "other").lower()
        label = ev.get("label") or ""

        time_str = f"{time_ms:.2f}" if isinstance(time_ms, (int, float)) else "—"
        rel_str = f"{rel_ms:+.2f}" if isinstance(rel_ms, (int, float)) else "—"
        state_str = "ON (1)" if state == 1 else ("OFF (0)" if state == 0 else "—")
        cat_hex = CATEGORY_HEX.get(category, CATEGORY_HEX["other"])
        cat_label = CATEGORY_LABEL.get(category, category)

        data.append([
            Paragraph(time_str, styles["body"]),
            Paragraph(rel_str, styles["body_muted"]),
            Paragraph(channel, styles["body"]),
            Paragraph(state_str, styles["body"]),
            Paragraph(f"<font color='{cat_hex}'><b>{cat_label}</b></font>", styles["body"]),
            Paragraph(label or "", styles["body"]),
        ])

    usable = PAGE_W - 2 * MARGIN
    soe_table = Table(
        data,
        colWidths=[
            18 * mm,  # time
            18 * mm,  # delta
            32 * mm,  # channel
            18 * mm,  # state
            24 * mm,  # category
            usable - (18 + 18 + 32 + 18 + 24) * mm,  # label fills the rest
        ],
        hAlign="LEFT",
        repeatRows=1,
    )
    soe_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_BG_SOFT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BRAND_BG_SOFT]),
        ("LINEBELOW", (0, 0), (-1, -2), 0.2, BRAND_BORDER),
        ("BOX", (0, 0), (-1, -1), 0.4, BRAND_BORDER),
    ]))
    flowables.append(soe_table)
    return flowables


def _build_chart_block(styles: dict, kicker: str, chart: ChartImage, max_height_mm: float = 90) -> list:
    """Decode base64 PNG and add as a sized Image flowable."""
    try:
        raw = base64.b64decode(chart.image_b64)
    except Exception:
        return [
            Paragraph(kicker, styles["section_kicker"]),
            Paragraph(chart.title, styles["section"]),
            Paragraph(
                f"<i>Chart tidak dapat dimuat ({chart.id}).</i>",
                styles["body_muted"],
            ),
        ]

    buf = io.BytesIO(raw)
    img = Image(buf)
    # Scale to fit page width while respecting max height
    max_w = PAGE_W - 2 * MARGIN
    max_h = max_height_mm * mm
    aspect = img.imageHeight / img.imageWidth if img.imageWidth else 1
    target_w = max_w
    target_h = target_w * aspect
    if target_h > max_h:
        target_h = max_h
        target_w = target_h / aspect if aspect > 0 else max_w
    img.drawWidth = target_w
    img.drawHeight = target_h
    img.hAlign = "LEFT"
    return [
        Paragraph(kicker, styles["section_kicker"]),
        Paragraph(chart.title, styles["section"]),
        img,
    ]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def _build_pdf(payload: dict, request: ReportRequest, analysis_id: str) -> bytes:
    relay_type = request.relay_type.upper() if request.relay_type else "21"
    relay_label = RELAY_LABELS.get(relay_type, relay_type)
    timestamp = _format_datetime()
    station = payload.get("station_name") or "Stasiun Tidak Diketahui"
    device = payload.get("rec_dev_id") or "Device Tidak Diketahui"

    fault_class = _compute_fault_classification(payload)
    elec = _compute_electrical_params(payload)

    buf = io.BytesIO()
    header_top_clearance = 22 * mm + 4 * mm  # header band + gap
    footer_clearance = 10 * mm
    frame = Frame(
        MARGIN,
        MARGIN + footer_clearance,
        PAGE_W - 2 * MARGIN,
        PAGE_H - 2 * MARGIN - header_top_clearance - footer_clearance,
        showBoundary=0,
    )
    hf = _HeaderFooter(station, device, analysis_id, relay_label, timestamp)
    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title=f"Laporan Analisis Gangguan — {station}",
        author="PLN UIT JBT",
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=hf.on_page)])

    styles = _build_styles()

    story: list = []
    story.append(_build_conclusion(styles, fault_class, request.ai_analysis, elec))
    story.append(Spacer(1, 10))

    story.extend(_build_metadata_section(styles, payload))
    story.append(Spacer(1, 8))

    if relay_type == "21":
        story.extend(_build_electrical_section(styles, elec))
        story.append(Spacer(1, 8))

    ai_section = _build_ai_analysis_section(styles, request.ai_analysis)
    if ai_section:
        story.extend(ai_section)
        story.append(Spacer(1, 8))

    soe_section = _build_soe_section(styles, request.soe_events)
    if soe_section:
        story.extend(soe_section)
        story.append(Spacer(1, 8))

    chart_titles = {
        "impedance_locus":        ("VISUALISASI", "Impedance Locus (R-X Trajectory)"),
        "impedance_locus_ground": ("VISUALISASI", "Impedance Locus — Phase-to-Ground"),
        "impedance_locus_phase":  ("VISUALISASI", "Impedance Locus — Phase-to-Phase"),
        "waveform_strip":         ("VISUALISASI", "Strip Waveform & Locus Events"),
        "waveform_voltage":       ("VISUALISASI", "Waveform Tegangan"),
        "waveform_current":       ("VISUALISASI", "Waveform Arus"),
        "diff_restraint":         ("VISUALISASI", "Diff / Restraint Plot"),
        "overcurrent_overlay":    ("VISUALISASI", "Overcurrent Overlay"),
    }

    for chart in request.charts:
        kicker, default_title = chart_titles.get(chart.id, ("VISUALISASI", chart.title))
        title = chart.title or default_title
        story.extend(_build_chart_block(styles, kicker, ChartImage(id=chart.id, title=title, image_b64=chart.image_b64)))
        story.append(Spacer(1, 8))

    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/{analysis_id}")
async def generate_report(analysis_id: str, body: ReportRequest):
    """Generate a structured PDF report for the analysis session."""
    payload = load_analysis(analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")

    loop = asyncio.get_event_loop()
    try:
        pdf_bytes = await loop.run_in_executor(None, _build_pdf, payload, body, analysis_id)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {exc}") from exc

    station_slug = (payload.get("station_name") or "report").replace(" ", "_").replace("/", "-")
    filename = f"laporan_gangguan_{station_slug}_{analysis_id[:8]}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
