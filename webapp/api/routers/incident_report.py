"""Incident report: one incident's reconstruction on the template every
report shares (report.py).

The page builds the incident's story (summary, sequence, cause, what to
check, record roles) in incidentStory.ts, and the report prints that same
story, so the PDF says what the page says. The backend adds what it holds
itself: each fault's reasoning ledger (episode.interpretation.reasoning) and
each fault record's signal sequence (its snapshot's reasoning.signals).

Page 1: summary, sequence of events, cause, what to check. Then the
reasoning per fault. Attachments: the records with their roles, and each
fault record's signal sequence.
"""

from __future__ import annotations

import asyncio
import io
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import CondPageBreak, KeepTogether, Paragraph, Spacer, Table, TableStyle

from ..incidents import service as incident_service
from ..incidents.models import FaultEpisode, Incident, IncidentRecord
from ..incidents.service import IncidentServiceError
from .incidents import _require_multi_comtrade_enabled
from .location_report import _MONTHS, FRAME_W, RULE, Method, _document, _pdf_response, _rule_table, _section, _styles
from .report import BRAND_MUTED, PAGE_H, _format_datetime, _HeaderFooter, _safe_text

router = APIRouter(prefix="/api/incidents", tags=["incident-report"])

# The report's section kickers and rules in the brand blue of every report.
INCIDENT = Method(
    title="Rekonstruksi insiden",
    subtitle="Rekonstruksi insiden",
    basis="",
    accent_hex="#2563eb",
    tint_hex="#eff6ff",
    footer="TFA · insiden",
)

# The incident page's tones (index.css, light theme): ink, soft fill, line.
TONE = {
    "fault": ("#9a3412", colors.HexColor("#fff8f3"), colors.HexColor("#f3c9ae")),
    "reclose": ("#1e40af", colors.HexColor("#eff4ff"), colors.HexColor("#bfd0f5")),
    "neutral": ("#26313d", colors.HexColor("#eef1f5"), colors.HexColor("#dde2e8")),
    "warning": ("#92400e", colors.HexColor("#fffbeb"), colors.HexColor("#fde68a")),
}
CARD_TONE = {"fault": "fault", "reclose": "reclose", "after": "neutral"}
CONFIDENCE = {"high": "Tinggi", "medium": "Sedang", "low": "Rendah", "ai": "Bacaan AI", "flag": "Perlu dicek"}


# ---------------------------------------------------------------------------
# The story, as incidentStory.ts builds it
# ---------------------------------------------------------------------------

class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore")


class StoryChip(_Loose):
    label: str
    tone: str = "neutral"


class StoryTile(_Loose):
    label: str
    value: str
    detail: Optional[str] = None


class StoryCard(_Loose):
    kind: str
    title: str
    time: Optional[str] = None
    headline: str = ""
    bullets: list[str] = []
    recordName: Optional[str] = None
    emphasis: bool = False


class StoryConnector(_Loose):
    kind: str
    label: str
    detail: str = ""


class StoryEntry(_Loose):
    type: str
    card: Optional[StoryCard] = None
    connector: Optional[StoryConnector] = None


class StoryPattern(_Loose):
    title: str
    strength: str = ""
    text: str = ""
    tone: str = "neutral"


class StoryCandidate(_Loose):
    cause: str
    percent: float


class StoryAi(_Loose):
    title: str
    recordName: str = ""
    kind: str
    cause: Optional[str] = None
    percent: Optional[float] = None
    note: Optional[str] = None
    candidates: list[StoryCandidate] = []


class StoryCause(_Loose):
    status: StoryChip
    headline: str
    pattern: Optional[StoryPattern] = None
    ai: list[StoryAi] = []
    footnote: str = ""


class StoryCheck(_Loose):
    id: str
    title: str
    detail: str = ""


class StoryRecord(_Loose):
    recordId: str
    name: str
    roleLabel: str
    roleSuffix: str = ""
    start: str = ""
    line: str = ""
    note: str = ""


class StoryFault(_Loose):
    episodeId: str
    number: int
    time: Optional[str] = None


class IncidentStoryIn(_Loose):
    chips: list[StoryChip] = []
    headline: str
    narrative: str = ""
    tiles: list[StoryTile] = []
    sequenceMeta: str = ""
    sequence: list[StoryEntry] = []
    cause: Optional[StoryCause] = None
    checklist: list[StoryCheck] = []
    records: list[StoryRecord] = []
    faults: list[StoryFault] = []
    clockLabel: str = "jam DFR"


class IncidentReportRequest(BaseModel):
    story: IncidentStoryIn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _extra_styles(styles: dict) -> dict:
    body = styles["body"]
    styles["headline"] = ParagraphStyle("ir_headline", parent=body, fontName="Helvetica-Bold", fontSize=15, leading=19)
    styles["narrative"] = ParagraphStyle("ir_narrative", parent=body, fontSize=10, leading=14)
    styles["tile_label"] = ParagraphStyle("ir_tile_label", parent=styles["label"], fontSize=7)
    styles["tile_value"] = ParagraphStyle("ir_tile_value", parent=body, fontName="Helvetica-Bold", fontSize=11, leading=14)
    styles["card_title"] = ParagraphStyle("ir_card_title", parent=body, fontName="Helvetica-Bold", fontSize=8, leading=10)
    styles["card_time"] = ParagraphStyle("ir_card_time", parent=body, fontName="Courier", fontSize=8.5, leading=10.5)
    styles["card_headline"] = ParagraphStyle("ir_card_headline", parent=body, fontName="Helvetica-Bold", fontSize=9.5, leading=12)
    styles["bullet"] = ParagraphStyle("ir_bullet", parent=body, fontSize=8.5, leading=11, leftIndent=9, bulletIndent=0)
    styles["connector"] = ParagraphStyle("ir_connector", parent=styles["note"], alignment=1)
    styles["cause_headline"] = ParagraphStyle("ir_cause_headline", parent=body, fontName="Helvetica-Bold", fontSize=12, leading=15)
    styles["fault_head"] = ParagraphStyle("ir_fault_head", parent=body, fontName="Helvetica-Bold", fontSize=10.5, leading=13)
    styles["conflict"] = ParagraphStyle("ir_conflict", parent=styles["td"], fontSize=8.5, leading=11, textColor=colors.HexColor("#991b1b"))
    styles["evidence"] = ParagraphStyle("ir_evidence", parent=styles["td"], fontSize=8.5, leading=11, textColor=colors.HexColor("#26313d"))
    styles["mono_td"] = ParagraphStyle("ir_mono_td", parent=styles["td"], fontName="Courier", fontSize=8.5, leading=11)
    return styles


def _chips(kicker: str, chips: list[StoryChip]) -> Table:
    """The kicker, then the status chips the page shows next to it. Plain
    strings, so each cell sizes to its text."""
    cells: list[Any] = [kicker]
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONT", (0, 0), (0, 0), "Helvetica-Bold", 7.5),
        ("TEXTCOLOR", (0, 0), (0, 0), BRAND_MUTED),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    widths: list[Optional[float]] = [None]
    for i, chip in enumerate(chips, start=1):
        ink, soft, line = TONE.get(chip.tone, TONE["neutral"])
        cells.append(chip.label)
        widths.append(None)
        commands += [
            ("FONT", (i, 0), (i, 0), "Helvetica-Bold", 7.5),
            ("TEXTCOLOR", (i, 0), (i, 0), colors.HexColor(ink)),
            ("BACKGROUND", (i, 0), (i, 0), soft),
            ("BOX", (i, 0), (i, 0), 0.6, line),
            ("LEFTPADDING", (i, 0), (i, 0), 6),
            ("RIGHTPADDING", (i, 0), (i, 0), 6),
        ]
    table = Table([cells], colWidths=widths, hAlign="LEFT")
    table.setStyle(TableStyle(commands))
    return table


def _tiles(styles: dict, tiles: list[StoryTile]) -> Optional[Table]:
    if not tiles:
        return None
    width = FRAME_W / len(tiles)
    cells = []
    for tile in tiles:
        value = _safe_text(tile.value) + (f" <font size=8 color='#4a5563'>({_safe_text(tile.detail)})</font>" if tile.detail else "")
        cells.append([Paragraph(_safe_text(tile.label), styles["tile_label"]), Paragraph(value, styles["tile_value"])])
    table = Table([cells], colWidths=[width] * len(tiles))
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LINEAFTER", (0, 0), (-2, -1), 0.5, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _card(styles: dict, card: StoryCard) -> Table:
    """One event of the sequence: a fault, a reclose or what came after."""
    ink, soft, line = TONE[CARD_TONE.get(card.kind, "neutral")]
    head = f"<font color='{ink}'>{_safe_text(card.title.upper())}</font>"
    if card.time:
        head += f"    <font name='Courier' size=8.5>{_safe_text(card.time)}</font>"
    content: list[Any] = [Paragraph(head, styles["card_title"]), Spacer(1, 2)]
    if card.headline:
        content.append(Paragraph(_safe_text(card.headline), styles["card_headline"]))
    if card.kind == "after" and card.bullets:
        content.append(Paragraph(_safe_text(" ".join(card.bullets)), styles["note"]))
    else:
        content += [Paragraph(_safe_text(text), styles["bullet"], bulletText="•") for text in card.bullets]
    if card.recordName:
        content.append(Paragraph(f"Rekaman {_safe_text(card.recordName)}", styles["label"]))
    table = Table([[content]], colWidths=[FRAME_W])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), soft),
        ("BOX", (0, 0), (-1, -1), 1.4 if card.emphasis else 0.6, colors.HexColor("#c2410c") if card.emphasis else line),
        ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(ink)),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def _connector(styles: dict, connector: StoryConnector) -> Paragraph:
    """What lies between two events: the dead time, a re-fault, a gap."""
    ink = TONE["fault"][0] if connector.kind == "refault" else "#4a5563"
    detail = f"  ·  {_safe_text(connector.detail)}" if connector.detail else ""
    return Paragraph(f"↓  <font color='{ink}'><b>{_safe_text(connector.label)}</b></font>{detail}", styles["connector"])


def _checkbox() -> Drawing:
    box = Drawing(11, 11)
    box.add(Rect(1, 0.5, 9, 9, strokeColor=colors.HexColor("#4a5563"), strokeWidth=0.8, fillColor=None))
    return box


def _cause(styles: dict, cause: StoryCause) -> list:
    flowables: list[Any] = [_chips("PENYEBAB", [cause.status]), Spacer(1, 4)]
    flowables.append(Paragraph(_safe_text(cause.headline), styles["cause_headline"]))
    flowables.append(Spacer(1, 6))
    if cause.pattern:
        ink, soft, line = TONE.get(cause.pattern.tone, TONE["neutral"])
        box = Table([[[
            Paragraph(
                f"<b>{_safe_text(cause.pattern.title)}</b>"
                + (f"    <font color='{ink}' size=8><b>{_safe_text(cause.pattern.strength)}</b></font>"
                   if cause.pattern.strength else ""),
                styles["value"],
            ),
            Paragraph(_safe_text(cause.pattern.text), styles["note"]),
        ]]], colWidths=[FRAME_W])
        box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), soft),
            ("BOX", (0, 0), (-1, -1), 0.6, line),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        flowables += [box, Spacer(1, 8)]
    if cause.ai:
        rows = [[
            Paragraph("Bacaan AI per rekaman gangguan", styles["th"]),
            Paragraph("Dibaca terpisah, tidak dirata-rata", styles["th_right"]),
        ]]
        for reading in cause.ai:
            who = (f"{_safe_text(reading.title)}<br/><font name='Courier' size=8 color='#4a5563'>"
                   f"{_safe_text(reading.recordName)}</font>")
            if reading.kind == "reading":
                text = f"<b>{_safe_text(reading.cause or '—')}</b>  {_safe_text(_percent(reading.percent))}"
                if reading.note:
                    text += f"<br/><font color='#4a5563'>{_safe_text(reading.note)}</font>"
            elif reading.kind == "no_dominant":
                text = ("<b>Tidak ada yang dominan</b>  "
                        + _safe_text(" / ".join(_percent(c.percent) for c in reading.candidates))
                        + f"<br/><font color='#4a5563'>{_safe_text(' · '.join(c.cause for c in reading.candidates))}</font>")
            else:
                text = f"<font color='#4a5563'>{_safe_text(reading.note or '—')}</font>"
            rows.append([Paragraph(who, styles["td"]), Paragraph(text, styles["td_right"])])
        flowables.append(_rule_table(rows, [FRAME_W * 0.45, FRAME_W * 0.55]))
        flowables.append(Spacer(1, 4))
    if cause.footnote:
        flowables.append(Paragraph(_safe_text(cause.footnote), styles["note"]))
    return flowables


def _percent(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.0f}%"


def _checklist(styles: dict, items: list[StoryCheck]) -> list:
    if not items:
        return []
    rows = [
        [_checkbox(), [Paragraph(f"<b>{_safe_text(item.title)}</b>", styles["value"]),
                       Paragraph(_safe_text(item.detail), styles["note"])]]
        for item in items
    ]
    table = Table(rows, colWidths=[16, FRAME_W - 16])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (0, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE),
    ]))
    return [*_section(styles, "PENGECEKAN", "Yang perlu dicek"), Spacer(1, 2), table]


def _ledger(styles: dict, story: IncidentStoryIn, fault: StoryFault, episode: Optional[FaultEpisode]) -> list:
    """One fault's conclusions, each with its evidence and rules."""
    reasoning = (episode.interpretation or {}).get("reasoning") if episode else None
    when = f" · {fault.time} ({story.clockLabel})" if fault.time else ""
    head: list[Any] = [Paragraph(f"Gangguan #{fault.number}{_safe_text(when)}", styles["fault_head"])]
    if not reasoning:
        return [*head, Spacer(1, 3), Paragraph(
            "Penalaran belum tersedia untuk gangguan ini: rekamannya dianalisa sebelum rantai aturan ada. Muat "
            "ulang analisa rekaman di Detail teknis, lalu cetak ulang laporan ini.", styles["note"])]
    rows_in = reasoning.get("rows") or []
    conclusions = sum(1 for row in rows_in if row.get("confidence") != "flag")
    head.append(Paragraph(
        f"{conclusions} kesimpulan · {int(reasoning.get('flag_count') or 0)} ditandai · "
        f"{int(reasoning.get('conflict_count') or 0)} konflik", styles["label"]))
    rows = [[Paragraph(text, styles["th"]) for text in ("Langkah", "Kesimpulan dan bukti", "Aturan", "Keyakinan")]]
    flagged = []
    for i, row in enumerate(rows_in, start=1):
        cell: list[Any] = [Paragraph(f"<b>{_safe_text(row.get('title', ''))}</b>", styles["td"])]
        cell += [Paragraph(_safe_text(line), styles["evidence"]) for line in row.get("evidence") or []]
        cell += [Paragraph(_safe_text(line), styles["conflict"]) for line in row.get("conflicts") or []]
        confidence = str(row.get("confidence") or "")
        rows.append([
            Paragraph(_safe_text(row.get("label", "")), styles["td"]),
            cell,
            Paragraph(_safe_text(", ".join(row.get("rules") or [])), styles["mono_td"]),
            Paragraph(_safe_text(CONFIDENCE.get(confidence, confidence)), styles["td"]),
        ])
        if confidence == "flag":
            flagged.append(i)
    table = _rule_table(rows, [FRAME_W * 0.15, FRAME_W * 0.57, FRAME_W * 0.13, FRAME_W * 0.15])
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]
                              + [("BACKGROUND", (0, i), (-1, i), TONE["warning"][1]) for i in flagged]))
    # The heading stays with its table: a fault's ledger starts on the page
    # that can hold it.
    return [KeepTogether([*head, Spacer(1, 4), table])]


def _signals(
    styles: dict, name: str, role: str, record: IncidentRecord, same_silent_as: Optional[tuple[str, list]] = None,
) -> tuple[list, list]:
    """The status channels that became active in a fault record. The waveform
    events are in the ledger already (fault start, current ceased), and the
    resets and contact bounce add length, not meaning. Returns the block and
    the record's silent protection channels, so the next record can say
    "the same" instead of listing them again."""
    signals = ((record.canonical_snapshot or {}).get("reasoning") or {}).get("signals") or {}
    events = [
        ev for ev in signals.get("events") or []
        if not ev.get("muted") and ev.get("role") != "Gelombang"
        and not str(ev.get("change", "")).lower().startswith("reset")
    ]
    # A protection channel that was recorded but never moved is evidence too
    # (rule P4), e.g. a Send that stayed silent.
    silent = [f"{item.get('channel')} ({item.get('role')})" for item in signals.get("silent") or [] if item.get("role")]
    if not events:
        return [], silent
    reference = ("Waktu dihitung dari awal gangguan." if signals.get("reference") == "fault_start"
                 else "Waktu dihitung dari awal rekaman.")
    rows = [[Paragraph(text, styles["th"]) for text in ("Waktu (ms)", "Kanal", "Peran", "Perubahan")]]
    for ev in events:
        rows.append([
            Paragraph(_signed_ms(ev.get("t_ms")), styles["mono_td"]),
            Paragraph(_safe_text(ev.get("channel", "")), styles["td"]),
            Paragraph(_safe_text(ev.get("role", "")), styles["td"]),
            Paragraph(_safe_text(ev.get("change", "")), styles["td"]),
        ])
    table = _rule_table(rows, [FRAME_W * 0.14, FRAME_W * 0.38, FRAME_W * 0.16, FRAME_W * 0.32])
    table.setStyle(TableStyle([("TOPPADDING", (0, 1), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 1), (-1, -1), 2.5)]))
    block: list[Any] = [
        Paragraph(f"<b>{_safe_text(name)}</b>  <font color='#4a5563'>{_safe_text(role)}</font>", styles["value"]),
        Paragraph(_safe_text(reference), styles["label"]),
        Spacer(1, 3),
        table,
    ]
    if silent:
        if same_silent_as and same_silent_as[1] == silent:
            text = f"Kanal proteksi yang terekam tetapi tidak pernah aktif: sama seperti {same_silent_as[0]}."
        else:
            text = f"Kanal proteksi yang terekam tetapi tidak pernah aktif: {', '.join(silent)}."
        block += [Spacer(1, 3), Paragraph(_safe_text(text), styles["note"])]
    return [KeepTogether(block)], silent


def _wib(iso: Optional[str]) -> Optional[str]:
    """A stored UTC timestamp as WIB, "10 Okt 2026, 22:40 WIB"."""
    if not iso:
        return None
    try:
        moment = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone(timedelta(hours=7)))
    return f"{moment.day} {_MONTHS[moment.month - 1]} {moment.year}, {moment:%H:%M} WIB"


def _signed_ms(value) -> str:
    if value is None:
        return "—"
    text = f"{abs(float(value)):.1f}".replace(".", ",")
    return text if float(value) == 0 else ("−" if float(value) < 0 else "+") + text


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def build_incident_pdf(
    incident: Incident,
    records: list[IncidentRecord],
    episodes: list[FaultEpisode],
    story: IncidentStoryIn,
    built_at: Optional[str] = None,
) -> bytes:
    """``built_at``: when the reconstruction the ledgers come from was built,
    so a reader can tell an old analysis from a current one."""
    styles = _extra_styles(_styles(INCIDENT))
    short_id = incident.incident_id[:8]
    included = [r for r in records if r.inclusion_status != "EXCLUDED"]
    place = " · ".join(part for part in (
        incident.station_name, f"bay {incident.bay_name}" if incident.bay_name else None,
        f"{incident.voltage_level_kv:g} kV" if incident.voltage_level_kv else None,
    ) if part)

    buf = io.BytesIO()
    header = _HeaderFooter(
        incident.title, place or "—", short_id, f"Rekonstruksi insiden · {len(included)} rekaman",
        _format_datetime(), kicker="LAPORAN INSIDEN GANGGUAN", id_label="ID INSIDEN",
        footer_left=f"TFA · insiden {short_id} · {incident.title}",
    )
    doc = _document(buf, INCIDENT, header, f"Laporan insiden — {incident.title}")

    story_flow: list[Any] = [_chips("RINGKASAN", story.chips), Spacer(1, 5)]
    story_flow.append(Paragraph(_safe_text(story.headline), styles["headline"]))
    if story.narrative:
        story_flow += [Spacer(1, 4), Paragraph(_safe_text(story.narrative), styles["narrative"])]
    tiles = _tiles(styles, story.tiles)
    if tiles is not None:
        story_flow += [Spacer(1, 8), tiles]
    story_flow.append(Spacer(1, 12))

    if story.sequence:
        story_flow += _section(styles, "URUTAN", "Urutan kejadian", story.sequenceMeta)
        story_flow.append(Spacer(1, 3))
        for entry in story.sequence:
            if entry.type == "card" and entry.card:
                story_flow += [KeepTogether([_card(styles, entry.card)]), Spacer(1, 3)]
            elif entry.type == "connector" and entry.connector:
                story_flow += [_connector(styles, entry.connector), Spacer(1, 3)]
        story_flow.append(Spacer(1, 9))

    if story.cause:
        story_flow += [KeepTogether(_cause(styles, story.cause)), Spacer(1, 12)]
    story_flow += [KeepTogether(_checklist(styles, story.checklist))]

    # Reasoning per fault, from a fresh page unless most of one is left.
    episodes_by_id = {episode.episode_id: episode for episode in episodes}
    if story.faults:
        story_flow += [CondPageBreak(PAGE_H * 0.5), Spacer(1, 6)]
        story_flow += _section(styles, "PENALARAN", "Penalaran per gangguan")
        built = _wib(built_at)
        story_flow.append(Paragraph(
            "Setiap kesimpulan dengan bukti dan aturannya (katalog aturan F1–F8). Baris berlatar kuning perlu "
            "dicek manusia." + (f" Disusun dari rekonstruksi {built}; muat ulang analisa rekaman di Detail teknis "
                                "untuk membaca dengan aturan terbaru." if built else ""), styles["note"]))
        for fault in story.faults:
            story_flow += [Spacer(1, 8), *_ledger(styles, story, fault, episodes_by_id.get(fault.episodeId))]

    # Attachments.
    story_flow += [CondPageBreak(PAGE_H * 0.35), Spacer(1, 10)]
    story_flow += _section(styles, "LAMPIRAN", f"Rekaman ({len(story.records)})")
    rows = [[Paragraph(text, styles["th"]) for text in ("Record", "Role in sequence", "Start", "Analysed line", "Notes")]]
    for row in story.records:
        rows.append([
            Paragraph(_safe_text(row.name), styles["mono_td"]),
            Paragraph(_safe_text(f"{row.roleLabel}{row.roleSuffix}"), styles["td"]),
            Paragraph(_safe_text(row.start), styles["mono_td"]),
            Paragraph(_safe_text(row.line), styles["td"]),
            Paragraph(_safe_text(row.note or "—"), styles["evidence"]),
        ])
    story_flow.append(_rule_table(rows, [FRAME_W * 0.19, FRAME_W * 0.2, FRAME_W * 0.14, FRAME_W * 0.11, FRAME_W * 0.36]))

    names = {row.recordId: (row.name, f"{row.roleLabel}{row.roleSuffix}") for row in story.records}
    records_by_id = {record.incident_record_id: record for record in records}
    fault_record_ids = []
    for fault in story.faults:
        episode = episodes_by_id.get(fault.episodeId)
        record_id = (((episode.interpretation or {}).get("reasoning") or {}).get("fault_record_id")
                     if episode else None)
        if record_id and record_id not in fault_record_ids:
            fault_record_ids.append(record_id)
    signal_blocks = []
    previous: Optional[tuple[str, list]] = None
    for record_id in fault_record_ids:
        record = records_by_id.get(record_id)
        if record is None:
            continue
        name, role = names.get(record_id, (record.source_filename or record.analysis_id[:8], ""))
        block, silent = _signals(styles, name, role, record, previous)
        if block:
            signal_blocks += [Spacer(1, 8), *block]
            previous = (name, silent)
    if signal_blocks:
        story_flow += [Spacer(1, 12), *_section(styles, "LAMPIRAN", "Urutan sinyal rekaman gangguan")]
        story_flow.append(Paragraph(
            "Kanal status yang aktif di rekaman tiap gangguan, tanpa reset dan pantulan kontak. Urutan lengkap, "
            "termasuk rekaman lain, ada di Detail teknis.", styles["note"]))
        story_flow += signal_blocks

    doc.build(story_flow)
    return buf.getvalue()


@router.post("/{incident_id}/report")
async def incident_report(incident_id: str, body: IncidentReportRequest):
    _require_multi_comtrade_enabled()
    try:
        incident = incident_service.get_incident(incident_id)
        records = incident_service.list_records(incident_id)
        episodes = incident_service.get_episodes(incident_id)
        built_at = incident_service.get_reconstruction(incident_id).created_at
    except IncidentServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    loop = asyncio.get_event_loop()
    pdf = await loop.run_in_executor(None, build_incident_pdf, incident, records, episodes, body.story, built_at)
    return _pdf_response(pdf, f"laporan_insiden_{incident_id[:8]}.pdf")
