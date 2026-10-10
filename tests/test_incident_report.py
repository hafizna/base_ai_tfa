"""Incident report: the page's story, plus each fault's ledger and signal
sequence from the stored reconstruction."""

import io

from pypdf import PdfReader

from webapp.api.incidents.models import FaultEpisode, Incident, IncidentRecord
from webapp.api.routers.incident_report import IncidentStoryIn, build_incident_pdf


def _text(pdf: bytes) -> str:
    """The report's text with line wraps undone."""
    assert pdf.startswith(b"%PDF")
    return " ".join(" ".join(page.extract_text().split()) for page in PdfReader(io.BytesIO(pdf)).pages)


STORY = {
    "chips": [
        {"label": "Reclose did not hold", "tone": "fault"},
        {"label": "Final state: trip — no further reclose", "tone": "neutral"},
    ],
    "headline": "Gangguan berulang setelah reclose",
    "narrative": "Line MJSNG2 terganggu S-T, reclose berhasil, lalu gangguan yang sama kembali 5,7 s kemudian.",
    "tiles": [
        {"label": "First fault", "value": "15:15:03,738", "mono": True},
        {"label": "Dead time", "value": "5,0 s", "mono": True},
        {"label": "Analysed line", "value": "MJSNG2"},
    ],
    "sequenceMeta": "Dari 2 rekaman · waktu menurut jam DFR GI BRINGIN",
    "sequence": [
        {"type": "card", "card": {
            "kind": "fault", "title": "Gangguan #1", "time": "15:15:03,738", "headline": "S-T, padam 77 ms",
            "bullets": ["Trip Z1 tiga fasa"], "recordId": "r1", "recordName": "ZQ6D", "emphasis": False,
        }},
        {"type": "connector", "connector": {"kind": "dead_time", "label": "Dead time 5,0 s", "detail": "CB menutup kembali"}},
        {"type": "card", "card": {
            "kind": "reclose", "title": "Reclose", "time": "15:15:08,724", "headline": "Line bertegangan lagi",
            "bullets": [], "recordId": "r2", "recordName": "ZQ6E", "emphasis": False,
        }},
    ],
    "cause": {
        "status": {"label": "Pattern: persistent contact", "tone": "fault"},
        "headline": "Pola urutan menunjuk kontak fisik, seperti pohon",
        "pattern": {"title": "Gangguan kembali di fasa yang sama", "strength": "Sedang",
                    "text": "Re-fault S-T setelah reclose berhasil.", "tone": "fault"},
        "ai": [{"title": "Gangguan #1", "recordName": "ZQ6D", "kind": "reading", "cause": "Petir", "percent": 92}],
        "footnote": "Bacaan AI tiap rekaman dibaca terpisah.",
    },
    "checklist": [{"id": "patrol", "title": "Patroli di sekitar 15 km dari GI BRINGIN",
                   "detail": "Cari bekas sentuhan pohon."}],
    "records": [
        {"recordId": "r1", "name": "ZQ6D", "roleLabel": "Gangguan #1", "roleTone": "fault", "roleSuffix": "",
         "start": "15:15:03,600", "line": "MJSNG2", "note": ""},
        {"recordId": "r2", "name": "ZQ6E", "roleLabel": "Reclose", "roleTone": "reclose", "roleSuffix": " · #1",
         "start": "15:15:08,500", "line": "MJSNG2", "note": "Mulai saat line mati"},
    ],
    "faults": [{"episodeId": "e1", "number": 1, "time": "15:15:03,738"}],
    "clockLabel": "jam DFR GI BRINGIN",
}

LEDGER = {
    "fault_record_id": "r1",
    "fault_start_ms": 0.0,
    "rows": [
        {"key": "phases", "step": 4, "label": "Fasa", "title": "S-T, tidak ke tanah",
         "evidence": ["I0/I1 0,06; loop S-T 6,3 Ω"], "rules": ["F4.1", "F4.3"], "confidence": "high",
         "value": {}, "conflicts": []},
        {"key": "flag_send_silent", "step": 9, "label": "Ditandai", "title": "Kanal Send di GI ini tidak aktif",
         "evidence": ["GI lawan menerima sinyal pada +50 ms."], "rules": ["F5.4"], "confidence": "flag",
         "value": {}, "conflicts": ["Periksa pemetaan kanal LP SEND."]},
    ],
    "flag_count": 1,
    "conflict_count": 0,
}

SIGNALS = {
    "reference": "fault_start",
    "events": [
        {"t_ms": 0.1, "channel": "Arus fasa S", "role": "Gelombang", "change": "Gangguan mulai"},
        {"t_ms": 20.5, "channel": "TRIP Z1", "role": "Trip", "change": "Aktif: trip"},
        {"t_ms": 21.0, "channel": "TRIP Z1", "role": "Trip", "change": "Pantulan kontak", "muted": True},
        {"t_ms": 77.0, "channel": "TRIP Z1", "role": "Trip", "change": "Reset"},
    ],
    "silent": [{"channel": "LP SEND MJSNG2", "role": "Teleproteksi"}, {"channel": "SPARE", "role": None}],
    "channel_count": 12,
}


def _pdf(ledger: dict | None = LEDGER) -> str:
    incident = Incident(incident_id="83a55c02470149c0", title="Bringin–Mojosongo #2", station_name="GI BRINGIN",
                        bay_name="MJSNG2", voltage_level_kv=150.0)
    records = [
        IncidentRecord(incident_record_id="r1", incident_id=incident.incident_id, analysis_id="a1",
                       canonical_snapshot={"reasoning": {"signals": SIGNALS}}),
        IncidentRecord(incident_record_id="r2", incident_id=incident.incident_id, analysis_id="a2"),
    ]
    episode = FaultEpisode(episode_id="e1", incident_id=incident.incident_id,
                           interpretation={"reasoning": ledger} if ledger else {})
    return _text(build_incident_pdf(incident, records, [episode], IncidentStoryIn(**STORY),
                                    built_at="2026-10-10T15:40:12.345+00:00"))


def test_the_report_prints_the_page_story_in_its_order():
    text = _pdf()
    assert "LAPORAN INSIDEN GANGGUAN" in text and "ID INSIDEN" in text
    assert "GI BRINGIN · bay MJSNG2 · 150 kV" in text
    order = [
        "Reclose did not hold", "Gangguan berulang setelah reclose", "Urutan kejadian", "CB menutup kembali",
        "Pola urutan menunjuk kontak fisik", "Petir 92%", "Yang perlu dicek", "Patroli di sekitar 15 km",
    ]
    positions = [text.index(part) for part in order]
    assert positions == sorted(positions)


def test_each_fault_carries_its_ledger_with_rules_and_flags():
    text = _pdf()
    assert "Penalaran per gangguan" in text
    assert "Gangguan #1 · 15:15:03,738 (jam DFR GI BRINGIN)" in text
    assert "1 kesimpulan · 1 ditandai · 0 konflik" in text
    assert "S-T, tidak ke tanah" in text and "F4.1, F4.3" in text
    assert "Perlu dicek" in text and "Periksa pemetaan kanal LP SEND." in text
    # The ledger's age, in WIB, so an old analysis is not mistaken for a current one.
    assert "Disusun dari rekonstruksi 10 Okt 2026, 22:40 WIB" in text


def test_the_attachments_list_records_and_the_channels_that_became_active():
    text = _pdf()
    assert "Rekaman (2)" in text and "Reclose · #1" in text
    assert "Urutan sinyal rekaman gangguan" in text
    assert "+20,5" in text and "Aktif: trip" in text
    # Waveform events are in the ledger, and bounce and resets add only length.
    assert "Gangguan mulai" not in text
    assert "+21,0" not in text and "+77,0" not in text
    # A protection channel that never moved is evidence; a spare is not.
    assert "Kanal proteksi yang terekam tetapi tidak pernah aktif: LP SEND MJSNG2 (Teleproteksi)." in text
    assert "SPARE" not in text


def test_a_fault_without_a_ledger_asks_for_a_refresh():
    text = _pdf(ledger=None)
    assert "Penalaran belum tersedia untuk gangguan ini" in text
    assert "Urutan sinyal rekaman gangguan" not in text
