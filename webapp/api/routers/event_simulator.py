"""Deterministic notification simulator for mapped IEC 61850 events.

The scenarios in this module intentionally mirror the relay mapping table in
``TFA_Notif_Architecture_v1.0`` section 4.  The simulator is a lab surface for
showing how raw MMS/polling changes become clusters, notifications, grouped
incidents, and JSON artifacts without connecting to a real IED.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/event-simulator", tags=["event-simulator"])

TRIP_LNS = {"PDIS", "PDIF", "PTRC", "PTOC", "PSOF", "PTEF", "RDIF"}
MEASUREMENT_LNS = {"RFLO", "MMXU", "MMXN", "MMTR"}
STATUS_WORDS = ("52B", "CB HEALTHY", "CB MANUAL", "CB OPEN", "CB CLOSE", "POSITION")
ALARM_WORDS = ("ALARM", "FAIL", "BLOCK", "PRESSURE", "LOCKOUT", "ANOMALY")

TIER_META = {
    1: {
        "cluster": "GANGGUAN",
        "label": "Gangguan",
        "timing": "immediate",
        "color": "#dc2626",
    },
    2: {
        "cluster": "CB RECLOSE",
        "label": "CB Reclose",
        "timing": "delay_window_5000ms",
        "color": "#d97706",
    },
    3: {
        "cluster": "STATUS CB",
        "label": "Status CB",
        "timing": "debounce_500ms",
        "color": "#2563eb",
    },
    4: {
        "cluster": "ALARM OPERASIONAL",
        "label": "Alarm Operasional",
        "timing": "debounce_2000ms",
        "color": "#7c3aed",
    },
}

DOC_DECISION_TREE = [
    {
        "order": 1,
        "question": "Apakah LN sumber adalah fungsi proteksi yang output-nya perintah TRIP?",
        "result": "GANGGUAN / Tier 1",
        "ln": ["PDIS", "PDIF", "PTRC", "PTOC", "PSOF", "PTEF", "RDIF"],
        "timing": "Kirim segera tanpa delay.",
    },
    {
        "order": 2,
        "question": "Apakah LN sumber adalah RREC?",
        "result": "CB RECLOSE / Tier 2",
        "ln": ["RREC"],
        "timing": "Kirim setelah sequence RREC resolve atau delay window.",
    },
    {
        "order": 3,
        "question": "Apakah LN sumber adalah XCBR atau GGIO posisi/auxiliary CB?",
        "result": "STATUS CB / Tier 3",
        "ln": ["XCBR", "GGIO"],
        "timing": "Debounce 500 ms.",
    },
    {
        "order": 4,
        "question": "Apakah event alarm/warning relay non-trip?",
        "result": "ALARM OPERASIONAL / Tier 4",
        "ln": ["GGIO", "PSCH", "RDIF"],
        "timing": "Debounce 2000 ms.",
    },
    {
        "order": 5,
        "question": "Apakah event data pengukuran?",
        "result": "Tidak dibuat notifikasi.",
        "ln": ["RFLO", "MMXU"],
        "timing": "Tampilkan sebagai context/detail event.",
    },
]


@dataclass
class PendingNotification:
    tier: int
    incident_id: str
    first_ms: float
    last_ms: float
    event_ids: list[str] = field(default_factory=list)


def _doc_signal(
    *,
    display_name: str,
    relay_model: str,
    doc_section: str,
    ld: str,
    ln: str,
    do_name: str,
    da: str,
    function: str,
    ln_category: str,
    cluster: str | None,
    tier: int | None,
    condition: str,
    cdc: str = "ST",
    data_type: str = "ST",
    notification_allowed: bool = True,
    contextual_group_with_trip: bool = False,
) -> dict[str, Any]:
    return {
        "display_name": display_name,
        "relay_model": relay_model,
        "source_doc_section": doc_section,
        "ld": ld,
        "ln": ln,
        "do": do_name,
        "da": da,
        "cdc": cdc,
        "data_type": data_type,
        "function": function,
        "ln_category": ln_category,
        "cluster": cluster,
        "notif_tier": tier,
        "condition": condition,
        "notification_allowed": notification_allowed,
        "contextual_group_with_trip": contextual_group_with_trip,
    }


def _event(
    event_id: str,
    t_ms: float,
    signal_ref: str,
    value: Any,
    *,
    device_id: str,
    bay_name: str,
    source: str = "mms_poll",
    quality: str = "good",
    unit: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": event_id,
        "t_ms": t_ms,
        "source": source,
        "device_id": device_id,
        "bay_name": bay_name,
        "signal_ref": signal_ref,
        "value": value,
        "quality": quality,
    }
    if unit:
        item["unit"] = unit
    return item


NR_PCR931S_MAPPING = {
    "NR_DIFF_TRIP": _doc_signal(
        display_name="DIFF TRIP",
        relay_model="NR Electric NR-PCR931S",
        doc_section="4.1",
        ld="PROT",
        ln="PDIF3",
        do_name="OP",
        da="t[ST]",
        function="87L Differential Line",
        ln_category="TRIP",
        cluster="GANGGUAN",
        tier=1,
        condition="stVal=true",
    ),
    "NR_ZONE_1": _doc_signal(
        display_name="ZONE 1",
        relay_model="NR Electric NR-PCR931S",
        doc_section="4.1",
        ld="PROT",
        ln="PDIS3",
        do_name="OP",
        da="t[ST]",
        function="21L Zone 1 Distance",
        ln_category="TRIP",
        cluster="GANGGUAN",
        tier=1,
        condition="stVal=true",
    ),
    "NR_21_SEND": _doc_signal(
        display_name="21 Send",
        relay_model="NR Electric NR-PCR931S",
        doc_section="4.1",
        ld="PROT",
        ln="PSCH1",
        do_name="ProTx",
        da="t[ST]",
        function="85-1 PUTT/POTT Send",
        ln_category="TRIP",
        cluster="GANGGUAN",
        tier=1,
        condition="stVal=true",
    ),
    "NR_21_RECEIVE": _doc_signal(
        display_name="21 Receive",
        relay_model="NR Electric NR-PCR931S",
        doc_section="4.1",
        ld="PROT",
        ln="PSCH1",
        do_name="ProRx",
        da="t[ST]",
        function="85-1 PUTT/POTT Receive",
        ln_category="TRIP",
        cluster="GANGGUAN",
        tier=1,
        condition="stVal=true",
    ),
    "NR_AR_BLOCK": _doc_signal(
        display_name="AR BLOCK",
        relay_model="NR Electric NR-PCR931S",
        doc_section="4.1 note",
        ld="LD0",
        ln="BIGGIO3",
        do_name="Ind6",
        da="stVal[ST]",
        function="AR Block; Tier 4 if isolated, attach to Gangguan if after trip in same window",
        ln_category="ALARM",
        cluster="ALARM OPERASIONAL",
        tier=4,
        condition="stVal=true",
        contextual_group_with_trip=True,
    ),
    "NR_MCB_VT_FAIL": _doc_signal(
        display_name="MCB VT FAIL",
        relay_model="NR Electric NR-PCR931S",
        doc_section="4.1",
        ld="LD0",
        ln="BIGGIO3",
        do_name="Ind7",
        da="stVal[ST]",
        function="MCB VT / Fuse Failure",
        ln_category="ALARM",
        cluster="ALARM OPERASIONAL",
        tier=4,
        condition="stVal=true",
    ),
    "NR_DIFF_ALARM": _doc_signal(
        display_name="Diff Alarm",
        relay_model="NR Electric NR-PCR931S",
        doc_section="4.1",
        ld="PROT",
        ln="PDIF1",
        do_name="OvDifAlm",
        da="t[ST]",
        function="87L Diff Alarm pre-trip",
        ln_category="ALARM",
        cluster="ALARM OPERASIONAL",
        tier=4,
        condition="stVal=true",
    ),
    "NR_52B_R": _doc_signal(
        display_name="52B R",
        relay_model="NR Electric NR-PCR931S",
        doc_section="4.1",
        ld="LD0",
        ln="BIGGIO3",
        do_name="Ind1",
        da="stVal[ST]",
        function="CB Open Phase R auxiliary",
        ln_category="STATUS_CB",
        cluster="STATUS CB",
        tier=3,
        condition="stVal=true",
    ),
    "NR_52B_S": _doc_signal(
        display_name="52B S",
        relay_model="NR Electric NR-PCR931S",
        doc_section="4.1",
        ld="LD0",
        ln="BIGGIO3",
        do_name="Ind2",
        da="stVal[ST]",
        function="CB Open Phase S auxiliary",
        ln_category="STATUS_CB",
        cluster="STATUS CB",
        tier=3,
        condition="stVal=true",
    ),
    "NR_52B_T": _doc_signal(
        display_name="52B T",
        relay_model="NR Electric NR-PCR931S",
        doc_section="4.1",
        ld="LD0",
        ln="BIGGIO3",
        do_name="Ind3",
        da="stVal[ST]",
        function="CB Open Phase T auxiliary",
        ln_category="STATUS_CB",
        cluster="STATUS CB",
        tier=3,
        condition="stVal=true",
    ),
    "NR_FAULT_LOCATOR": _doc_signal(
        display_name="Fault Locator",
        relay_model="NR Electric NR-PCR931S",
        doc_section="4.1",
        ld="RCD",
        ln="RFLO1",
        do_name="FltDiskm",
        da="MX",
        function="Lokasi gangguan km",
        ln_category="MEASUREMENT",
        cluster=None,
        tier=None,
        condition="measurement only",
        cdc="MV",
        data_type="MX",
        notification_allowed=False,
    ),
}

ABB_RED670_MAPPING = {
    "ABB_87L_DIFF_TRIP": _doc_signal(
        display_name="87L DIFF TRIP",
        relay_model="ABB RED670",
        doc_section="4.2",
        ld="PROT",
        ln="PDIF1",
        do_name="Op",
        da="general[ST]",
        function="87L Line Differential",
        ln_category="TRIP",
        cluster="GANGGUAN",
        tier=1,
        condition="stVal=true",
    ),
    "ABB_ZONE_1": _doc_signal(
        display_name="ZONE 1",
        relay_model="ABB RED670",
        doc_section="4.2",
        ld="PROT",
        ln="PDIS1",
        do_name="Op",
        da="general[ST]",
        function="21 Zone 1 Distance",
        ln_category="TRIP",
        cluster="GANGGUAN",
        tier=1,
        condition="stVal=true",
    ),
    "ABB_TRIP_GENERAL": _doc_signal(
        display_name="TRIP GENERAL",
        relay_model="ABB RED670",
        doc_section="4.2",
        ld="PROT",
        ln="PTRC1",
        do_name="Tr",
        da="general[ST]",
        function="General Trip command",
        ln_category="TRIP",
        cluster="GANGGUAN",
        tier=1,
        condition="stVal=true",
    ),
    "ABB_85_SEND": _doc_signal(
        display_name="85 SEND",
        relay_model="ABB RED670",
        doc_section="4.2",
        ld="PROT",
        ln="PSCH1",
        do_name="ProTx",
        da="general[ST]",
        function="Pilot scheme Send",
        ln_category="TRIP",
        cluster="GANGGUAN",
        tier=1,
        condition="stVal=true",
    ),
    "ABB_AR_INITIATE": _doc_signal(
        display_name="AR INITIATE",
        relay_model="ABB RED670",
        doc_section="4.2",
        ld="PROT",
        ln="RREC1",
        do_name="AutoRecSt",
        da="stVal[ST]",
        function="Autorecloser initiated",
        ln_category="RECLOSE",
        cluster="CB RECLOSE",
        tier=2,
        condition="stVal=true",
    ),
    "ABB_AR_SUCCESS": _doc_signal(
        display_name="AR SUCCESS",
        relay_model="ABB RED670",
        doc_section="4.2",
        ld="PROT",
        ln="RREC1",
        do_name="AutoRecSt",
        da="stVal[ST]",
        function="Reclose successful",
        ln_category="RECLOSE",
        cluster="CB RECLOSE",
        tier=2,
        condition="stVal=false",
    ),
    "ABB_AR_BLOCK": _doc_signal(
        display_name="AR BLOCK",
        relay_model="ABB RED670",
        doc_section="4.2",
        ld="PROT",
        ln="RREC1",
        do_name="CBLock",
        da="stVal[ST]",
        function="AR Blocked by relay",
        ln_category="RECLOSE",
        cluster="CB RECLOSE",
        tier=2,
        condition="stVal=true",
    ),
    "ABB_CB_POSITION": _doc_signal(
        display_name="CB POSITION",
        relay_model="ABB RED670",
        doc_section="4.2",
        ld="PROT",
        ln="XCBR1",
        do_name="Pos",
        da="stVal[ST]",
        function="CB Open/Close position",
        ln_category="STATUS_CB",
        cluster="STATUS CB",
        tier=3,
        condition="any change",
    ),
    "ABB_FAULT_LOCATOR": _doc_signal(
        display_name="FAULT LOCATOR",
        relay_model="ABB RED670",
        doc_section="4.2",
        ld="PROT",
        ln="RFLO1",
        do_name="FltDiskm",
        da="instMag[MX]",
        function="Fault location km",
        ln_category="MEASUREMENT",
        cluster=None,
        tier=None,
        condition="measurement only",
        cdc="MV",
        data_type="MX",
        notification_allowed=False,
    ),
}

MICOM_P545_MAPPING = {
    "MICOM_GEN_TRIP": _doc_signal(
        display_name="GEN TRIP",
        relay_model="MiCOM Alstom/GE P545",
        doc_section="4.3",
        ld="PROT",
        ln="PTRC1",
        do_name="Tr",
        da="general[ST]",
        function="General Trip output",
        ln_category="TRIP",
        cluster="GANGGUAN",
        tier=1,
        condition="stVal=true",
    ),
    "MICOM_DIFF_OPERATE": _doc_signal(
        display_name="DIFF OPERATE",
        relay_model="MiCOM Alstom/GE P545",
        doc_section="4.3",
        ld="PROT",
        ln="PDIF1",
        do_name="Op",
        da="general[ST]",
        function="87L Differential Operate",
        ln_category="TRIP",
        cluster="GANGGUAN",
        tier=1,
        condition="stVal=true",
    ),
    "MICOM_ZONE_1": _doc_signal(
        display_name="ZONE 1",
        relay_model="MiCOM Alstom/GE P545",
        doc_section="4.3",
        ld="PROT",
        ln="PDIS1",
        do_name="Op",
        da="general[ST]",
        function="21 Zone 1",
        ln_category="TRIP",
        cluster="GANGGUAN",
        tier=1,
        condition="stVal=true",
    ),
    "MICOM_SOTF_Z1": _doc_signal(
        display_name="SOTF Z1",
        relay_model="MiCOM Alstom/GE P545",
        doc_section="4.3",
        ld="PROT",
        ln="GGIO1",
        do_name="Ind",
        da="stVal[ST]",
        function="SOTF Zone 1 via DDB",
        ln_category="TRIP",
        cluster="GANGGUAN",
        tier=1,
        condition="stVal=true",
    ),
    "MICOM_TOR_Z1": _doc_signal(
        display_name="TOR Z1",
        relay_model="MiCOM Alstom/GE P545",
        doc_section="4.3",
        ld="PROT",
        ln="GGIO2",
        do_name="Ind",
        da="stVal[ST]",
        function="Trip on Reclose Z1 via DDB",
        ln_category="TRIP",
        cluster="GANGGUAN",
        tier=1,
        condition="stVal=true",
    ),
    "MICOM_AR_INITIATE": _doc_signal(
        display_name="AR INITIATE",
        relay_model="MiCOM Alstom/GE P545",
        doc_section="4.3",
        ld="PROT",
        ln="RREC1",
        do_name="AutoRecSt",
        da="stVal[ST]",
        function="AR Initiate",
        ln_category="RECLOSE",
        cluster="CB RECLOSE",
        tier=2,
        condition="stVal=true",
    ),
    "MICOM_AR_READY": _doc_signal(
        display_name="AR READY",
        relay_model="MiCOM Alstom/GE P545",
        doc_section="4.3",
        ld="PROT",
        ln="RREC1",
        do_name="Readyin",
        da="stVal[ST]",
        function="AR Ready state",
        ln_category="RECLOSE",
        cluster="CB RECLOSE",
        tier=2,
        condition="stVal=true",
    ),
    "MICOM_CB_POS": _doc_signal(
        display_name="CB POS (3PH)",
        relay_model="MiCOM Alstom/GE P545",
        doc_section="4.3",
        ld="PROT",
        ln="XCBR1",
        do_name="Pos",
        da="stVal[ST]",
        function="CB Position 3-phase",
        ln_category="STATUS_CB",
        cluster="STATUS CB",
        tier=3,
        condition="any change",
    ),
    "MICOM_COM_FAILURE": _doc_signal(
        display_name="COM FAILURE",
        relay_model="MiCOM Alstom/GE P545",
        doc_section="4.3",
        ld="PROT",
        ln="GGIO_CF",
        do_name="Ind",
        da="stVal[ST]",
        function="Comm Fail vendor DDB",
        ln_category="ALARM",
        cluster="ALARM OPERASIONAL",
        tier=4,
        condition="stVal=true",
    ),
    "MICOM_FAULT_LOC": _doc_signal(
        display_name="FAULT I/V/LOC",
        relay_model="MiCOM Alstom/GE P545",
        doc_section="4.3",
        ld="RCD",
        ln="RFLO1",
        do_name="FltA*/V*",
        da="instMag[MX]",
        function="Pengukuran gangguan",
        ln_category="MEASUREMENT",
        cluster=None,
        tier=None,
        condition="measurement only",
        cdc="MV",
        data_type="MX",
        notification_allowed=False,
    ),
}

SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "nr-ponorogo-87l-distance",
        "title": "NR-PCR931S Ponorogo: 87L + 21 grouped trip",
        "subtitle": "DIFF TRIP, Zone 1, PSCH, 52B, AR Block, dan RFLO sesuai Sec 4.1.",
        "description": (
            "Scenario ini memakai mapping NR Electric NR-PCR931S pada dokumen. "
            "Event PSCH Send/Receive tetap Tier 1 karena tabel mapping menandainya "
            "GANGGUAN, sedangkan RFLO hanya menjadi context dan AR BLOCK menempel "
            "ke incident gangguan bila muncul setelah trip."
        ),
        "station_name": "GI PONOROGO 150 kV",
        "asset_name": "Ponorogo - Madiun Line 1",
        "asset_id": "GI-PONOROGO/LINE-MADIUN-1",
        "data_mapping": NR_PCR931S_MAPPING,
        "events": [
            _event("e001", 0, "NR_DIFF_TRIP", True, device_id="NR-PCR931S-PNG-MDN1", bay_name="Line Madiun 1"),
            _event("e002", 8, "NR_ZONE_1", True, device_id="NR-PCR931S-PNG-MDN1", bay_name="Line Madiun 1"),
            _event("e003", 14, "NR_21_SEND", True, device_id="NR-PCR931S-PNG-MDN1", bay_name="Line Madiun 1"),
            _event("e004", 18, "NR_21_RECEIVE", True, device_id="NR-PCR931S-PNG-MDN1", bay_name="Line Madiun 1"),
            _event("e005", 34, "NR_52B_R", True, device_id="BCU-PNG-MDN1", bay_name="Line Madiun 1"),
            _event("e006", 36, "NR_52B_S", True, device_id="BCU-PNG-MDN1", bay_name="Line Madiun 1"),
            _event("e007", 38, "NR_52B_T", True, device_id="BCU-PNG-MDN1", bay_name="Line Madiun 1"),
            _event("e008", 170, "NR_FAULT_LOCATOR", 23.7, device_id="NR-PCR931S-PNG-MDN1", bay_name="Line Madiun 1", unit="km"),
            _event("e009", 420, "NR_AR_BLOCK", True, device_id="NR-PCR931S-PNG-MDN1", bay_name="Line Madiun 1"),
        ],
    },
    {
        "id": "abb-red670-double-reclose",
        "title": "ABB RED670: Zone 1, AR success, second trip, AR block",
        "subtitle": "RREC adalah Tier 2; CB position adalah Tier 3; RFLO tidak menjadi notif.",
        "description": (
            "Scenario ini mengikuti mapping ABB RED670 pada Sec 4.2. Notifikasi "
            "Tier 1 muncul segera saat Zone 1 pertama, RREC ditahan sampai window "
            "reclose selesai, dan CB POSITION tetap Tier 3 meskipun terlihat seperti "
            "bagian reclose."
        ),
        "station_name": "GI PONOROGO 150 kV",
        "asset_name": "Ponorogo - Pacitan Line 2",
        "asset_id": "GI-PONOROGO/LINE-PACITAN-2",
        "data_mapping": ABB_RED670_MAPPING,
        "events": [
            _event("e101", 0, "ABB_ZONE_1", True, device_id="ABB-RED670-PNG-PCT2", bay_name="Line Pacitan 2"),
            _event("e102", 7, "ABB_TRIP_GENERAL", True, device_id="ABB-RED670-PNG-PCT2", bay_name="Line Pacitan 2"),
            _event("e103", 31, "ABB_CB_POSITION", "open", device_id="BCU-PNG-PCT2", bay_name="Line Pacitan 2"),
            _event("e104", 150, "ABB_FAULT_LOCATOR", 41.2, device_id="ABB-RED670-PNG-PCT2", bay_name="Line Pacitan 2", unit="km"),
            _event("e105", 720, "ABB_AR_INITIATE", True, device_id="ABB-RED670-PNG-PCT2", bay_name="Line Pacitan 2"),
            _event("e106", 1240, "ABB_CB_POSITION", "closed", device_id="BCU-PNG-PCT2", bay_name="Line Pacitan 2"),
            _event("e107", 1260, "ABB_AR_SUCCESS", False, device_id="ABB-RED670-PNG-PCT2", bay_name="Line Pacitan 2"),
            _event("e108", 1400, "ABB_ZONE_1", True, device_id="ABB-RED670-PNG-PCT2", bay_name="Line Pacitan 2"),
            _event("e109", 1408, "ABB_TRIP_GENERAL", True, device_id="ABB-RED670-PNG-PCT2", bay_name="Line Pacitan 2"),
            _event("e110", 1442, "ABB_CB_POSITION", "open", device_id="BCU-PNG-PCT2", bay_name="Line Pacitan 2"),
            _event("e111", 6200, "ABB_AR_BLOCK", True, device_id="ABB-RED670-PNG-PCT2", bay_name="Line Pacitan 2"),
        ],
    },
    {
        "id": "micom-p545-vendor-ddb",
        "title": "MiCOM P545: TOR/SOTF via GGIO plus isolated COM FAILURE",
        "subtitle": "GGIO bisa Tier 1 atau Tier 4 tergantung mapping vendor DDB di Sec 4.3.",
        "description": (
            "Scenario ini sengaja menunjukkan nuance dokumen: MiCOM TOR Z1/SOTF "
            "memakai GGIO vendor DDB tetapi tetap GANGGUAN Tier 1, sedangkan COM "
            "FAILURE memakai GGIO_CF dan masuk Alarm Operasional Tier 4."
        ),
        "station_name": "GI KEDIRI 150 kV",
        "asset_name": "Kediri - Tulungagung Line 1",
        "asset_id": "GI-KEDIRI/LINE-TLG-1",
        "data_mapping": MICOM_P545_MAPPING,
        "events": [
            _event("e201", 0, "MICOM_AR_INITIATE", True, device_id="MICOM-P545-KDR-TLG1", bay_name="Line Tulungagung 1"),
            _event("e202", 320, "MICOM_CB_POS", "closed", device_id="BCU-KDR-TLG1", bay_name="Line Tulungagung 1"),
            _event("e203", 440, "MICOM_TOR_Z1", True, device_id="MICOM-P545-KDR-TLG1", bay_name="Line Tulungagung 1"),
            _event("e204", 448, "MICOM_GEN_TRIP", True, device_id="MICOM-P545-KDR-TLG1", bay_name="Line Tulungagung 1"),
            _event("e205", 490, "MICOM_CB_POS", "open", device_id="BCU-KDR-TLG1", bay_name="Line Tulungagung 1"),
            _event("e206", 620, "MICOM_FAULT_LOC", 18.6, device_id="MICOM-P545-KDR-TLG1", bay_name="Line Tulungagung 1", unit="km"),
            _event("e207", 7400, "MICOM_COM_FAILURE", True, device_id="MICOM-P545-KDR-TLG1", bay_name="Line Tulungagung 1"),
        ],
    },
]


def _mapping_for(scenario: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    try:
        return scenario["data_mapping"][event["signal_ref"]]
    except KeyError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Scenario {scenario['id']} references unknown signal {event.get('signal_ref')}",
        ) from exc


def _condition_matches(value: Any, mapping: dict[str, Any]) -> bool:
    condition = str(mapping.get("condition") or "").strip().lower()
    if condition in {"", "any change", "measurement only"}:
        return True
    if condition == "stval=true":
        return value is True or value == 1 or str(value).strip().lower() == "true"
    if condition == "stval=false":
        return value is False or value == 0 or str(value).strip().lower() == "false"
    return True


def _classify_event(event: dict[str, Any], mapping: dict[str, Any]) -> dict[str, Any]:
    ln = str(mapping.get("ln", "")).upper()
    ln_category = str(mapping.get("ln_category") or "").upper()
    label = str(mapping.get("display_name") or event.get("signal_ref") or "")
    label_upper = label.upper()
    data_type = str(mapping.get("data_type") or "").upper()
    tier = mapping.get("notif_tier")
    condition_match = _condition_matches(event.get("value"), mapping)

    base = {
        "doc_section": mapping.get("source_doc_section"),
        "relay_model": mapping.get("relay_model"),
        "ln_category": mapping.get("ln_category"),
        "doc_expected_cluster": mapping.get("cluster"),
        "doc_expected_tier": tier,
        "condition": mapping.get("condition"),
        "condition_match": condition_match,
    }

    if not condition_match:
        return {
            **base,
            "category": "CONDITION_NOT_MET",
            "tier": None,
            "cluster": None,
            "notify": False,
            "timing": "ignored",
            "reason": f"Nilai event tidak memenuhi condition dokumen: {mapping.get('condition')}",
        }

    if ln_category == "ARTIFACT":
        return {
            **base,
            "category": "ARTIFACT",
            "tier": None,
            "cluster": None,
            "notify": False,
            "timing": "attach_to_incident",
            "reason": "Artifact muncul sebagai lampiran/detail incident, bukan notifikasi.",
        }

    if (
        mapping.get("notification_allowed") is False
        or ln_category == "MEASUREMENT"
        or ln in MEASUREMENT_LNS
        or data_type == "MX"
    ):
        return {
            **base,
            "category": "MEASUREMENT",
            "tier": None,
            "cluster": None,
            "notify": False,
            "timing": "context_only",
            "reason": "Data bertipe MX/measurement di dokumen tidak boleh dibuat event notifikasi.",
        }

    if isinstance(tier, int):
        meta = TIER_META[tier]
        return {
            **base,
            "category": ln_category,
            "tier": tier,
            "cluster": meta["cluster"],
            "notify": True,
            "timing": meta["timing"],
            "reason": f"Mapping dokumen Sec {mapping.get('source_doc_section')} menetapkan {label} ke {meta['cluster']} / Tier {tier}.",
        }

    if ln_category == "TRIP" or ln in TRIP_LNS:
        return {
            **base,
            "category": "TRIP",
            "tier": 1,
            "cluster": TIER_META[1]["cluster"],
            "notify": True,
            "timing": TIER_META[1]["timing"],
            "reason": f"Fallback decision tree: LN {ln} adalah fungsi trip.",
        }

    if ln_category == "RECLOSE" or ln.startswith("RREC"):
        return {
            **base,
            "category": "RECLOSE",
            "tier": 2,
            "cluster": TIER_META[2]["cluster"],
            "notify": True,
            "timing": TIER_META[2]["timing"],
            "reason": "Fallback decision tree: LN RREC adalah autorecloser.",
        }

    if ln_category == "STATUS_CB" or ln.startswith("XCBR") or any(word in label_upper for word in STATUS_WORDS):
        return {
            **base,
            "category": "STATUS_CB",
            "tier": 3,
            "cluster": TIER_META[3]["cluster"],
            "notify": True,
            "timing": TIER_META[3]["timing"],
            "reason": "Fallback decision tree: status CB memakai debounce 500 ms.",
        }

    if ln_category == "ALARM" or any(word in label_upper for word in ALARM_WORDS):
        return {
            **base,
            "category": "ALARM",
            "tier": 4,
            "cluster": TIER_META[4]["cluster"],
            "notify": True,
            "timing": TIER_META[4]["timing"],
            "reason": "Fallback decision tree: alarm operasional memakai debounce 2000 ms.",
        }

    return {
        **base,
        "category": "UNMAPPED",
        "tier": None,
        "cluster": None,
        "notify": False,
        "timing": "ignored",
        "reason": "Sinyal belum punya LN category yang bisa dinotifikasikan.",
    }


def _new_incident(scenario: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    start_ms = float(event["t_ms"])
    return {
        "id": f"inc-{scenario['id']}-{int(start_ms):04d}",
        "station_name": scenario["station_name"],
        "asset_name": scenario["asset_name"],
        "asset_id": scenario["asset_id"],
        "start_ms": start_ms,
        "last_event_ms": start_ms,
        "status": "open",
        "primary_tier": None,
        "title": "Incident window opened",
        "summary": "",
        "event_count": 0,
        "events": [],
        "measurements": [],
        "artifacts": [],
        "notifications": [],
        "relay_functions": [],
        "cb_states": [],
        "reclose_sequence": [],
    }


def _signal_path(mapping: dict[str, Any]) -> str:
    return ".".join(
        part
        for part in (
            str(mapping.get("ld") or ""),
            str(mapping.get("ln") or ""),
            str(mapping.get("do") or ""),
            str(mapping.get("da") or ""),
        )
        if part
    )


def _add_event_to_incident(
    incident: dict[str, Any],
    event: dict[str, Any],
    mapping: dict[str, Any],
    classification: dict[str, Any],
) -> None:
    category = classification["category"]
    item = {
        "id": event["id"],
        "t_ms": float(event["t_ms"]),
        "source": event.get("source"),
        "device_id": event.get("device_id"),
        "bay_name": event.get("bay_name"),
        "signal_ref": event.get("signal_ref"),
        "signal_path": _signal_path(mapping),
        "label": mapping.get("display_name") or event.get("signal_ref"),
        "relay_model": mapping.get("relay_model"),
        "doc_section": mapping.get("source_doc_section"),
        "function": mapping.get("function"),
        "condition": mapping.get("condition"),
        "value": event.get("value"),
        "unit": event.get("unit"),
        "category": category,
        "tier": classification.get("tier"),
    }
    incident["last_event_ms"] = max(float(event["t_ms"]), float(incident["last_event_ms"]))

    if category == "MEASUREMENT":
        incident["measurements"].append(item)
        return
    if category == "ARTIFACT":
        incident["artifacts"].append(item)
        return
    if category == "CONDITION_NOT_MET":
        return

    incident["events"].append(item)
    incident["event_count"] = len(incident["events"])
    tier = classification.get("tier")
    if tier is not None:
        if incident["primary_tier"] is None:
            incident["primary_tier"] = tier
        else:
            incident["primary_tier"] = min(int(incident["primary_tier"]), int(tier))

    if category == "TRIP":
        label = str(mapping.get("display_name") or "")
        if label and label not in incident["relay_functions"]:
            incident["relay_functions"].append(label)
    elif category == "STATUS_CB":
        incident["cb_states"].append(item)
    elif category == "RECLOSE":
        incident["reclose_sequence"].append(item)

    _refresh_incident_text(incident)


def _refresh_incident_text(incident: dict[str, Any]) -> None:
    relays = incident["relay_functions"]
    labels_upper = " ".join(relays).upper()
    has_diff = "DIFF" in labels_upper or "87L" in labels_upper
    has_distance = "ZONE" in labels_upper or "SOTF" in labels_upper
    has_tor = "TOR" in labels_upper
    has_psch = "21 SEND" in labels_upper or "21 RECEIVE" in labels_upper or "85 SEND" in labels_upper
    has_reclose = bool(incident["reclose_sequence"])

    if has_tor:
        title = "Gangguan Line: Trip on Reclose"
        summary = "TOR/SOTF vendor DDB dipetakan sebagai GANGGUAN Tier 1 sesuai tabel relay."
    elif has_diff and has_distance:
        title = "Gangguan Line: Differential + Distance"
        summary = "Differential, distance, dan/atau PSCH berada dalam incident window yang sama."
    elif has_distance and has_reclose:
        title = "Gangguan Line dengan Reclose Sequence"
        summary = "Distance trip muncul bersama rangkaian RREC dan status CB."
    elif has_psch:
        title = "Gangguan Line: Pilot Scheme"
        summary = "PSCH send/receive dipetakan sebagai Tier 1 pada data mapping relay."
    elif relays:
        title = "Gangguan Proteksi"
        summary = ", ".join(relays[:3])
    elif has_reclose:
        title = "CB Reclose Sequence"
        summary = "Rangkaian RREC terdeteksi."
    else:
        title = "Operational Event"
        summary = "Incident window berisi status/alarm/context."

    incident["title"] = title
    incident["summary"] = summary


def _event_by_id(incident: dict[str, Any], event_id: str) -> dict[str, Any] | None:
    for collection_name in ("events", "measurements", "artifacts"):
        for item in incident.get(collection_name, []):
            if item.get("id") == event_id:
                return item
    return None


def _notification_title(incident: dict[str, Any], tier: int, event_ids: list[str]) -> str:
    first = _event_by_id(incident, event_ids[0]) if event_ids else None
    trigger = str(first.get("label")) if first else incident["title"]
    return f"{TIER_META[tier]['cluster']}: {trigger}"


def _refresh_notification(notification: dict[str, Any], incident: dict[str, Any]) -> None:
    event_ids = list(notification.get("event_ids") or [])
    notification["title"] = _notification_title(incident, int(notification["tier"]), event_ids)
    notification["message"] = (
        f"{incident['asset_name']} -> {incident['station_name']} "
        f"[+{max(len(event_ids) - 1, 0)} events]"
    )
    notification["badge_count"] = len(event_ids)
    notification["group_event_count"] = incident["event_count"]


def _build_notification(
    incident: dict[str, Any],
    pending: PendingNotification,
    emit_ms: float,
    note: str,
) -> dict[str, Any]:
    tier = pending.tier
    meta = TIER_META[tier]
    notification = {
        "id": f"notif-{incident['id']}-t{tier}-{len(incident['notifications']) + 1}",
        "incident_id": incident["id"],
        "tier": tier,
        "cluster": meta["cluster"],
        "label": meta["label"],
        "color": meta["color"],
        "title": "",
        "message": "",
        "first_event_ms": pending.first_ms,
        "last_event_ms": pending.last_ms,
        "emit_ms": emit_ms,
        "event_ids": list(pending.event_ids),
        "timing_note": note,
        "badge_count": len(pending.event_ids),
        "group_event_count": incident["event_count"],
    }
    _refresh_notification(notification, incident)
    incident["notifications"].append(notification)
    return notification


def _find_notification(incident: dict[str, Any], tier: int) -> dict[str, Any] | None:
    for notification in incident["notifications"]:
        if notification.get("tier") == tier:
            return notification
    return None


def _snapshot(incident: dict[str, Any] | None) -> dict[str, Any] | None:
    if incident is None:
        return None
    return {
        "id": incident["id"],
        "title": incident["title"],
        "primary_tier": incident["primary_tier"],
        "event_count": incident["event_count"],
        "measurement_count": len(incident["measurements"]),
        "artifact_count": len(incident["artifacts"]),
        "notification_count": len(incident["notifications"]),
        "last_event_ms": incident["last_event_ms"],
        "relay_functions": list(incident["relay_functions"]),
    }


def _scenario_by_id(scenario_id: str) -> dict[str, Any] | None:
    return next((scenario for scenario in SCENARIOS if scenario["id"] == scenario_id), None)


def process_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    scenario = deepcopy(scenario)
    events = sorted(scenario["events"], key=lambda event: float(event["t_ms"]))
    incident: dict[str, Any] | None = None
    incidents: list[dict[str, Any]] = []
    notifications: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    ignored_measurements: list[dict[str, Any]] = []
    pending_debounce: dict[str, PendingNotification] = {}
    pending_reclose: dict[str, PendingNotification] = {}

    def flush_due(now_ms: float) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        if incident is None:
            return emitted

        for key, pending in list(pending_debounce.items()):
            delay = 500.0 if pending.tier == 3 else 2000.0
            if now_ms - pending.last_ms >= delay:
                note = f"debounced for {int(delay)} ms"
                emitted.append(_build_notification(incident, pending, pending.last_ms + delay, note))
                del pending_debounce[key]

        for key, pending in list(pending_reclose.items()):
            if now_ms - pending.last_ms >= 5000.0:
                emitted.append(
                    _build_notification(
                        incident,
                        pending,
                        pending.last_ms + 5000.0,
                        "RREC sequence delayed until 5 s quiet window",
                    )
                )
                del pending_reclose[key]

        notifications.extend(emitted)
        return emitted

    for step_no, event in enumerate(events, start=1):
        emitted_before = flush_due(float(event["t_ms"]))
        mapping = _mapping_for(scenario, event)
        classification = _classify_event(event, mapping)

        if incident is None or float(event["t_ms"]) - float(incident["last_event_ms"]) > 5000.0:
            if incident is not None:
                incident["status"] = "closed"
            incident = _new_incident(scenario, event)
            incidents.append(incident)

        _add_event_to_incident(incident, event, mapping, classification)

        emitted_now: list[dict[str, Any]] = []
        decision = "context_only"
        tier = classification.get("tier")
        if classification["category"] == "MEASUREMENT":
            ignored_measurements.append(
                {
                    "event_id": event["id"],
                    "signal_ref": event["signal_ref"],
                    "label": mapping.get("display_name"),
                    "relay_model": mapping.get("relay_model"),
                    "doc_section": mapping.get("source_doc_section"),
                    "value": event.get("value"),
                    "unit": event.get("unit"),
                    "reason": classification["reason"],
                }
            )
        elif tier == 1:
            existing = _find_notification(incident, 1)
            if existing is None:
                pending = PendingNotification(
                    tier=1,
                    incident_id=incident["id"],
                    first_ms=float(event["t_ms"]),
                    last_ms=float(event["t_ms"]),
                    event_ids=[event["id"]],
                )
                emitted_now.append(
                    _build_notification(incident, pending, float(event["t_ms"]), "Tier 1 dikirim segera")
                )
                notifications.extend(emitted_now)
                decision = "notify_immediately"
            else:
                existing["event_ids"].append(event["id"])
                existing["last_event_ms"] = float(event["t_ms"])
                existing["timing_note"] = "Tier 1 already sent; event appended to grouped incident"
                _refresh_notification(existing, incident)
                decision = "append_to_existing_tier1_group"
        elif (
            tier == 4
            and mapping.get("contextual_group_with_trip")
            and incident.get("primary_tier") == 1
        ):
            decision = "attach_alarm_to_existing_gangguan_group"
        elif tier == 2:
            key = f"{incident['id']}:tier2"
            pending = pending_reclose.get(key)
            if pending is None:
                pending_reclose[key] = PendingNotification(
                    tier=2,
                    incident_id=incident["id"],
                    first_ms=float(event["t_ms"]),
                    last_ms=float(event["t_ms"]),
                    event_ids=[event["id"]],
                )
            else:
                pending.last_ms = float(event["t_ms"])
                pending.event_ids.append(event["id"])
            decision = "queued_reclose_delay"
        elif tier in {3, 4}:
            key = f"{incident['id']}:tier{tier}:{event['signal_ref']}"
            pending_debounce[key] = PendingNotification(
                tier=int(tier),
                incident_id=incident["id"],
                first_ms=float(event["t_ms"]),
                last_ms=float(event["t_ms"]),
                event_ids=[event["id"]],
            )
            decision = "queued_debounce"
        elif classification["category"] == "ARTIFACT":
            decision = "attached_artifact"
        elif classification["category"] == "CONDITION_NOT_MET":
            decision = "ignored_condition_not_met"

        trace.append(
            {
                "step": step_no,
                "t_ms": float(event["t_ms"]),
                "raw_event": event,
                "mapping": mapping,
                "classification": classification,
                "decision": decision,
                "emitted_before_this_event": emitted_before,
                "emitted_from_this_event": emitted_now,
                "incident_snapshot": _snapshot(incident),
            }
        )

    final_now = max((float(event["t_ms"]) for event in events), default=0.0) + 6000.0
    final_emitted = flush_due(final_now)
    if trace:
        trace.append(
            {
                "step": len(trace) + 1,
                "t_ms": final_now,
                "raw_event": None,
                "mapping": None,
                "classification": None,
                "decision": "flush_pending_timers",
                "emitted_before_this_event": final_emitted,
                "emitted_from_this_event": [],
                "incident_snapshot": _snapshot(incident),
            }
        )

    if incident is not None:
        incident["status"] = "closed"

    artifacts = {
        "source_document": "TFA_Notif_Architecture_v1.0, Sec 3-4 and Sec 9 backlog",
        "decision_tree": DOC_DECISION_TREE,
        "raw_mms_events": events,
        "data_mapping": scenario["data_mapping"],
        "incident_window_rule": {
            "window_ms_after_last_event": 5000,
            "group_key": "bay/device incident window",
            "tier_1": "send first notification immediately; append later Tier 1 events to same grouped incident",
            "tier_2": "send after RREC quiet window",
            "tier_3": "debounce 500 ms",
            "tier_4": "debounce 2000 ms, unless contextual alarm belongs to active Tier 1 incident",
            "measurement": "never notify; attach as context/detail event",
        },
        "ignored_measurements": ignored_measurements,
        "notification_outbox": notifications,
    }

    return {
        "scenario": {
            key: scenario[key]
            for key in ("id", "title", "subtitle", "description", "station_name", "asset_name", "asset_id")
        },
        "incidents": incidents,
        "notifications": notifications,
        "trace": trace,
        "artifacts": artifacts,
    }


@router.get("/scenarios")
async def list_scenarios():
    return [
        {
            "id": scenario["id"],
            "title": scenario["title"],
            "subtitle": scenario["subtitle"],
            "station_name": scenario["station_name"],
            "asset_name": scenario["asset_name"],
            "event_count": len(scenario["events"]),
        }
        for scenario in SCENARIOS
    ]


@router.get("/scenarios/{scenario_id}")
async def get_scenario(scenario_id: str):
    scenario = _scenario_by_id(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Simulator scenario not found.")
    return process_scenario(scenario)
