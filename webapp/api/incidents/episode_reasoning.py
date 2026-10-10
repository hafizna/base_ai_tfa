"""The reasoning ledger for one fault of an incident.

A fault's conclusions come from the record that holds it (its ``reasoning``
chain, see ``webapp.api.fault_reasoning``). An incident adds what that record
alone cannot see:

- the reclose, when it was captured in another file of the same recorder
  (F6.4, F6.5);
- the other line end's view of the same fault (F7.3): its phases, clearing
  time and breaker, and the teleprotection it shows — a receive answered by a
  send at a weak-infeed end is the echo of a POTT scheme;
- the cause: the AI reading of the fault record (F8.1), the sequence pattern
  of the incident (F8.2), and what lightning sub-mechanisms need (F8.3);
- review flags raised across the records (e.g. the two ends reading
  different phases).

Rows keep the shape of the record chain (label, title, evidence, rule IDs,
confidence), so one renderer shows both.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Optional

from ..fault_reasoning import _int, _ms, _num, _rule_order, is_send_channel
from .models import FaultEpisode, IncidentRecord

_PLN = {"A": "R", "B": "S", "C": "T"}
_ORDER = ("A", "B", "C")

_PATTERN_TITLES = {
    "REFAULT_AFTER_SUCCESSFUL_RECLOSE": "Gangguan berulang setelah reclose berhasil: indikasi kontak fisik",
    "FAILED_RECLOSE_INDICATES_PERMANENT_FAULT": "Reclose gagal: indikasi gangguan permanen",
    "ESCALATING_PHASE_INVOLVEMENT": "Fasa terganggu bertambah: indikasi kontak fisik yang meluas",
    "RECURRING_SAME_SIGNATURE": "Gangguan berulang dengan pola sama",
    "SINGLE_TRANSIENT_NO_RECURRENCE": "Satu gangguan singkat, reclose berhasil: pola transien",
    "REPEATED_ESCALATING_SIGNATURE_AMBIGUOUS": "Petir atau kontak fisik — belum bisa dibedakan",
    "POSSIBLE_EVOLVING_FAULT": "Kemungkinan gangguan yang berkembang",
}
_CAUSE_NAMES = {
    "PETIR": "Petir", "POHON": "Pohon", "LAYANG": "Layang-layang", "BENDA_ASING": "Benda asing",
    "KONDUKTOR": "Konduktor", "PERALATAN": "Peralatan", "HEWAN": "Hewan",
}


def _phases_text(phases: list[str]) -> str:
    names = [_PLN[p] for p in _ORDER if p in phases]
    if not names:
        return "?"
    return f"{names[0]}-N" if len(names) == 1 else "-".join(names)


def _row(key: str, step: int, label: str, title: str, evidence: list[str], rules: list[str], confidence: str,
         value: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return {
        "key": key, "step": step, "label": label, "title": title, "evidence": evidence,
        "rules": sorted(set(rules), key=_rule_order), "confidence": confidence, "value": value or {}, "conflicts": [],
    }


def _reasoning(record: Optional[IncidentRecord]) -> dict[str, Any]:
    return ((record.canonical_snapshot or {}).get("reasoning") or {}) if record is not None else {}


def _rows_by_key(reasoning: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["key"]: row for row in reasoning.get("conclusions") or []}


def _name(record: IncidentRecord) -> str:
    """"ZQ6E" for "ZQ6E.cfg"; an IEEE C37.232 name ("230821,081508780,+7h0,…")
    by its leading date and time — as the incident page names records."""
    stem = re.split(r"\.(?:cfg|dat|cff)\b", record.source_filename or "", flags=re.IGNORECASE)[0].strip()
    comname = re.match(r"^(\d{6},\d{6,9}),[+-]?\d", stem)
    return (comname.group(1) if comname else stem) or record.analysis_id[:8]


def _fault_record(episode: FaultEpisode, records_by_id: dict[str, IncidentRecord]) -> Optional[IncidentRecord]:
    other_ids = {rid for o in (episode.observed_facts or {}).get("other_recorders") or [] for rid in o["member_record_ids"]}
    for rid in episode.member_record_ids:
        record = records_by_id.get(rid)
        if record is not None and rid not in other_ids and _reasoning(record).get("has_fault"):
            return record
    return None


def _reclose_row(row: dict[str, Any], episode: FaultEpisode, records_by_id: dict[str, IncidentRecord],
                 fault: IncidentRecord) -> dict[str, Any]:
    """The trip-and-reclose row, completed with a reclose captured in its own file."""
    facts = episode.observed_facts or {}
    outcome = episode.reclose_outcome
    if (row.get("value") or {}).get("reclose_success") is not None:
        return row
    if outcome is None:
        row = copy.deepcopy(row)
        row["title"] = row["title"].replace("reclose tidak terekam di rekaman ini", "tidak ada reclose yang terekam di insiden ini")
        return row
    other_ids = {rid for o in facts.get("other_recorders") or [] for rid in o["member_record_ids"]}
    capture = next(
        (records_by_id[rid] for rid in episode.member_record_ids
         if rid in records_by_id and rid != fault.incident_record_id and rid not in other_ids
         and ((records_by_id[rid].canonical_snapshot or {}).get("protection_interpretation") or {}).get("event_class") == "RECLOSE_CAPTURE"),
        None,
    )
    row = copy.deepcopy(row)
    mode = row["title"].split(";")[0].split(", reclose")[0]
    dead = facts.get("reclose_dead_time_s")
    after = f" setelah {_num(dead)} s" if isinstance(dead, (int, float)) else ""
    where = f" (rekaman {_name(capture)})" if capture is not None else ""
    row["title"] = f"{mode}, reclose {'berhasil' if outcome == 'successful' else 'gagal'}{after}{where}"
    if capture is not None:
        row["evidence"] = [
            f"Reclose terekam di {_name(capture)}: rekaman dimulai saat PMT terbuka dan menangkap PMT menutup."
        ] + row["evidence"]
    row["rules"] = sorted(set(row["rules"]) | {"F6.4", "F6.5"}, key=_rule_order)
    return row


def _other_end_rows(episode: FaultEpisode, records_by_id: dict[str, IncidentRecord],
                    chain: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """F7.3: the other line end's view of the same fault, and the review
    flags that come from comparing the two ends."""
    local = _rows_by_key(chain)
    silent_send = [
        s["channel"] for s in (chain.get("signals") or {}).get("silent") or [] if is_send_channel(s["channel"])
    ]
    rows: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []
    for other in (episode.observed_facts or {}).get("other_recorders") or []:
        if other.get("same_station"):
            continue
        record = records_by_id.get(other.get("fault_record_id") or "")
        remote = _rows_by_key(_reasoning(record))
        evidence: list[str] = []
        phases = other.get("faulted_phases") or []
        fct = other.get("fct_ms")
        title = f"{other['station']}: fasa {_phases_text(phases)}" if phases else other["station"]
        if isinstance(fct, (int, float)):
            title += f", padam {_int(fct)} ms"
        remote_phase = remote.get("phases") or {}
        if (remote_phase.get("value") or {}).get("weak_infeed"):
            evidence.append(next((e for e in remote_phase["evidence"] if "weak infeed" in e), "Ujung weak infeed."))
        local_phases = (local.get("phases") or {}).get("value", {}).get("phases") or []
        if phases and local_phases:
            evidence.append(
                "Fasa sama di kedua ujung." if set(phases) == set(local_phases)
                else f"Fasa berbeda: GI ini {_phases_text(local_phases)}, {other['station']} {_phases_text(phases)}."
            )
        remote_path = remote.get("trip_path") or {}
        if remote_path:
            evidence.append(f"Jalur trip di sana: {remote_path['title']}.")
        echo = (remote_path.get("value") or {}).get("echo")
        rules = ["F7.3"]
        confidence = "medium"
        if echo:
            evidence.append(
                f"{other['station']} menerima sinyal {_ms(echo['receive_ms'])} ms lalu memantulkannya "
                f"{_ms(echo['send_ms'])} ms: pola echo weak infeed pada skema POTT."
            )
            rules.append("F5.4")
            if silent_send:
                # P4: a recorded send that never asserted, while the far end received.
                flags.append(_row(
                    "flag_send_silent", 9, "Ditandai", "GI lawan menerima sinyal, kanal Send di GI ini tidak aktif",
                    [f"{other['station']} menerima sinyal {_ms(echo['receive_ms'])} ms, tetapi {', '.join(silent_send)} "
                     "di rekaman GI ini tidak pernah aktif.",
                     "Kanal Send di DFR ini kemungkinan bukan dari relay yang mengirim, atau tidak terhubung — cek "
                     "pemetaan kanal teleproteksi."],
                    ["F5.4", "F7.3"], "flag",
                ))
        if other.get("reclose_outcome"):
            dead = other.get("reclose_dead_time_s")
            evidence.append(
                f"Reclose {'berhasil' if other['reclose_outcome'] == 'successful' else 'gagal'}"
                + (f" setelah dead time {_num(dead)} s." if isinstance(dead, (int, float)) else ".")
            )
        clock = other.get("clock") or {}
        if clock.get("method") == "fault_aligned":
            shift = []
            if clock.get("zone_offset_h"):
                hours = clock["zone_offset_h"]
                shift.append(f"{'+' if hours > 0 else '−'}{_num(abs(hours), 0 if float(hours).is_integer() else 2)} jam")
            if clock.get("clock_offset_ms") is not None and abs(clock["clock_offset_ms"]) >= 1:
                shift.append(f"{_ms(clock['clock_offset_ms']).replace(',0', '')} ms")
            evidence.append(
                "Jam perekam diselaraskan pada awal gangguan" + (f" (koreksi {', '.join(shift)})." if shift else ".")
            )
            confidence = "high" if phases and set(phases) == set(local_phases) else "medium"
        rows.append(_row("other_end", 7, "Ujung lain", title, evidence, rules, confidence,
                         {"station": other["station"], "record_id": other.get("fault_record_id")}))
    return rows, flags


def _cause_row(episode: FaultEpisode, fault: IncidentRecord, physical_cause: dict[str, Any],
               hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
    """F8: the AI reading, the sequence pattern, and what lightning needs."""
    entry = next(
        (e for e in physical_cause.get("records") or [] if e.get("incident_record_id") == fault.incident_record_id),
        None,
    )
    evidence: list[str] = []
    rules = ["F8.1"]
    title = "Bacaan AI tidak tersedia"
    if entry and entry.get("top_hypothesis"):
        cause = entry["top_hypothesis"]
        ranking = entry.get("cause_ranking") or []
        top = entry.get("confidence") or (ranking[0].get("confidence") if ranking else 0.0) or 0.0
        second = ranking[1].get("confidence") if len(ranking) > 1 else None
        if top < 0.5 and second is not None and top - second < 0.1:
            # Same threshold as the incident page: no reading stands out.
            title = "Tidak ada penyebab yang dominan (bacaan AI)"
            evidence.append("Kandidat: " + ", ".join(
                f"{_CAUSE_NAMES.get(c.get('cause'), c.get('cause'))} {_int((c.get('confidence') or 0.0) * 100)}%"
                for c in ranking[:3]
            ) + ".")
        else:
            title = f"{_CAUSE_NAMES.get(cause, cause)} {_int(top * 100)}% (bacaan AI)"
        evidence.append("Bacaan AI per rekaman; tidak mengubah fakta di atas.")
        if cause == "PETIR":
            rules.append("F8.3")
            evidence.append("Sub-mekanisme (SF/BFO) tidak dapat ditentukan tanpa data LDS.")
    patterns = [h for h in hypotheses if episode.episode_index in (h.get("episode_indices") or []) and h.get("hypothesis") in _PATTERN_TITLES]
    for pattern in patterns:
        rules.append("F8.2")
        evidence.append(f"Pola urutan: {_PATTERN_TITLES[pattern['hypothesis']]}.")
    return _row("cause", 8, "Penyebab", title, evidence, sorted(set(rules)), "ai",
                {"top_hypothesis": entry.get("top_hypothesis") if entry else None})


def build_episode_reasoning(
    episode: FaultEpisode,
    records_by_id: dict[str, IncidentRecord],
    physical_cause: dict[str, Any],
    hypotheses: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """The ledger for one fault: the fault record's conclusions, completed
    with the reclose, the other end and the cause. None when no member
    record carries a reasoning chain with a fault (older snapshots)."""
    fault = _fault_record(episode, records_by_id)
    if fault is None:
        return None
    chain = _reasoning(fault)
    rows = copy.deepcopy([row for row in chain.get("conclusions") or [] if row.get("confidence") != "flag"])
    flags = copy.deepcopy([row for row in chain.get("conclusions") or [] if row.get("confidence") == "flag"])

    rows = [_reclose_row(row, episode, records_by_id, fault) if row["key"] == "trip_reclose" else row for row in rows]
    other_rows, other_flags = _other_end_rows(episode, records_by_id, chain)
    rows += other_rows
    flags += other_flags
    rows.append(_cause_row(episode, fault, physical_cause, hypotheses))
    for item in episode.missing_evidence or []:
        if item.get("type") == "ENDS_DISAGREE_ON_FAULTED_PHASES":
            flags.append(_row("flag_ends_phases", 9, "Ditandai", "Kedua ujung membaca fasa berbeda",
                              [item["description"]], ["F4.8", "F7.3"], "flag"))
    rows += flags
    return {
        "fault_record_id": fault.incident_record_id,
        "fault_start_ms": chain.get("fault_start_ms"),
        "rows": rows,
        "flag_count": sum(1 for row in rows if row["confidence"] == "flag"),
        "conflict_count": sum(len(row.get("conflicts") or []) for row in rows),
    }
