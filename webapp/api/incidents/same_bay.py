"""Same-bay assessment — Stage 2.

Before reconstruction runs, assess whether the incident's attached records
plausibly describe the same bay. This is deliberately lenient: vendor
metadata is often sparse or inconsistent, so the assessment only downgrades
to ``MISMATCH_REQUIRES_REVIEW`` on an explicit, strong conflict (different
non-empty station names with no operator override), never merely because
metadata is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .models import IncidentRecord, SAME_BAY_STATUSES  # noqa: F401 (re-export for callers)
from .time_axis import TimeAxis, same_station, station_of


def _far_end_stations(records: list[IncidentRecord], axis: Optional[TimeAxis], home: Optional[str]) -> set[str]:
    """Stations at the other end of the line: another substation whose
    recorder the time axis lined up with this end's on a fault both recorded
    (one fault starts at one instant everywhere on the line)."""
    if axis is None:
        return set()
    if not home:
        reference = next((r for r in records if axis.lane(r) == axis.reference_group), None)
        home = station_of(reference) if reference is not None else None
    if not home:
        return set()
    far = set()
    for record in records:
        placement = axis.placement(record)
        station = record.station_name or station_of(record)
        if placement is not None and placement.method == "fault_aligned" and station and not same_station(station, home):
            far.add(station)
    return far


@dataclass
class SameBayAssessment:
    status: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    override: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "evidence": self.evidence, "override": self.override}


def assess_same_bay(
    incident_station_name: Optional[str],
    incident_bay_name: Optional[str],
    records: list[IncidentRecord],
    *,
    override_reason: Optional[str] = None,
    override_operator: Optional[str] = None,
    override_at_iso: Optional[str] = None,
    axis: Optional[TimeAxis] = None,
) -> SameBayAssessment:
    """Assess whether ``records`` plausibly belong to the same bay.

    Evidence sources considered (per spec section 2): explicit incident bay
    name, incident station metadata, per-record station name, relay/device
    ID, protection family, voltage level. Channel-name bay tags and file
    metadata are considered when available on the record but Stage 2 does not
    require them. Station names match regardless of case, spacing and a "GI"
    prefix. Records from the other end of the line — another substation
    whose clock ``axis`` lined up on a shared fault — are not a conflict.
    """
    evidence: list[dict[str, Any]] = []
    override: Optional[dict[str, Any]] = None

    if override_reason or override_operator:
        override = {
            "operator": override_operator,
            "reason": override_reason,
            "at_iso": override_at_iso,
        }
        evidence.append({
            "type": "OPERATOR_OVERRIDE",
            "description": f"Operator override applied: {override_reason or '(no reason given)'}",
        })
        return SameBayAssessment(status="CONFIRMED_SAME_BAY", evidence=evidence, override=override)

    if len(records) <= 1:
        evidence.append({"type": "SINGLE_RECORD", "description": "Only one record attached; same-bay comparison not applicable."})
        return SameBayAssessment(status="UNKNOWN", evidence=evidence, override=None)

    far_end = _far_end_stations(records, axis, incident_station_name)
    if far_end:
        evidence.append({
            "type": "OTHER_LINE_END",
            "description": (
                f"Records from {sorted(far_end)} were lined up with this end's on a fault both recorded — "
                "the other end of the line, not another bay."
            ),
        })
    near = [r for r in records if not (r.station_name and r.station_name in far_end)]
    station_names = {r.station_name for r in near if r.station_name}
    relay_ids = {r.relay_id for r in near if r.relay_id}
    protection_types = {r.protection_type for r in records if r.protection_type}
    bay_names = {r.bay_name for r in near if r.bay_name}

    def distinct(names: set[str]) -> list[str]:
        kept: list[str] = []
        for name in sorted(names):
            if not any(same_station(name, other) for other in kept):
                kept.append(name)
        return kept

    # Strong conflict: two or more distinct, non-empty station names and no
    # incident-level station to arbitrate, or records disagree with the
    # incident's own station metadata.
    conflicting_stations = False
    if incident_station_name:
        mismatched = {s for s in station_names if not same_station(s, incident_station_name)}
        if mismatched:
            conflicting_stations = True
            evidence.append({
                "type": "STATION_NAME_MISMATCH",
                "description": f"Incident station '{incident_station_name}' does not match record station(s) {sorted(mismatched)}.",
                "requires_review": True,
            })
    elif len(distinct(station_names)) > 1:
        conflicting_stations = True
        evidence.append({
            "type": "STATION_NAME_MISMATCH",
            "description": f"Records report different station names: {sorted(station_names)}.",
            "requires_review": True,
        })

    if bay_names:
        if len(bay_names) > 1 and not incident_bay_name:
            evidence.append({
                "type": "BAY_NAME_MISMATCH",
                "description": f"Records report different bay names: {sorted(bay_names)}.",
                "requires_review": True,
            })
            conflicting_stations = True  # bay-level conflict is equally strong
        elif incident_bay_name and bay_names - {incident_bay_name}:
            evidence.append({
                "type": "BAY_NAME_MISMATCH",
                "description": f"Incident bay '{incident_bay_name}' does not match record bay(s) {sorted(bay_names - {incident_bay_name})}.",
                "requires_review": True,
            })
            conflicting_stations = True

    if conflicting_stations:
        return SameBayAssessment(status="MISMATCH_REQUIRES_REVIEW", evidence=evidence, override=None)

    if incident_bay_name:
        evidence.append({"type": "EXPLICIT_BAY_NAME", "description": f"Incident bay name '{incident_bay_name}' provided by user."})
    if incident_station_name and len(distinct(station_names)) <= 1:
        evidence.append({"type": "STATION_NAME_CONSISTENT", "description": f"All records agree with incident station '{incident_station_name}'."})
    elif len(distinct(station_names)) == 1:
        evidence.append({"type": "STATION_NAME_CONSISTENT", "description": f"All records report the same station '{next(iter(station_names))}'."})

    if len(relay_ids) == 1:
        evidence.append({"type": "SAME_RELAY_ID", "description": f"All records report the same relay/device id '{next(iter(relay_ids))}'."})
    elif len(relay_ids) > 1:
        evidence.append({"type": "DIFFERENT_RELAY_IDS", "description": f"Records come from different relay/device ids: {sorted(relay_ids)} — consistent with same-bay multi-relay capture (e.g. distance + differential)."})

    if len(protection_types) > 1:
        evidence.append({"type": "MIXED_PROTECTION_FAMILY", "description": f"Records use different protection types: {sorted(protection_types)} — consistent with multiple relays on one bay."})

    if not station_names and not bay_names and not relay_ids:
        evidence.append({
            "type": "INSUFFICIENT_METADATA",
            "description": "No station, bay, or relay metadata available on any record; same-bay grouping is based on manual attachment only.",
        })
        return SameBayAssessment(status="UNKNOWN", evidence=evidence, override=None)

    # Strong signal (explicit bay name or consistent station+relay) -> confirmed.
    if incident_bay_name or (len(distinct(station_names)) <= 1 and len(relay_ids) <= 1 and station_names):
        return SameBayAssessment(status="CONFIRMED_SAME_BAY", evidence=evidence, override=None)

    return SameBayAssessment(status="LIKELY_SAME_BAY", evidence=evidence, override=None)
