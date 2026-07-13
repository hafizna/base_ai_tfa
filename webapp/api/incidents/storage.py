"""Incident storage — Stage 1.

Persists incidents, incident records, evidence, and incident feedback
separately from the single-record analysis session storage
(``webapp.api.storage``). Follows the same PostgreSQL-with-filesystem-fallback
pattern: PostgreSQL when ``DATABASE_URL`` is configured, otherwise JSON files
under a local directory so incident data survives process restarts in
development too (unlike the analysis-session store, incidents are not
meant to expire on a TTL).

Never stores waveform samples — only ``analysis_id`` references, metadata,
and the canonical-analysis snapshot captured at attach time. Raw COMTRADE
retention continues to follow Stage 0's ``training_retention`` /
``webapp.api.storage`` paths.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

try:
    import psycopg
except Exception:  # pragma: no cover - optional dependency in some local envs
    psycopg = None

from .models import (
    FaultEpisode,
    Incident,
    IncidentEvidence,
    IncidentFeedback,
    IncidentRecord,
    IncidentTimelineEvent,
    Reconstruction,
    RecordRelationship,
)

INCIDENTS_DIR = Path(os.getenv("INCIDENTS_DATA_DIR", "") or (Path(tempfile.gettempdir()) / "dfr_fastapi_incidents"))
INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
(INCIDENTS_DIR / "incidents").mkdir(parents=True, exist_ok=True)
(INCIDENTS_DIR / "records").mkdir(parents=True, exist_ok=True)
(INCIDENTS_DIR / "evidence").mkdir(parents=True, exist_ok=True)
(INCIDENTS_DIR / "feedback").mkdir(parents=True, exist_ok=True)
(INCIDENTS_DIR / "timeline_events").mkdir(parents=True, exist_ok=True)
(INCIDENTS_DIR / "relationships").mkdir(parents=True, exist_ok=True)
(INCIDENTS_DIR / "reconstructions").mkdir(parents=True, exist_ok=True)
(INCIDENTS_DIR / "episodes").mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://") :]

_DB_INITIALIZED = False


def _db_enabled() -> bool:
    return bool(DATABASE_URL and psycopg is not None)


def _db_connect():
    if not _db_enabled():
        raise RuntimeError("PostgreSQL storage is not enabled.")
    return psycopg.connect(DATABASE_URL, autocommit=True, connect_timeout=5)


def get_storage_backend() -> str:
    return "postgres" if _db_enabled() else "filesystem"


def _init_db() -> None:
    global _DB_INITIALIZED
    if _DB_INITIALIZED or not _db_enabled():
        return

    with _db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    incident_id TEXT PRIMARY KEY,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    status TEXT NOT NULL DEFAULT 'DRAFT',
                    station_name TEXT NOT NULL DEFAULT '',
                    payload_json JSONB NOT NULL,
                    deleted BOOLEAN NOT NULL DEFAULT FALSE
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS incident_records (
                    incident_record_id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    analysis_id TEXT NOT NULL,
                    payload_json JSONB NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS incident_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    payload_json JSONB NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS incident_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    payload_json JSONB NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS incident_timeline_events (
                    timeline_event_id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    payload_json JSONB NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS incident_relationships (
                    relationship_id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    payload_json JSONB NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS incident_reconstructions (
                    reconstruction_id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    payload_json JSONB NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS incident_episodes (
                    episode_id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    payload_json JSONB NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_incident_records_incident_id ON incident_records (incident_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_incident_evidence_incident_id ON incident_evidence (incident_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_incident_feedback_incident_id ON incident_feedback (incident_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_incident_timeline_events_incident_id ON incident_timeline_events (incident_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_incident_relationships_incident_id ON incident_relationships (incident_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_incident_reconstructions_incident_id ON incident_reconstructions (incident_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_incident_episodes_incident_id ON incident_episodes (incident_id)")
    _DB_INITIALIZED = True


# --- Filesystem helpers ------------------------------------------------------

def _path(kind: str, item_id: str) -> Path:
    return INCIDENTS_DIR / kind / f"{item_id}.json"


def _write_json(kind: str, item_id: str, data: dict[str, Any]) -> None:
    _path(kind, item_id).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _read_json(kind: str, item_id: str) -> Optional[dict[str, Any]]:
    path = _path(kind, item_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _delete_json(kind: str, item_id: str) -> None:
    try:
        path = _path(kind, item_id)
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _list_json(kind: str) -> list[dict[str, Any]]:
    out = []
    dir_path = INCIDENTS_DIR / kind
    if not dir_path.exists():
        return out
    for path in sorted(dir_path.glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


# --- Incident CRUD ------------------------------------------------------------

def create_incident(incident: Incident) -> Incident:
    data = incident.to_dict()
    if not _db_enabled():
        _write_json("incidents", incident.incident_id, data)
        return incident

    try:
        _init_db()
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO incidents (incident_id, status, station_name, payload_json)
                    VALUES (%s, %s, %s, %s::jsonb)
                    """,
                    (incident.incident_id, incident.status, incident.station_name or "", json.dumps(data, ensure_ascii=False)),
                )
        return incident
    except Exception:
        _write_json("incidents", incident.incident_id, data)
        return incident


def update_incident(incident: Incident) -> Incident:
    data = incident.to_dict()
    if not _db_enabled():
        _write_json("incidents", incident.incident_id, data)
        return incident

    try:
        _init_db()
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO incidents (incident_id, status, station_name, payload_json)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (incident_id) DO UPDATE SET
                        updated_at = NOW(),
                        status = EXCLUDED.status,
                        station_name = EXCLUDED.station_name,
                        payload_json = EXCLUDED.payload_json,
                        deleted = FALSE
                    """,
                    (incident.incident_id, incident.status, incident.station_name or "", json.dumps(data, ensure_ascii=False)),
                )
        return incident
    except Exception:
        _write_json("incidents", incident.incident_id, data)
        return incident


def get_incident(incident_id: str) -> Optional[Incident]:
    if not _db_enabled():
        data = _read_json("incidents", incident_id)
        return Incident.from_dict(data) if data else None

    try:
        _init_db()
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload_json FROM incidents WHERE incident_id = %s AND deleted = FALSE",
                    (incident_id,),
                )
                row = cur.fetchone()
                if row is None:
                    data = _read_json("incidents", incident_id)
                    return Incident.from_dict(data) if data else None
                payload = row[0]
                if isinstance(payload, str):
                    payload = json.loads(payload)
                return Incident.from_dict(payload)
    except Exception:
        data = _read_json("incidents", incident_id)
        return Incident.from_dict(data) if data else None


def list_incidents() -> list[Incident]:
    if not _db_enabled():
        return [Incident.from_dict(d) for d in _list_json("incidents")]

    try:
        _init_db()
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload_json FROM incidents WHERE deleted = FALSE ORDER BY updated_at DESC")
                rows = cur.fetchall()
                out = []
                for (payload,) in rows:
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    out.append(Incident.from_dict(payload))
                return out
    except Exception:
        return [Incident.from_dict(d) for d in _list_json("incidents")]


def delete_incident(incident_id: str) -> None:
    """Archive rather than hard-delete: sets status ARCHIVED. Callers that
    truly want removal should use ``purge_incident``."""
    incident = get_incident(incident_id)
    if incident is None:
        return
    incident.status = "ARCHIVED"
    update_incident(incident)


def purge_incident(incident_id: str) -> None:
    """Hard-delete an incident and its child records/evidence/feedback."""
    for rec in list_incident_records(incident_id):
        _delete_json("records", rec.incident_record_id)
    for ev in list_incident_evidence(incident_id):
        _delete_json("evidence", ev.evidence_id)
    for fb in list_incident_feedback(incident_id):
        _delete_json("feedback", fb.feedback_id)

    if _db_enabled():
        try:
            _init_db()
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM incident_records WHERE incident_id = %s", (incident_id,))
                    cur.execute("DELETE FROM incident_evidence WHERE incident_id = %s", (incident_id,))
                    cur.execute("DELETE FROM incident_feedback WHERE incident_id = %s", (incident_id,))
                    cur.execute("DELETE FROM incidents WHERE incident_id = %s", (incident_id,))
        except Exception:
            pass
    _delete_json("incidents", incident_id)


# --- Incident records ---------------------------------------------------------

def create_incident_record(record: IncidentRecord) -> IncidentRecord:
    data = record.to_dict()
    if not _db_enabled():
        _write_json("records", record.incident_record_id, data)
        return record

    try:
        _init_db()
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO incident_records (incident_record_id, incident_id, analysis_id, payload_json)
                    VALUES (%s, %s, %s, %s::jsonb)
                    """,
                    (record.incident_record_id, record.incident_id, record.analysis_id, json.dumps(data, ensure_ascii=False)),
                )
        return record
    except Exception:
        _write_json("records", record.incident_record_id, data)
        return record


def update_incident_record(record: IncidentRecord) -> IncidentRecord:
    data = record.to_dict()
    if not _db_enabled():
        _write_json("records", record.incident_record_id, data)
        return record

    try:
        _init_db()
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO incident_records (incident_record_id, incident_id, analysis_id, payload_json)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (incident_record_id) DO UPDATE SET
                        payload_json = EXCLUDED.payload_json
                    """,
                    (record.incident_record_id, record.incident_id, record.analysis_id, json.dumps(data, ensure_ascii=False)),
                )
        return record
    except Exception:
        _write_json("records", record.incident_record_id, data)
        return record


def get_incident_record(incident_record_id: str) -> Optional[IncidentRecord]:
    if not _db_enabled():
        data = _read_json("records", incident_record_id)
        return IncidentRecord.from_dict(data) if data else None

    try:
        _init_db()
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload_json FROM incident_records WHERE incident_record_id = %s",
                    (incident_record_id,),
                )
                row = cur.fetchone()
                if row is None:
                    data = _read_json("records", incident_record_id)
                    return IncidentRecord.from_dict(data) if data else None
                payload = row[0]
                if isinstance(payload, str):
                    payload = json.loads(payload)
                return IncidentRecord.from_dict(payload)
    except Exception:
        data = _read_json("records", incident_record_id)
        return IncidentRecord.from_dict(data) if data else None


def list_incident_records(incident_id: str) -> list[IncidentRecord]:
    if not _db_enabled():
        records = [IncidentRecord.from_dict(d) for d in _list_json("records") if d.get("incident_id") == incident_id]
    else:
        try:
            _init_db()
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT payload_json FROM incident_records WHERE incident_id = %s",
                        (incident_id,),
                    )
                    rows = cur.fetchall()
                    records = []
                    for (payload,) in rows:
                        if isinstance(payload, str):
                            payload = json.loads(payload)
                        records.append(IncidentRecord.from_dict(payload))
        except Exception:
            records = [IncidentRecord.from_dict(d) for d in _list_json("records") if d.get("incident_id") == incident_id]

    def sort_key(r: IncidentRecord):
        if r.manual_order is not None:
            return (0, r.manual_order)
        return (1, r.sequence_index)

    return sorted(records, key=sort_key)


def delete_incident_record(incident_record_id: str) -> None:
    if _db_enabled():
        try:
            _init_db()
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM incident_records WHERE incident_record_id = %s", (incident_record_id,))
        except Exception:
            pass
    _delete_json("records", incident_record_id)


# --- Evidence ------------------------------------------------------------------

def create_incident_evidence(evidence: IncidentEvidence) -> IncidentEvidence:
    data = evidence.to_dict()
    if not _db_enabled():
        _write_json("evidence", evidence.evidence_id, data)
        return evidence

    try:
        _init_db()
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO incident_evidence (evidence_id, incident_id, payload_json)
                    VALUES (%s, %s, %s::jsonb)
                    """,
                    (evidence.evidence_id, evidence.incident_id, json.dumps(data, ensure_ascii=False)),
                )
        return evidence
    except Exception:
        _write_json("evidence", evidence.evidence_id, data)
        return evidence


def list_incident_evidence(incident_id: str) -> list[IncidentEvidence]:
    if not _db_enabled():
        items = [IncidentEvidence.from_dict(d) for d in _list_json("evidence") if d.get("incident_id") == incident_id]
    else:
        try:
            _init_db()
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT payload_json FROM incident_evidence WHERE incident_id = %s",
                        (incident_id,),
                    )
                    rows = cur.fetchall()
                    items = []
                    for (payload,) in rows:
                        if isinstance(payload, str):
                            payload = json.loads(payload)
                        items.append(IncidentEvidence.from_dict(payload))
        except Exception:
            items = [IncidentEvidence.from_dict(d) for d in _list_json("evidence") if d.get("incident_id") == incident_id]
    return sorted(items, key=lambda e: e.created_at)


def delete_incident_evidence(evidence_id: str) -> None:
    if _db_enabled():
        try:
            _init_db()
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM incident_evidence WHERE evidence_id = %s", (evidence_id,))
        except Exception:
            pass
    _delete_json("evidence", evidence_id)


# --- Feedback --------------------------------------------------------------

def create_incident_feedback(feedback: IncidentFeedback) -> IncidentFeedback:
    data = feedback.to_dict()
    if not _db_enabled():
        _write_json("feedback", feedback.feedback_id, data)
        return feedback

    try:
        _init_db()
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO incident_feedback (feedback_id, incident_id, payload_json)
                    VALUES (%s, %s, %s::jsonb)
                    """,
                    (feedback.feedback_id, feedback.incident_id, json.dumps(data, ensure_ascii=False)),
                )
        return feedback
    except Exception:
        _write_json("feedback", feedback.feedback_id, data)
        return feedback


def list_incident_feedback(incident_id: str) -> list[IncidentFeedback]:
    if not _db_enabled():
        items = [IncidentFeedback.from_dict(d) for d in _list_json("feedback") if d.get("incident_id") == incident_id]
    else:
        try:
            _init_db()
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT payload_json FROM incident_feedback WHERE incident_id = %s",
                        (incident_id,),
                    )
                    rows = cur.fetchall()
                    items = []
                    for (payload,) in rows:
                        if isinstance(payload, str):
                            payload = json.loads(payload)
                        items.append(IncidentFeedback.from_dict(payload))
        except Exception:
            items = [IncidentFeedback.from_dict(d) for d in _list_json("feedback") if d.get("incident_id") == incident_id]
    return sorted(items, key=lambda f: f.created_at)


def new_id() -> str:
    return uuid.uuid4().hex


# --- Stage 2: timeline events --------------------------------------------------

def replace_timeline_events(incident_id: str, events: list[IncidentTimelineEvent]) -> None:
    """Reconstruction is deterministic and re-run wholesale, so the previous
    version's timeline events for this incident are replaced rather than
    accumulated. The Reconstruction record itself is what's versioned/kept
    for audit — see ``create_reconstruction``."""
    for existing in list_timeline_events(incident_id):
        _delete_json("timeline_events", existing.timeline_event_id)
    if _db_enabled():
        try:
            _init_db()
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM incident_timeline_events WHERE incident_id = %s", (incident_id,))
        except Exception:
            pass

    for event in events:
        data = event.to_dict()
        if not _db_enabled():
            _write_json("timeline_events", event.timeline_event_id, data)
            continue
        try:
            _init_db()
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO incident_timeline_events (timeline_event_id, incident_id, payload_json)
                        VALUES (%s, %s, %s::jsonb)
                        """,
                        (event.timeline_event_id, incident_id, json.dumps(data, ensure_ascii=False)),
                    )
        except Exception:
            _write_json("timeline_events", event.timeline_event_id, data)


def list_timeline_events(incident_id: str) -> list[IncidentTimelineEvent]:
    if not _db_enabled():
        items = [IncidentTimelineEvent.from_dict(d) for d in _list_json("timeline_events") if d.get("incident_id") == incident_id]
    else:
        try:
            _init_db()
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT payload_json FROM incident_timeline_events WHERE incident_id = %s",
                        (incident_id,),
                    )
                    rows = cur.fetchall()
                    items = []
                    for (payload,) in rows:
                        if isinstance(payload, str):
                            payload = json.loads(payload)
                        items.append(IncidentTimelineEvent.from_dict(payload))
        except Exception:
            items = [IncidentTimelineEvent.from_dict(d) for d in _list_json("timeline_events") if d.get("incident_id") == incident_id]

    def sort_key(e: IncidentTimelineEvent):
        return (e.relative_incident_ms if e.relative_incident_ms is not None else float("inf"), e.absolute_time_iso or "")

    return sorted(items, key=sort_key)


# --- Stage 2: relationships ------------------------------------------------------

def replace_relationships(incident_id: str, relationships: list[RecordRelationship]) -> None:
    for existing in list_relationships(incident_id):
        _delete_json("relationships", existing.relationship_id)
    if _db_enabled():
        try:
            _init_db()
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM incident_relationships WHERE incident_id = %s", (incident_id,))
        except Exception:
            pass

    for rel in relationships:
        _persist_relationship(rel)


def _persist_relationship(rel: RecordRelationship) -> RecordRelationship:
    data = rel.to_dict()
    if not _db_enabled():
        _write_json("relationships", rel.relationship_id, data)
        return rel
    try:
        _init_db()
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO incident_relationships (relationship_id, incident_id, payload_json)
                    VALUES (%s, %s, %s::jsonb)
                    ON CONFLICT (relationship_id) DO UPDATE SET payload_json = EXCLUDED.payload_json
                    """,
                    (rel.relationship_id, rel.incident_id, json.dumps(data, ensure_ascii=False)),
                )
        return rel
    except Exception:
        _write_json("relationships", rel.relationship_id, data)
        return rel


def update_relationship(rel: RecordRelationship) -> RecordRelationship:
    return _persist_relationship(rel)


def get_relationship(relationship_id: str) -> Optional[RecordRelationship]:
    if not _db_enabled():
        data = _read_json("relationships", relationship_id)
        return RecordRelationship.from_dict(data) if data else None
    try:
        _init_db()
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload_json FROM incident_relationships WHERE relationship_id = %s",
                    (relationship_id,),
                )
                row = cur.fetchone()
                if row is None:
                    data = _read_json("relationships", relationship_id)
                    return RecordRelationship.from_dict(data) if data else None
                payload = row[0]
                if isinstance(payload, str):
                    payload = json.loads(payload)
                return RecordRelationship.from_dict(payload)
    except Exception:
        data = _read_json("relationships", relationship_id)
        return RecordRelationship.from_dict(data) if data else None


def list_relationships(incident_id: str) -> list[RecordRelationship]:
    if not _db_enabled():
        return [RecordRelationship.from_dict(d) for d in _list_json("relationships") if d.get("incident_id") == incident_id]
    try:
        _init_db()
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload_json FROM incident_relationships WHERE incident_id = %s",
                    (incident_id,),
                )
                rows = cur.fetchall()
                items = []
                for (payload,) in rows:
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    items.append(RecordRelationship.from_dict(payload))
                return items
    except Exception:
        return [RecordRelationship.from_dict(d) for d in _list_json("relationships") if d.get("incident_id") == incident_id]


# --- Stage 2: reconstruction versioning ------------------------------------------

def create_reconstruction(reconstruction: Reconstruction) -> Reconstruction:
    """Mark all prior reconstructions for this incident as non-latest, then
    persist the new one. History is never deleted."""
    for existing in list_reconstructions(reconstruction.incident_id):
        if existing.is_latest:
            existing.is_latest = False
            _persist_reconstruction(existing)

    return _persist_reconstruction(reconstruction)


def _persist_reconstruction(reconstruction: Reconstruction) -> Reconstruction:
    data = reconstruction.to_dict()
    if not _db_enabled():
        _write_json("reconstructions", reconstruction.reconstruction_id, data)
        return reconstruction
    try:
        _init_db()
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO incident_reconstructions (reconstruction_id, incident_id, payload_json)
                    VALUES (%s, %s, %s::jsonb)
                    ON CONFLICT (reconstruction_id) DO UPDATE SET payload_json = EXCLUDED.payload_json
                    """,
                    (reconstruction.reconstruction_id, reconstruction.incident_id, json.dumps(data, ensure_ascii=False)),
                )
        return reconstruction
    except Exception:
        _write_json("reconstructions", reconstruction.reconstruction_id, data)
        return reconstruction


def list_reconstructions(incident_id: str) -> list[Reconstruction]:
    if not _db_enabled():
        items = [Reconstruction.from_dict(d) for d in _list_json("reconstructions") if d.get("incident_id") == incident_id]
    else:
        try:
            _init_db()
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT payload_json FROM incident_reconstructions WHERE incident_id = %s",
                        (incident_id,),
                    )
                    rows = cur.fetchall()
                    items = []
                    for (payload,) in rows:
                        if isinstance(payload, str):
                            payload = json.loads(payload)
                        items.append(Reconstruction.from_dict(payload))
        except Exception:
            items = [Reconstruction.from_dict(d) for d in _list_json("reconstructions") if d.get("incident_id") == incident_id]
    return sorted(items, key=lambda r: r.created_at)


def get_latest_reconstruction(incident_id: str) -> Optional[Reconstruction]:
    versions = list_reconstructions(incident_id)
    latest = [r for r in versions if r.is_latest]
    if latest:
        return latest[-1]
    return versions[-1] if versions else None


def replace_episodes(incident_id: str, episodes: list[FaultEpisode]) -> None:
    for existing in list_episodes(incident_id):
        _delete_json("episodes", existing.episode_id)
    if _db_enabled():
        try:
            _init_db()
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM incident_episodes WHERE incident_id = %s", (incident_id,))
        except Exception:
            pass

    for episode in episodes:
        data = episode.to_dict()
        if not _db_enabled():
            _write_json("episodes", episode.episode_id, data)
            continue
        try:
            _init_db()
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO incident_episodes (episode_id, incident_id, payload_json)
                        VALUES (%s, %s, %s::jsonb)
                        """,
                        (episode.episode_id, incident_id, json.dumps(data, ensure_ascii=False)),
                    )
        except Exception:
            _write_json("episodes", episode.episode_id, data)


def list_episodes(incident_id: str) -> list[FaultEpisode]:
    if not _db_enabled():
        items = [FaultEpisode.from_dict(d) for d in _list_json("episodes") if d.get("incident_id") == incident_id]
    else:
        try:
            _init_db()
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT payload_json FROM incident_episodes WHERE incident_id = %s",
                        (incident_id,),
                    )
                    rows = cur.fetchall()
                    items = []
                    for (payload,) in rows:
                        if isinstance(payload, str):
                            payload = json.loads(payload)
                        items.append(FaultEpisode.from_dict(payload))
        except Exception:
            items = [FaultEpisode.from_dict(d) for d in _list_json("episodes") if d.get("incident_id") == incident_id]
    return sorted(items, key=lambda e: e.episode_index)


def get_reconstruction(reconstruction_id: str) -> Optional[Reconstruction]:
    if not _db_enabled():
        data = _read_json("reconstructions", reconstruction_id)
        return Reconstruction.from_dict(data) if data else None
    try:
        _init_db()
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload_json FROM incident_reconstructions WHERE reconstruction_id = %s",
                    (reconstruction_id,),
                )
                row = cur.fetchone()
                if row is None:
                    data = _read_json("reconstructions", reconstruction_id)
                    return Reconstruction.from_dict(data) if data else None
                payload = row[0]
                if isinstance(payload, str):
                    payload = json.loads(payload)
                return Reconstruction.from_dict(payload)
    except Exception:
        data = _read_json("reconstructions", reconstruction_id)
        return Reconstruction.from_dict(data) if data else None
