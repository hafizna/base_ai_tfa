"""Analysis session storage with PostgreSQL persistence and file fallback."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import psycopg
except Exception:  # pragma: no cover - optional dependency in some local envs
    psycopg = None

ANALYSIS_DIR = Path(tempfile.gettempdir()) / "dfr_fastapi_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://") :]

SESSION_TTL_HOURS = max(1, int(os.getenv("ANALYSIS_SESSION_TTL_HOURS", "24")))

_DB_INITIALIZED = False


def _db_enabled() -> bool:
    return bool(DATABASE_URL and psycopg is not None)


def _db_connect():
    if not _db_enabled():
        raise RuntimeError("PostgreSQL storage is not enabled.")
    return psycopg.connect(DATABASE_URL, autocommit=True, connect_timeout=5)


def _analysis_path(analysis_id: str) -> Path:
    return ANALYSIS_DIR / f"{analysis_id}.json"


def _save_analysis_file(payload: dict[str, Any], analysis_id: str) -> str:
    _analysis_path(analysis_id).write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return analysis_id


def _load_analysis_file(analysis_id: str) -> dict[str, Any] | None:
    path = _analysis_path(analysis_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _delete_analysis_file(analysis_id: str) -> None:
    try:
        path = _analysis_path(analysis_id)
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _init_db() -> None:
    global _DB_INITIALIZED

    if _DB_INITIALIZED or not _db_enabled():
        return

    with _db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    analysis_id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMPTZ NOT NULL,
                    station_name TEXT NOT NULL DEFAULT '',
                    rec_dev_id TEXT NOT NULL DEFAULT '',
                    total_samples INTEGER NOT NULL DEFAULT 0,
                    payload_json JSONB NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_analysis_sessions_expires_at
                ON analysis_sessions (expires_at)
                """
            )
    _DB_INITIALIZED = True


def _cleanup_expired_sessions() -> None:
    if not _db_enabled():
        return

    _init_db()
    with _db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM analysis_sessions WHERE expires_at <= NOW()")


def get_storage_backend() -> str:
    return "postgres" if _db_enabled() else "filesystem"


def get_session_ttl_hours() -> int:
    return SESSION_TTL_HOURS


def save_analysis(payload: dict[str, Any]) -> str:
    analysis_id = uuid.uuid4().hex

    if not _db_enabled():
        return _save_analysis_file(payload, analysis_id)

    expires_at = datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS)
    _cleanup_expired_sessions()

    try:
        _init_db()
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO analysis_sessions (
                        analysis_id,
                        expires_at,
                        station_name,
                        rec_dev_id,
                        total_samples,
                        payload_json
                    ) VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        analysis_id,
                        expires_at,
                        str(payload.get("station_name", "")),
                        str(payload.get("rec_dev_id", "")),
                        int(payload.get("total_samples", 0) or 0),
                        json.dumps(payload, ensure_ascii=False),
                    ),
                )
        return analysis_id
    except Exception:
        return _save_analysis_file(payload, analysis_id)


def update_analysis(analysis_id: str, payload: dict[str, Any]) -> str:
    if not _db_enabled():
        return _save_analysis_file(payload, analysis_id)

    expires_at = datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS)

    try:
        _init_db()
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO analysis_sessions (
                        analysis_id,
                        expires_at,
                        station_name,
                        rec_dev_id,
                        total_samples,
                        payload_json
                    ) VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (analysis_id) DO UPDATE SET
                        expires_at = EXCLUDED.expires_at,
                        station_name = EXCLUDED.station_name,
                        rec_dev_id = EXCLUDED.rec_dev_id,
                        total_samples = EXCLUDED.total_samples,
                        payload_json = EXCLUDED.payload_json
                    """,
                    (
                        analysis_id,
                        expires_at,
                        str(payload.get("station_name", "")),
                        str(payload.get("rec_dev_id", "")),
                        int(payload.get("total_samples", 0) or 0),
                        json.dumps(payload, ensure_ascii=False),
                    ),
                )
        return analysis_id
    except Exception:
        return _save_analysis_file(payload, analysis_id)


def load_analysis(analysis_id: str) -> dict[str, Any] | None:
    if not _db_enabled():
        return _load_analysis_file(analysis_id)

    try:
        _init_db()
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT payload_json
                    FROM analysis_sessions
                    WHERE analysis_id = %s
                    AND expires_at > NOW()
                    """,
                    (analysis_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return _load_analysis_file(analysis_id)
                payload = row[0]
                if isinstance(payload, str):
                    return json.loads(payload)
                return payload
    except Exception:
        return _load_analysis_file(analysis_id)


def delete_analysis(analysis_id: str) -> None:
    if _db_enabled():
        try:
            _init_db()
            with _db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM analysis_sessions WHERE analysis_id = %s",
                        (analysis_id,),
                    )
        except Exception:
            pass

    _delete_analysis_file(analysis_id)
