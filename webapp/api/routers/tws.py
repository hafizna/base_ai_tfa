"""TWS FL .cdb upload and retrieval endpoints."""

import asyncio
import logging
from functools import partial
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from core.tws_cdb_parser import TwsCdbParseError, parse_tws_cdb_bytes
from ..json_safety import replace_non_finite_numbers
from ..storage import load_analysis, save_analysis
from ..training_retention import RetainedUploadFile, retain_upload

router = APIRouter(prefix="/api/tws", tags=["tws-fl"])
logger = logging.getLogger("uvicorn")


class TwsUploadResponse(BaseModel):
    analysis_id: str
    source_file: str
    circuit_name: str
    line_length_km: float
    endpoint_count: int
    total_samples: int
    warnings: list[str]


@router.post("/upload-cdb", response_model=TwsUploadResponse)
async def upload_tws_cdb(cdb_file: UploadFile = File(...)):
    """Parse a Qualitrol/Cashel TWS FL .cdb export and create a session."""
    filename = Path(cdb_file.filename or "record.cdb").name
    if not filename.lower().endswith(".cdb"):
        raise HTTPException(status_code=422, detail="Please upload a TWS FL .cdb export file.")

    cdb_bytes = await cdb_file.read()
    loop = asyncio.get_event_loop()
    try:
        payload = await loop.run_in_executor(
            None,
            partial(parse_tws_cdb_bytes, cdb_bytes, filename),
        )
    except TwsCdbParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Could not parse the TWS FL .cdb export.") from exc

    payload = replace_non_finite_numbers(payload)
    analysis_id = save_analysis(payload)
    first_result = (payload.get("results") or [{}])[0]

    try:
        retain_upload(
            analysis_id=analysis_id,
            source_type="tws_cdb",
            files=[
                RetainedUploadFile(
                    field_name="cdb_file",
                    filename=filename,
                    content_type=cdb_file.content_type,
                    data=cdb_bytes,
                )
            ],
            metadata={
                "station_name": payload.get("station_name", ""),
                "rec_dev_id": payload.get("rec_dev_id", ""),
                "source_file": payload.get("source_file", filename),
                "circuit_name": first_result.get("circuit_name", ""),
                "line_length_km": float(first_result.get("line_length_km") or 0.0),
                "endpoint_count": len(first_result.get("endpoints") or []),
                "total_samples": int(payload.get("total_samples") or 0),
                "warnings": payload.get("warnings", []),
            },
        )
    except Exception as exc:
        logger.warning("Failed to retain raw TWS upload for analysis_id=%s: %s", analysis_id, exc)

    return TwsUploadResponse(
        analysis_id=analysis_id,
        source_file=payload.get("source_file", filename),
        circuit_name=first_result.get("circuit_name", ""),
        line_length_km=float(first_result.get("line_length_km") or 0.0),
        endpoint_count=len(first_result.get("endpoints") or []),
        total_samples=int(payload.get("total_samples") or 0),
        warnings=payload.get("warnings", []),
    )


@router.get("/{analysis_id}")
async def get_tws_analysis(analysis_id: str):
    payload = load_analysis(analysis_id)
    if payload is None or payload.get("source_type") != "tws_cdb":
        raise HTTPException(status_code=404, detail="TWS FL analysis session not found or expired.")
    return replace_non_finite_numbers(payload)
