"""
COMTRADE Fault Analyser — FastAPI backend v2
============================================
Run from the repo root:
    uvicorn webapp.api.main:app --reload --port 8000
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from . import ml_predict
from .routers import upload, relay_21, relay_87l, relay_87t, relay_ocr, relay_ref, tws, report
from .storage import get_session_ttl_hours, get_storage_backend

logger = logging.getLogger("uvicorn")

# Cache so /api/health can report the warmup result without re-running it.
_WARMUP_STATE: dict = {"ran": False}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Eagerly load the fault-classifier model + sklearn/LightGBM imports.

    Without this, the very first request pays a 200–500 ms one-time cost for
    pickle deserialisation and lightgbm/sklearn module import. With workers > 1
    each worker pays the cost once at boot instead — but still amortised.
    """
    t0 = time.perf_counter()
    status = ml_predict.warmup()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _WARMUP_STATE.update(status)
    _WARMUP_STATE["ran"] = True
    _WARMUP_STATE["elapsed_ms"] = round(elapsed_ms, 1)
    logger.info(
        "ml_predict.warmup completed in %.1f ms — model_loaded=%s version=%s calibration=%s",
        elapsed_ms, status.get("model_loaded"), status.get("model_version"), status.get("calibration"),
    )
    yield
    # No teardown work required.


app = FastAPI(
    title="COMTRADE Fault Analyser",
    version="2.1.0",
    description="Relay-type-aware COMTRADE analysis platform for DFR UIT JBT",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev
        "http://localhost:4173",   # Vite preview
        "http://localhost:8000",   # Same-origin (prod)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compress large responses (waveform JSON is multi-MB at the
# /api/analysis/{id} and /api/recalculate-ratio endpoints — decimal-float
# text compresses 5–8×). Small responses bypass the middleware so health
# checks and tiny metadata payloads don't pay the CPU cost.
app.add_middleware(GZipMiddleware, minimum_size=10_000, compresslevel=6)

app.include_router(upload.router)
app.include_router(relay_21.router)
app.include_router(relay_87l.router)
app.include_router(relay_87t.router)
app.include_router(relay_ocr.router)
app.include_router(relay_ref.router)
app.include_router(tws.router)
app.include_router(report.router)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "analysis_storage": get_storage_backend(),
        "analysis_ttl_hours": get_session_ttl_hours(),
        "warmup": _WARMUP_STATE,
    }


# Serve React production build — static assets + SPA catch-all
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    # Mount /assets so hashed JS/CSS bundles are served directly
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # Serve specific files (favicon, manifest, etc.) if they exist
        candidate = frontend_dist / full_path
        if candidate.is_file():
            return FileResponse(str(candidate))
        # All other paths → index.html (React Router handles routing)
        return FileResponse(str(frontend_dist / "index.html"))
