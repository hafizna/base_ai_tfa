"""
COMTRADE Fault Analyser — FastAPI backend v2
============================================
Run from the repo root:
    uvicorn webapp.api.main:app --reload --port 8000
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .routers import upload, relay_21, relay_87l, relay_87t, relay_ocr, relay_ref
from .storage import get_session_ttl_hours, get_storage_backend

app = FastAPI(
    title="COMTRADE Fault Analyser",
    version="2.0.0",
    description="Relay-type-aware COMTRADE analysis platform for DFR UIT JBT",
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

app.include_router(upload.router)
app.include_router(relay_21.router)
app.include_router(relay_87l.router)
app.include_router(relay_87t.router)
app.include_router(relay_ocr.router)
app.include_router(relay_ref.router)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "analysis_storage": get_storage_backend(),
        "analysis_ttl_hours": get_session_ttl_hours(),
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
