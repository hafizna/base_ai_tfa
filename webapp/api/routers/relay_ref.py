"""REF / GFR / SBEF relay — base panels only, no additional analysis endpoint needed."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/analyze/ref", tags=["relay-ref"])


@router.get("/info")
async def relay_info():
    return {
        "relay_type": "REF/GFR/SBEF",
        "panels": ["analog_recap", "comtrade_explorer", "ctvt_ratio", "soe"],
        "note": "Extended analysis panels not yet available for REF/GFR/SBEF relay types.",
    }
