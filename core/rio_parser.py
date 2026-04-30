from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TripCharacteristic:
    kind: str  # "circle" or "poly"
    start: Optional[tuple[float, float]] = None
    radius: Optional[float] = None
    points: list[tuple[float, float]] = field(default_factory=list)
    center: tuple[float, float] = (0.0, 0.0)


@dataclass
class Zone:
    name: str = ""
    time1: float = 0.0
    timem: float = 0.0
    phase: Optional[TripCharacteristic] = None
    earth: Optional[TripCharacteristic] = None
    category: str = "ZONE"
    order: int = 0


@dataclass
class ProtectionDevice:
    device: str = ""
    substation: str = ""
    feeder: str = ""
    lineangle: Optional[float] = None
    re_rl: Optional[tuple[float, float]] = None
    xe_xl: Optional[tuple[float, float]] = None
    zs: Optional[tuple[float, float]] = None
    zones: list[Zone] = field(default_factory=list)


def _parse_float_pair(text: str) -> tuple[float, float]:
    nums = [x.strip() for x in text.split(",")]
    if len(nums) < 2:
        raise ValueError(f"expected pair, got: {text!r}")
    return float(nums[0]), float(nums[1])


def _parse_optional_pair(text: str, key: str) -> Optional[tuple[float, float]]:
    m = re.search(rf"\b{re.escape(key)}\s+([-0-9.]+)\s*,\s*([-0-9.]+)", text, re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _parse_tripchar_polygon(block: str) -> Optional[TripCharacteristic]:
    points: list[tuple[float, float]] = []
    start = re.search(r"\bSTART\s+([-0-9.]+)\s*,\s*([-0-9.]+)", block, re.IGNORECASE)
    if start:
        points.append((float(start.group(1)), float(start.group(2))))
    for m in re.finditer(r"\bLINE\s+([-0-9.]+)\s*,\s*([-0-9.]+)", block, re.IGNORECASE):
        points.append((float(m.group(1)), float(m.group(2))))
    if len(points) < 3:
        return None
    return TripCharacteristic(kind="poly", start=points[0], points=points)


def _parse_tripchar_circle(block: str) -> Optional[TripCharacteristic]:
    start = re.search(r"\bSTART\s+([-0-9.]+)\s*,\s*([-0-9.]+)", block, re.IGNORECASE)
    arc = re.search(
        r"\bARC\s+([-0-9.]+)\s*,\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*,\s*(CW|CCW)",
        block,
        re.IGNORECASE,
    )
    if not arc:
        return None
    radius = float(arc.group(1))
    center_x = float(arc.group(2))
    sweep = float(arc.group(3))
    start_pt = None
    if start:
        start_pt = (float(start.group(1)), float(start.group(2)))
    if radius <= 0 or abs(sweep) < 359 or abs(center_x) > 1e-6:
        return None
    return TripCharacteristic(kind="circle", start=start_pt, radius=radius, center=(0.0, 0.0))


def parse_protection_device_rio_text(text: str) -> Optional[ProtectionDevice]:
    if not re.search(r"BEGIN\s+PROTECTIONDEVICE", text, re.IGNORECASE):
        return None

    pd = ProtectionDevice()
    for key, attr in (("DEVICE", "device"), ("SUBSTATION", "substation"), ("FEEDER", "feeder")):
        m = re.search(
            rf"\b{key}\s+(.+?)(?=\s+(?:SUBSTATION|FEEDER|RATING|MAX|TOL-T|TOL-Z|CURRGROUND|LINEANGLE|RE/RL|XE/XL|KS|ZS|TIME0MAX|IMPCORR|DIRCHAR|BEGIN)\b|$)",
            text,
            re.IGNORECASE,
        )
        if m:
            setattr(pd, attr, m.group(1).strip())

    lineangle = re.search(r"\bLINEANGLE\s+([-0-9.]+)", text, re.IGNORECASE)
    if lineangle:
        pd.lineangle = float(lineangle.group(1))
    pd.re_rl = _parse_optional_pair(text, "RE/RL")
    pd.xe_xl = _parse_optional_pair(text, "XE/XL")
    pd.zs = _parse_optional_pair(text, "ZS")

    block_re = re.compile(r"BEGIN\s+(ZONE(?:-OVERREACH)?)\b([\s\S]*?)END\s+\1", re.IGNORECASE)
    for order, m in enumerate(block_re.finditer(text)):
        category = m.group(1).upper()
        block = m.group(0)
        zone = Zone(category=category, order=order)

        name_m = re.search(
            r"\bNAME\s+(.+?)(?=\s+(?:TIME1|TIMEM|BEGIN|ACTIVE|INDEX|FAULTLOOP)\b|$)",
            block,
            re.IGNORECASE,
        )
        zone.name = name_m.group(1).strip() if name_m else f"{category} {order + 1}"

        time1_m = re.search(r"\bTIME1\s+([-0-9.]+)", block, re.IGNORECASE)
        timem_m = re.search(r"\bTIMEM\s+([-0-9.]+)", block, re.IGNORECASE)
        if time1_m:
            zone.time1 = float(time1_m.group(1))
        if timem_m:
            zone.timem = float(timem_m.group(1))

        trip_phase = re.search(r"BEGIN\s+TRIPCHAR([\s\S]*?)END\s+TRIPCHAR", block, re.IGNORECASE)
        trip_earth = re.search(r"BEGIN\s+TRIPCHAR-EARTH([\s\S]*?)END\s+TRIPCHAR-EARTH", block, re.IGNORECASE)
        if trip_phase:
            zone.phase = _parse_tripchar_circle(trip_phase.group(1))
        if trip_earth:
            zone.earth = _parse_tripchar_polygon(trip_earth.group(1))

        pd.zones.append(zone)

    return pd


def protection_device_to_relay_data(pd: ProtectionDevice) -> dict:
    ph_gnd: list[dict] = []
    ph_ph: list[dict] = []
    for idx, zone in enumerate(pd.zones):
        label = zone.name or f"Z{idx + 1}"
        if zone.earth and zone.earth.kind == "poly":
            ph_gnd.append(
                {
                    "index": idx + 1,
                    "order": zone.order,
                    "label": label,
                    "shapeType": "poly",
                    "poly": [{"r": round(r, 6), "x": round(x, 6)} for r, x in zone.earth.points],
                    "category": zone.category,
                    "time1": zone.time1,
                    "timem": zone.timem,
                }
            )
        if zone.phase and zone.phase.kind == "circle" and zone.phase.radius is not None:
            ph_ph.append(
                {
                    "index": idx + 1,
                    "order": zone.order,
                    "label": label,
                    "shapeType": "circle",
                    "centerR": round(zone.phase.center[0], 6),
                    "centerX": round(zone.phase.center[1], 6),
                    "radius": round(zone.phase.radius, 6),
                    "start": (
                        {"r": round(zone.phase.start[0], 6), "x": round(zone.phase.start[1], 6)}
                        if zone.phase.start
                        else None
                    ),
                    "category": zone.category,
                    "time1": zone.time1,
                    "timem": zone.timem,
                }
            )

    return {
        "kind": "rio",
        "device": {
            "device": pd.device,
            "substation": pd.substation,
            "feeder": pd.feeder,
            "lineangle": pd.lineangle,
            "re_rl": list(pd.re_rl) if pd.re_rl else None,
            "xe_xl": list(pd.xe_xl) if pd.xe_xl else None,
            "zs": list(pd.zs) if pd.zs else None,
        },
        "phGnd": ph_gnd,
        "phPh": ph_ph,
    }


def parse_rio_text_to_relay_data(text: str) -> Optional[dict]:
    pd = parse_protection_device_rio_text(text)
    if not pd:
        return None
    return protection_device_to_relay_data(pd)
