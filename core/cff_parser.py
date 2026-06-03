"""
ABB CFF parser
==============
Extracts ABB combined fault files (.cff) into their embedded COMTRADE
configuration/data parts, then delegates waveform parsing to comtrade_parser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import tempfile
from typing import Optional

from .comtrade_parser import ComtradeRecord, parse_comtrade


_CFF_MARKER_RE = re.compile(
    rb"(?m)^---\s*file type:\s*(?P<kind>[A-Za-z0-9_ ]+?)"
    rb"(?:\s*:\s*(?P<meta>[^-\r\n]+?))?\s*---\r?\n"
)


class CffParseError(ValueError):
    """Raised when a CFF archive does not contain usable COMTRADE parts."""


@dataclass
class CffPart:
    kind: str
    meta: str
    content: bytes


@dataclass
class CffArchive:
    cfg: bytes
    dat: Optional[bytes]
    dat_format: str = ""
    expected_dat_size: Optional[int] = None
    warnings: list[str] = field(default_factory=list)


def extract_cff(data: bytes) -> CffArchive:
    """Extract embedded COMTRADE CFG/DAT bytes from an ABB CFF payload."""
    markers = list(_CFF_MARKER_RE.finditer(data))
    if not markers:
        raise CffParseError("CFF marker not found.")

    parts: list[CffPart] = []
    for idx, marker in enumerate(markers):
        start = marker.end()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(data)
        parts.append(
            CffPart(
                kind=marker.group("kind").decode("ascii", errors="ignore").strip().upper(),
                meta=(marker.group("meta") or b"").decode("ascii", errors="ignore").strip(),
                content=data[start:end],
            )
        )

    cfg_part = next((part for part in parts if part.kind == "CFG"), None)
    if cfg_part is None or not cfg_part.content.strip():
        raise CffParseError("CFF archive does not contain a CFG section.")

    dat_part = next((part for part in parts if part.kind.startswith("DAT")), None)
    warnings: list[str] = []
    dat_format = ""
    expected_size: Optional[int] = None
    dat_bytes: Optional[bytes] = None

    if dat_part is None:
        warnings.append("CFF archive does not contain a DAT section - metadata only")
    else:
        dat_bytes = dat_part.content
        dat_format = dat_part.kind.replace("DAT", "", 1).strip()
        if dat_part.meta:
            try:
                expected_size = int(dat_part.meta)
            except ValueError:
                warnings.append(f"CFF DAT size metadata is not numeric: {dat_part.meta}")
            else:
                if expected_size != len(dat_bytes):
                    warnings.append(
                        f"CFF DAT section size mismatch: marker={expected_size} bytes, actual={len(dat_bytes)} bytes"
                    )

    return CffArchive(
        cfg=cfg_part.content,
        dat=dat_bytes,
        dat_format=dat_format,
        expected_dat_size=expected_size,
        warnings=warnings,
    )


def parse_cff_bytes(data: bytes, source_filename: str = "record.cff") -> Optional[ComtradeRecord]:
    """Parse ABB CFF bytes into the standard ComtradeRecord model."""
    archive = extract_cff(data)
    source_stem = Path(source_filename or "record.cff").stem or "record"

    with tempfile.TemporaryDirectory(prefix="cff_upload_") as tmp_dir:
        tmp = Path(tmp_dir)
        cfg_path = tmp / f"{source_stem}.cfg"
        dat_path = tmp / f"{source_stem}.dat"

        cfg_path.write_bytes(archive.cfg)
        if archive.dat is not None:
            dat_path.write_bytes(archive.dat)
            record = parse_comtrade(str(cfg_path), str(dat_path))
        else:
            record = parse_comtrade(str(cfg_path), None)

    if record is not None:
        record.cfg_path = source_filename
        record.dat_path = source_filename if archive.dat is not None else None
        record.warnings.extend(
            [
                "Loaded from ABB CFF archive",
                *archive.warnings,
            ]
        )
    return record


def parse_cff(cff_path: str | Path) -> Optional[ComtradeRecord]:
    """Parse an ABB CFF file from disk."""
    path = Path(cff_path)
    return parse_cff_bytes(path.read_bytes(), path.name)
