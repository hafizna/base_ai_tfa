"""
Path Heuristics
===============
Shared helpers for deriving discovery labels from raw_data path strings.

These helpers are intentionally lightweight and are used by the webapp browse
page and batch prediction output so transformer cases can be found even when
the file name itself is generic.
"""

from __future__ import annotations

import re

# Folder/path fragments used for line-cause discovery during browse/search.
_TRANSIENT_LABEL_MAP = [
    ("petir", "PETIR"),
    ("layang", "LAYANG-LAYANG"),
    ("pohon", "POHON"),
    ("tower roboh", "KONDUKTOR"),
    ("konduktor", "KONDUKTOR"),
    ("alat isolator", "PERALATAN"),
    ("isolator", "PERALATAN"),
    ("kerusakan peralatan", "PERALATAN"),
    ("gangguan peralatan", "PERALATAN"),
    ("pilot wire", "PERALATAN"),
    ("pilotwire", "PERALATAN"),
    ("teleprotection", "PERALATAN"),
    ("teleproteksi", "PERALATAN"),
    ("plcc", "PERALATAN"),
    ("hewan", "HEWAN"),
    ("ular", "HEWAN"),
    ("babi", "HEWAN"),
    ("kukang", "HEWAN"),
    ("monyet", "HEWAN"),
    ("benda asing", "BENDA ASING"),
    ("bfo", "BFO"),
    # Equipment / protection / telecom-origin cases
    ("kerusakan peralatan", "PERALATAN"),
    ("gangguan peralatan",  "PERALATAN"),
    ("pilot wire",          "PERALATAN"),
    ("pilotwire",           "PERALATAN"),
    ("teleprotection",      "PERALATAN"),
    ("teleproteksi",        "PERALATAN"),
    ("plcc",                "PERALATAN"),
    ("peralatan",           "PERALATAN"),
    ("lain", "LAIN-LAIN"),
]

# Transformer path hints. We keep this broader than a single file name because
# many records use generic CFG names but meaningful folder names like TRAFO/TRF.
_TRANSFORMER_PATH_RE = re.compile(
    r"(^|[\\/_.\-\s])(TRAFO|TRF|TRANSFORMER|XFMR)(?=$|[\\/_.\-\s#\d])",
    re.IGNORECASE,
)

_TRANSFORMER_CONFIRMED_RE = re.compile(
    r"(?:^|[\\/_.\-\s])(87T|PDIF|TRANSFORMER\s+DIFF|RELE\s+DIFFERENSIAL|DIFF(?:REF)?|DIFFERENTIAL)(?=$|[\\/_.\-\s#\d])",
    re.IGNORECASE,
)

_TRANSFORMER_OCR_RE = re.compile(r"\bOCR\b", re.IGNORECASE)


def infer_transient_label(path_str: str) -> str:
    """Return a transient-cause label inferred from a path string, or empty."""
    low = (path_str or "").lower()
    for fragment, label in _TRANSIENT_LABEL_MAP:
        if fragment in low:
            return label
    return ""


def is_transformer_path(path_str: str) -> bool:
    """Return True when the path looks like a transformer case/folder."""
    return bool(_TRANSFORMER_PATH_RE.search(path_str or ""))


def infer_path_tag(path_str: str) -> str:
    """
    Return a display tag for discovery.

    Priority:
    - transient label if the path contains one
    - otherwise transformer-specific tag when the path hints at a transformer case
    - otherwise empty string
    """
    label = infer_transient_label(path_str)
    if label:
        return label
    if is_transformer_path(path_str):
        low = path_str or ""
        if _TRANSFORMER_CONFIRMED_RE.search(low):
            return "87T CONFIRMED"
        if _TRANSFORMER_OCR_RE.search(low):
            return "OCR ONLY"
        return "TRAFO CANDIDATE"
    return ""


def infer_path_kind(path_str: str) -> str:
    """Return TRANSIENT, TRANSFORMER, or empty string for unknown paths."""
    label = infer_transient_label(path_str)
    if label:
        return "TRANSIENT"
    if is_transformer_path(path_str):
        return "TRANSFORMER"
    return ""


def infer_status_data(path_str: str) -> str:
    """Return a coarse status group for discovery / rekap views."""
    kind = infer_path_kind(path_str)
    return kind or "UNKNOWN"


def infer_suspected_label(path_str: str) -> str:
    """
    Return a human-readable label prefixed as a suspected / not-fixed label.

    This intentionally stays descriptive rather than authoritative.
    """
    tag = infer_path_tag(path_str)
    if tag:
        return f"DIDUGA {tag}"
    return "DIDUGA UNKNOWN"
