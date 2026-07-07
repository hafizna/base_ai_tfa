"""Bridge between COMTRADE session JSON and the existing fault_classifier.pkl model.

Extracts the same 17-feature vector that models/train.py uses, then runs the
LightGBM multi-class classifier with the same calibration and confidence caps
as models/predict.py.

Pipeline order
--------------
1. extract_ml_features      → 17 analog + digital sequence features
2. _augment_row_with_soe_context  → inject SOE-derived loop hints
3. Tier 1 (models/rules.apply_rules) — deterministic structural rules
   (KONDUKTOR / GANGGUAN PERMANEN / CT anomaly / SOE mismatch)
4. Tier 2 LightGBM predict_proba
5. Probability calibration (fitted calibrator pickle if present, else T=1.5)
6. Confidence caps (transient ambiguity, equipment caution, PETIR digital
   caution) — each cap is recorded in `applied_caps`
7. Structured evidence (text + severity + weight) for richer UI rendering

The response also exposes introspection fields: raw + calibrated
probabilities, the actual feature vector used, applied caps, and model
metadata (version, training profile, file SHA-256).
"""

import hashlib
import pickle
import re
import sys
from pathlib import Path
from typing import Optional, Any

import numpy as np

_PIPELINE_DIR = Path(__file__).parent.parent.parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from models.predict import _petir_subtype_description, _augment_row_with_soe_context  # noqa: E402
from models.rules import apply_rules  # noqa: E402
from core.current_anomaly import detect_ct_measurement_anomaly  # noqa: E402
from .fault_detection import detect_fault_presence, _is_operate_status  # noqa: E402

_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "fault_classifier.pkl"
_CALIBRATOR_PATH = Path(__file__).parent.parent.parent / "models" / "proba_calibrator.pkl"

# Cached so we hash the model file + load the pickle once per process.
# Set ``_MODEL_BUNDLE_CACHE``/``_CALIBRATOR_CACHE`` to ``None`` to invalidate
# (mostly used by tests). Each cache uses a sentinel-empty dict to mean
# "already tried, nothing available" — distinct from ``None`` ("not tried").
_MODEL_BUNDLE_CACHE: Optional[dict] = None
_MODEL_META_CACHE: Optional[dict] = None
_CALIBRATOR_CACHE: Optional[dict] = None
_FEATURE_VERSION = "v1.2025-04"  # bump when feature schema changes incompatibly

_LABEL_DISPLAY = {
    "PETIR":       "Petir / Lightning",
    "LAYANG":      "Layang-Layang / Kite",
    "POHON":       "Pohon / Vegetasi",
    "HEWAN":       "Hewan / Binatang",
    "BENDA_ASING": "Benda Asing",
    "KONDUKTOR":   "Konduktor / Tower",
    "PERALATAN":   "Peralatan Rusak / Anomali Proteksi",
}

_TRANSIENT = {"PETIR", "LAYANG", "HEWAN", "BENDA_ASING"}


def _load_model() -> Optional[dict]:
    """Return the model bundle, loading it from disk at most once per process.

    The bundle is cached in module scope so subsequent calls are a dict lookup
    rather than a pickle deserialise. Call ``warmup()`` from the FastAPI
    startup lifespan to pay the load cost before the first request.
    """
    global _MODEL_BUNDLE_CACHE
    if _MODEL_BUNDLE_CACHE is not None:
        return _MODEL_BUNDLE_CACHE or None
    if not _MODEL_PATH.exists():
        _MODEL_BUNDLE_CACHE = {}
        return None
    try:
        with open(_MODEL_PATH, "rb") as f:
            _MODEL_BUNDLE_CACHE = pickle.load(f)
    except Exception:
        _MODEL_BUNDLE_CACHE = {}
        return None
    return _MODEL_BUNDLE_CACHE


def warmup() -> dict:
    """Eagerly load model, calibrator, metadata, and Tier-2 helper modules.

    Called from the FastAPI startup lifespan so the very first request does
    not pay a 200–500 ms pickle-deserialise + import cost. Returns a small
    status dict for logging.
    """
    bundle = _load_model()
    _ = _load_calibrator()
    meta = _model_metadata(bundle)
    # Force the deferred ``models.predict`` / ``models.train`` imports too —
    # these pull in LightGBM and sklearn which dominate cold-start time.
    deferred_loaded = False
    try:
        from models.train import FEATURE_COLS  # noqa: F401
        from models.predict import (  # noqa: F401
            _build_feature_vector,
            _apply_transient_ambiguity_confidence_cap,
            _apply_equipment_caution_cap,
        )
        deferred_loaded = True
    except Exception:
        pass
    return {
        "model_loaded": bool(bundle),
        "model_version": meta.get("model_version"),
        "calibration": meta.get("calibration", {}).get("method"),
        "deferred_imports_loaded": deferred_loaded,
    }


def _load_calibrator() -> Optional[dict]:
    """Optional probability calibrator fitted on a held-out validation split.

    Produced by ``python models/calibrate.py``. When present, the bundle
    contains ``{calibrator, classes_, method, fitted_at_utc, n_samples}`` and
    is preferred over the default temperature scaling.
    """
    global _CALIBRATOR_CACHE
    if _CALIBRATOR_CACHE is not None:
        return _CALIBRATOR_CACHE or None
    if not _CALIBRATOR_PATH.exists():
        _CALIBRATOR_CACHE = {}
        return None
    try:
        with open(_CALIBRATOR_PATH, "rb") as f:
            _CALIBRATOR_CACHE = pickle.load(f)
    except Exception:
        _CALIBRATOR_CACHE = {}
        return None
    return _CALIBRATOR_CACHE


def _model_metadata(bundle: Optional[dict]) -> dict:
    """Return small metadata dict for response introspection.

    Cached per-process. ``model_version`` is derived from training timestamp +
    file SHA-256 prefix so any retrain produces a new value automatically.
    """
    global _MODEL_META_CACHE
    if _MODEL_META_CACHE is not None:
        return _MODEL_META_CACHE

    meta: dict = {
        "feature_version": _FEATURE_VERSION,
        "model_present": bundle is not None,
        "model_path": str(_MODEL_PATH.name),
    }
    if not _MODEL_PATH.exists():
        _MODEL_META_CACHE = meta
        return meta

    try:
        h = hashlib.sha256(_MODEL_PATH.read_bytes()).hexdigest()[:12]
        meta["model_sha256_prefix"] = h
    except Exception:
        meta["model_sha256_prefix"] = "unknown"

    if isinstance(bundle, dict):
        profile = bundle.get("training_profile") or {}
        trained_at = profile.get("trained_at_utc") or "unknown"
        meta["model_trained_at_utc"] = trained_at
        meta["model_type"] = bundle.get("model_type", "unknown")
        meta["model_version"] = f"{trained_at[:10] if trained_at != 'unknown' else 'untrained'}+{meta.get('model_sha256_prefix', '????????')}"
        meta["feature_cols"] = list(bundle.get("feature_cols") or [])
        meta["classes"] = list(bundle.get("classes") or bundle.get("all_classes") or [])
        # Cast numpy.int64 → int so FastAPI / JSON serializer accepts the response.
        meta["class_counts"] = {str(k): int(v) for k, v in (bundle.get("class_counts") or {}).items()}
        meta["training_profile"] = dict(profile) if profile else {}

    calibrator_bundle = _load_calibrator()
    if calibrator_bundle:
        meta["calibration"] = {
            "method": calibrator_bundle.get("method", "platt"),
            "fitted_at_utc": calibrator_bundle.get("fitted_at_utc", "unknown"),
            "held_out_samples": calibrator_bundle.get("n_samples"),
        }
    else:
        meta["calibration"] = {"method": "temperature_T1.5", "note": "no fitted calibrator found"}

    _MODEL_META_CACHE = meta
    return meta


def _apply_calibrator(
    proba: np.ndarray,
    classes: list,
    feature_row: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Prefer fitted calibrator if available, else temperature scaling.

    Returns (calibrated_proba, method_name). The fitted calibrator is a
    ``CalibratedClassifierCV(cv='prefit')`` produced by ``models/calibrate.py``.
    """
    bundle = _load_calibrator()
    if bundle and bundle.get("calibrator") is not None:
        try:
            cal = bundle["calibrator"]
            cal_classes = list(getattr(cal, "classes_", classes))
            cal_proba = cal.predict_proba(feature_row)[0]
            # Re-align to the order the caller expects (in case classes_ differs).
            class_to_idx = {c: i for i, c in enumerate(cal_classes)}
            aligned = np.zeros(len(classes), dtype=float)
            for i, c in enumerate(classes):
                j = class_to_idx.get(c)
                if j is not None:
                    aligned[i] = float(cal_proba[j])
            if aligned.sum() > 0:
                aligned /= aligned.sum()
                return aligned, bundle.get("method", "platt")
        except Exception:
            pass

    # Fallback: temperature scaling
    p = np.asarray(proba, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    logits = np.log(p) / 1.5
    logits -= logits.max()
    exp = np.exp(logits)
    return exp / exp.sum(), "temperature_T1.5"


# Severity / weight buckets for structured evidence.
_SEVERITY_WEIGHT = {
    "verdict": 1.0,
    "critical": 0.9,
    "warning": 0.7,
    "notable": 0.5,
    "info": 0.3,
}


def _ev(text: str, severity: str = "info", weight: Optional[float] = None, kind: str = "narrative") -> dict:
    """Build a structured evidence item.

    Args:
        text: human-readable Indonesian sentence (UI renders this verbatim).
        severity: one of info | notable | warning | critical | verdict.
        weight: 0..1 importance; defaults to the bucket value if omitted.
        kind: tag describing the source (narrative | rule | cap | model | physics).
    """
    sev = severity if severity in _SEVERITY_WEIGHT else "info"
    return {
        "text": text,
        "severity": sev,
        "weight": float(weight) if weight is not None else _SEVERITY_WEIGHT[sev],
        "kind": kind,
    }


def _find_ch(channels: list, candidates: list[str]) -> Optional[np.ndarray]:
    for ch in channels:
        if ch.get("canonical_name") in candidates or ch.get("name", "").upper() in candidates:
            return np.array(ch["samples"], dtype=float)
    return None


def _rms_window(arr: np.ndarray, start: int, n: int) -> float:
    seg = arr[start: start + n]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) > 0 else 0.0


def _thd_percent(arr: np.ndarray, sr: float, freq: float) -> float:
    """Estimate THD% of a one-cycle window using FFT."""
    cycle = max(4, int(sr / freq))
    if len(arr) < cycle:
        return 0.0
    seg = arr[:cycle]
    spectrum = np.abs(np.fft.rfft(seg, n=cycle))
    if spectrum[1] < 1e-9:
        return 0.0
    harmonics = np.sqrt(np.sum(spectrum[2:] ** 2))
    return float(harmonics / spectrum[1] * 100.0)


def _fundamental_phasor(seg: np.ndarray) -> complex:
    """Fundamental (1st-harmonic) phasor of a one-cycle window via DFT.

    The window must be exactly one fundamental cycle. Returns a complex phasor
    whose magnitude is the peak amplitude of the fundamental.
    """
    n = len(seg)
    if n < 2:
        return 0j
    k = np.arange(n)
    # Project onto e^{-j 2π k / n}: real part = cos component, imag = sin component.
    coeff = np.exp(-2j * np.pi * k / n)
    return complex(2.0 / n * np.dot(seg.astype(float), coeff))


def _symmetrical_components(ia: np.ndarray, ib: np.ndarray, ic: np.ndarray):
    """Return (|I0|, |I1|, |I2|) sequence magnitudes from one-cycle windows.

    Fortescue requires PHASORS, not raw time samples: each phase is first
    reduced to its fundamental phasor (1-cycle DFT), then the symmetrical
    transform is applied. Feeding raw sinusoids straight into the a-operator
    matrix yields I1≈I2 for any balanced set (the old bug that made every
    record look 100% unbalanced).
    """
    a = np.exp(1j * 2 * np.pi / 3)
    pa = _fundamental_phasor(np.asarray(ia).real)
    pb = _fundamental_phasor(np.asarray(ib).real)
    pc = _fundamental_phasor(np.asarray(ic).real)
    i0 = (pa + pb + pc) / 3.0
    i1 = (pa + a * pb + (a ** 2) * pc) / 3.0
    i2 = (pa + (a ** 2) * pb + a * pc) / 3.0
    return abs(i0), abs(i1), abs(i2)


def _phase_from_status_name(name: str) -> Optional[str]:
    upper = name.upper()
    line_phase_patterns = {
        "A": [r"\bL1\b", r"\bPH\s*1\b", r"\bPH-?L1\b"],
        "B": [r"\bL2\b", r"\bPH\s*2\b", r"\bPH-?L2\b"],
        "C": [r"\bL3\b", r"\bPH\s*3\b", r"\bPH-?L3\b"],
    }
    rst_phase_patterns = {
        "A": [r"\bPH\.?\s*R\b", r"\bPH-?R\b", r"(?:^|[._:\-\s])R(?:$|[._:\-\s])"],
        "B": [r"\bPH\.?\s*S\b", r"\bPH-?S\b", r"(?:^|[._:\-\s])S(?:$|[._:\-\s])"],
        "C": [r"\bPH\.?\s*T\b", r"\bPH-?T\b", r"(?:^|[._:\-\s])T(?:$|[._:\-\s])"],
    }
    for phase, patterns in line_phase_patterns.items():
        if any(re.search(pattern, upper) for pattern in patterns):
            return phase
    for phase, patterns in rst_phase_patterns.items():
        if any(re.search(pattern, upper) for pattern in patterns):
            return phase
    for phase in "ABC":
        patterns = [
            rf"(?:^|[._:\-\s]){phase}(?:$|[._:\-\s])",
            rf"\bPH\s*{phase}\b",
            rf"\bPH-{phase}\b",
            rf"\bPH{phase}\b",
            rf"\bPHASE\s+{phase}\b",
            rf"\b{phase}\s+PHASE\b",
            rf"\b{phase}\s*-\s*PH\b",
            rf"\b{phase}\s+PH\b",
        ]
        if any(re.search(pattern, upper) for pattern in patterns):
            return phase
    return None


def _first_on_ms(samples: list, time: np.ndarray, start_idx: int = 0) -> Optional[float]:
    if len(time) == 0:
        return None
    n = min(len(samples), len(time))
    start = max(0, min(start_idx, n - 1))
    prev = int(samples[start - 1]) if start > 0 else 0
    for idx in range(start, n):
        val = int(samples[idx])
        if val == 1 and prev == 0:
            return float(time[idx] * 1000)
        prev = val
    if start == 0 and n and int(samples[0]) == 1:
        return float(time[0] * 1000)
    return None


def _first_off_ms_after(samples: list, time: np.ndarray, after_idx: int) -> Optional[float]:
    """First 1→0 transition strictly after `after_idx` (in ms)."""
    if len(time) == 0:
        return None
    n = min(len(samples), len(time))
    start = max(0, min(after_idx + 1, n))
    prev = int(samples[start - 1]) if start > 0 else 0
    for idx in range(start, n):
        val = int(samples[idx])
        if val == 0 and prev == 1:
            return float(time[idx] * 1000)
        prev = val
    return None


def _first_edge_ms_after(samples: list, time: np.ndarray, after_idx: int, from_state: int, to_state: int) -> Optional[float]:
    if len(time) == 0:
        return None
    n = min(len(samples), len(time))
    start = max(0, min(after_idx + 1, n))
    prev = int(samples[start - 1]) if start > 0 else 0
    for idx in range(start, n):
        val = int(samples[idx])
        if prev == from_state and val == to_state:
            return float(time[idx] * 1000)
        prev = val
    return None


def _first_stable_edge_ms_after(
    samples: list,
    time: np.ndarray,
    after_idx: int,
    from_state: int,
    to_state: int,
    min_stable_ms: float = 20.0,
) -> Optional[float]:
    """First edge that remains in the target state long enough to reject contact bounce."""
    if len(time) == 0:
        return None
    n = min(len(samples), len(time))
    start = max(0, min(after_idx + 1, n))
    prev = int(samples[start - 1]) if start > 0 else 0
    for idx in range(start, n):
        val = int(samples[idx])
        if prev == from_state and val == to_state:
            edge_ms = float(time[idx] * 1000)
            stable_until_ms = edge_ms + min_stable_ms
            stable = True
            reached_window = False
            for probe in range(idx, n):
                if int(samples[probe]) != to_state:
                    stable = False
                    break
                if float(time[probe] * 1000) >= stable_until_ms:
                    reached_window = True
                    break
            if stable and reached_window:
                return edge_ms
        prev = val
    return None


def _index_for_ms(time: np.ndarray, ms: float) -> int:
    if len(time) == 0:
        return 0
    target = ms / 1000.0
    idx = int(np.searchsorted(time, target))
    return max(0, min(idx, len(time) - 1))


def _status_any_on(samples: list) -> bool:
    return any(int(v) == 1 for v in samples or [])


def _is_ar_status_name(name: str) -> bool:
    compact = re.sub(r"[^A-Z0-9]+", "", name.upper())
    return (
        bool(re.search(r"\bA\s*/?\s*R\b", name.upper()))
        or "RECLOS" in compact
        or "RECLOSE" in compact
        or compact.startswith("AR")
        or "AR1P" in compact
        or "AR3P" in compact
    )


def _digital_sequence_features(status_channels: list, time: np.ndarray, inception_idx: int) -> dict:
    start_idx = max(0, inception_idx - 2)
    trip_phases: dict[str, float] = {}
    cb_open_phases: dict[str, float] = {}
    cb_close_phases: dict[str, float] = {}
    cb_contact_open_phases: dict[str, float] = {}
    cb_contact_close_phases: dict[str, float] = {}
    cb_contact_breakers: set[str] = set()
    explicit_success_seen = False
    ar_attempt_seen = False
    startup_phases: dict[str, float] = {}
    fault_phases: dict[str, float] = {}
    zone_times: dict[str, float] = {}
    ar_flags = {
        "lockout": False,
        "not_ready": False,
        "block": False,
        "successful": False,
        "failed": False,
    }

    for sch in status_channels:
        raw_name = str(sch.get("name", "") or "")
        name = raw_name.upper()
        samples = sch.get("samples") or []
        if not samples or not _status_any_on(samples):
            continue
        first_ms = _first_on_ms(samples, time, start_idx)
        phase = _phase_from_status_name(raw_name)

        is_trip = _is_operate_status(raw_name)
        compact_name = name.replace(" ", "")
        is_cb_open = (
            "CB OPEN" in name
            or "PMT OPEN" in name
            or "BREAKER OPEN" in name
            or "1POLEOPEN" in compact_name
            or "1 POLE OPEN" in name
            or ("52" in name and "OPEN" in name)
        )
        breaker_match = re.search(r"\b(CB\d*)\b", name)
        is_aux_cb = (
            "AUXCB" in compact_name
            or "CBAUX" in compact_name
            or re.search(r"\bAUX\s*CB\b", name) is not None
            or re.search(r"\bCB\s*AUX\b", name) is not None
        )
        breaker_id = breaker_match.group(1) if breaker_match is not None else ("AUXCB" if is_aux_cb else None)
        is_cb_aux_open_contact = (
            is_aux_cb
            and phase is not None
            and not any(block in name for block in ("HEALTH", "HEALTHY", "ALARM", "FAIL", "LOCK", "BLOCK"))
            and not any(close in name for close in ("CLOSE", "CLOSED", "52A", "CONT", "CONTACT"))
        )
        is_cb_open = is_cb_open or is_cb_aux_open_contact
        is_cb_closed_contact = (
            breaker_id is not None
            and ("CONT" in name or "CONTACT" in name or re.search(r"\b52A\b", name) is not None)
            and not any(block in name for block in ("TRIP", "ALARM", "FAIL", "LOCK", "BLOCK"))
        )
        is_startup = "STARTUP" in name or "START UP" in name or "PICKUP" in name or "PICK UP" in name
        is_fault = "FAULT" in name and phase is not None

        if phase and first_ms is not None:
            if is_trip:
                trip_phases[phase] = min(first_ms, trip_phases.get(phase, first_ms))
            if is_cb_open:
                ar_attempt_seen = True
                cb_open_phases[phase] = min(first_ms, cb_open_phases.get(phase, first_ms))
                # Detect when this same CB-open channel returns stably to 0 = CB reclosed.
                # Breaker auxiliary contacts often bounce for a few milliseconds; those
                # flickers must not become AR-success evidence or fake dead-time values.
                rise_idx = _index_for_ms(time, first_ms)
                close_ms = _first_stable_edge_ms_after(samples, time, rise_idx, 1, 0)
                if close_ms is not None:
                    cb_close_phases[phase] = min(close_ms, cb_close_phases.get(phase, close_ms))
            if is_cb_closed_contact:
                # Closed-contact channels (e.g. CB1.CONT.A): 1=CB closed,
                # 1->0=open/trip, 0->1=reclose. This is common in 1.5 breaker bays
                # where no explicit AR/CB-open bit is recorded.
                open_ms = _first_stable_edge_ms_after(samples, time, start_idx - 1, 1, 0)
                if open_ms is not None:
                    cb_contact_breakers.add(breaker_id)
                    cb_open_phases[phase] = min(open_ms, cb_open_phases.get(phase, open_ms))
                    cb_contact_open_phases[phase] = min(open_ms, cb_contact_open_phases.get(phase, open_ms))
                    open_idx = _index_for_ms(time, open_ms)
                    close_ms = _first_stable_edge_ms_after(samples, time, open_idx, 0, 1)
                    if close_ms is not None:
                        cb_close_phases[phase] = min(close_ms, cb_close_phases.get(phase, close_ms))
                        cb_contact_close_phases[phase] = min(close_ms, cb_contact_close_phases.get(phase, close_ms))
            if is_startup:
                startup_phases[phase] = min(first_ms, startup_phases.get(phase, first_ms))
            if is_fault:
                fault_phases[phase] = min(first_ms, fault_phases.get(phase, first_ms))

        if is_trip and "ZONE" in name and first_ms is not None:
            for zone in ("1", "2", "3", "4", "5"):
                if f"ZONE{zone}" in name.replace(" ", "") or f"Z{zone}" in name:
                    zone_times[f"Z{zone}"] = min(first_ms, zone_times.get(f"Z{zone}", first_ms))

        if _is_ar_status_name(name):
            if first_ms is not None:
                ar_attempt_seen = True
            if "LOCKOUT" in name or "LOCK OUT" in name:
                ar_flags["lockout"] = True
                ar_flags["failed"] = True
            if "NOT READY" in name or "NOTREADY" in name or "BLOCK" in name:
                ar_flags["not_ready" if "READY" in name else "block"] = True
                ar_flags["failed"] = True
            # Bare "CLOSE" / "RECLOS" matches AR command/output pulses that fire even
            # when the breaker fails to physically reclose (e.g. broken isolator). Only
            # treat very specific verified-success tokens as positive evidence here;
            # otherwise rely on the CB-OPEN falling edge captured in cb_close_phases.
            has_success_token = (
                "SUCCESS" in name
                or "AR OK" in name or "AR_OK" in name
                or "RECLOSE OK" in name or "RECLOSE_OK" in name
                or "REC OK" in name or "REC_OK" in name
            )
            if has_success_token and not (
                "LOCK" in name or "NOT READY" in name or "NOTREADY" in name or "BLOCK" in name
            ):
                explicit_success_seen = True

    def sorted_phases(data: dict[str, float]) -> list[str]:
        return sorted(data.keys(), key=lambda ph: data[ph])

    trip_type = None
    if len(cb_open_phases) >= 3 or len(trip_phases) >= 3:
        trip_type = "three_pole"
    elif len(cb_open_phases) == 1 or len(trip_phases) == 1:
        trip_type = "single_pole"

    # AR is only marked successful when we have hard evidence the breaker actually
    # reclosed: CB-OPEN channel returning to 0, or an explicit "*_OK / SUCCESS" status.
    if not ar_flags["failed"] and (cb_close_phases or explicit_success_seen):
        ar_flags["successful"] = True

    ar_status = None
    if ar_flags["failed"]:
        ar_status = False
    elif ar_flags["successful"]:
        ar_status = True

    first_trip_candidates = list(trip_phases.values()) + list(zone_times.values())
    first_trip_ms = min(first_trip_candidates) if first_trip_candidates else None
    if first_trip_ms is not None:
        startup_phases = {
            phase: ms for phase, ms in startup_phases.items()
            if ms <= first_trip_ms + 5.0
        }
    first_startup_ms = min(startup_phases.values()) if startup_phases else None

    # AR dead time: from first CB-open rising edge to first CB-open falling edge (CB reclosed).
    # Without a verified falling edge we don't fabricate a value — better unknown than wrong.
    first_cb_open_ms = min(cb_open_phases.values()) if cb_open_phases else None
    first_cb_close_ms = min(cb_close_phases.values()) if cb_close_phases else None
    ar_dead_time_ms = None
    if first_cb_open_ms is not None and first_cb_close_ms is not None and first_cb_close_ms > first_cb_open_ms:
        ar_dead_time_ms = first_cb_close_ms - first_cb_open_ms

    if len(cb_close_phases) >= 3:
        reclose_mode = "three_pole"
    elif len(cb_close_phases) == 1:
        reclose_mode = "single_pole"
    else:
        reclose_mode = None

    topology_hint = "one_and_half_breaker" if len(cb_contact_breakers) >= 2 else None

    return {
        "digital_trip_type": trip_type,
        "digital_trip_phases": sorted_phases(trip_phases),
        "digital_cb_open_phases": sorted_phases(cb_open_phases),
        "digital_cb_close_phases": sorted_phases(cb_close_phases),
        "digital_startup_phases": sorted_phases(startup_phases),
        "digital_fault_phases": sorted_phases(fault_phases),
        "digital_zone": min(zone_times, key=zone_times.get) if zone_times else "",
        "digital_ar_status": ar_status,
        "digital_ar_attempted": ar_attempt_seen,
        "digital_ar_lockout": ar_flags["lockout"],
        "digital_ar_not_ready": ar_flags["not_ready"],
        "digital_first_startup_ms": round(first_startup_ms, 2) if first_startup_ms is not None else None,
        "digital_first_trip_ms": round(first_trip_ms, 2) if first_trip_ms is not None else None,
        "digital_startup_to_trip_ms": round(first_trip_ms - first_startup_ms, 2)
        if first_startup_ms is not None and first_trip_ms is not None else None,
        "digital_first_cb_open_ms": round(first_cb_open_ms, 2) if first_cb_open_ms is not None else None,
        "digital_first_cb_close_ms": round(first_cb_close_ms, 2) if first_cb_close_ms is not None else None,
        "digital_ar_dead_time_ms": round(ar_dead_time_ms, 1) if ar_dead_time_ms is not None else None,
        "digital_reclose_mode": reclose_mode,
        "digital_topology_hint": topology_hint,
        "digital_cb_contact_breakers": sorted(cb_contact_breakers),
        "digital_cb_contact_open_phases": sorted_phases(cb_contact_open_phases),
        "digital_cb_contact_close_phases": sorted_phases(cb_contact_close_phases),
    }


def extract_ml_features(payload: dict, relay_type: str = "21") -> dict:
    """Build the 17-feature dict from a stored COMTRADE session payload."""
    channels = payload.get("analog_channels", [])
    time = np.array(payload.get("time", []), dtype=float)
    freq = float(payload.get("frequency", 50.0))
    status_channels = payload.get("status_channels", [])

    if len(time) < 4:
        return _empty_features()

    sr = 1.0 / (time[1] - time[0])
    cycle_n = max(4, int(sr / freq))

    # Current channels
    ia = _find_ch(channels, ["IA", "IL1", "I1"])
    ib = _find_ch(channels, ["IB", "IL2", "I2"])
    ic = _find_ch(channels, ["IC", "IL3", "I3"])
    # Voltage channels
    va = _find_ch(channels, ["VA", "VAN", "UA"])
    vb = _find_ch(channels, ["VB", "VBN", "UB"])
    vc = _find_ch(channels, ["VC", "VCN", "UC"])

    phase_currents = [(ia, "A"), (ib, "B"), (ic, "C")]
    scored_currents = []
    for arr, phase in phase_currents:
        if arr is None or len(arr) < 4:
            continue
        pre_n = min(2 * cycle_n, len(arr) // 4)
        pre = float(np.sqrt(np.mean(arr[:pre_n] ** 2))) if pre_n > 1 else 0.0
        peak = float(np.max(np.abs(arr)))
        scored_currents.append((peak / max(pre, 1.0), peak, arr, phase))

    if not scored_currents:
        return _empty_features()

    _, _, i_primary, primary_phase = max(scored_currents, key=lambda item: (item[0], item[1]))
    v_primary = {"A": va, "B": vb, "C": vc}.get(primary_phase)
    if v_primary is None:
        v_primary = va if va is not None else (vb if vb is not None else vc)

    # --- Fault inception detection (same logic as relay_21._extract_features_from_payload) ---
    pre_end = min(2 * cycle_n, len(i_primary) // 4)
    pre_rms = float(np.sqrt(np.mean(i_primary[:pre_end] ** 2))) if pre_end > 1 else 0.0
    threshold = max(pre_rms * 2.0, np.max(np.abs(i_primary)) * 0.3, 0.05)
    inception_idx = next(
        (k for k in range(pre_end, len(i_primary)) if abs(i_primary[k]) > threshold),
        int(np.argmax(np.abs(i_primary))),
    )
    extinction_idx = len(i_primary) - 1
    for k in range(inception_idx + cycle_n, len(i_primary)):
        s = max(0, k - cycle_n // 2)
        if float(np.sqrt(np.mean(i_primary[s: k + 1] ** 2))) < threshold * 0.6:
            extinction_idx = k
            break

    fault_duration_ms = float((time[extinction_idx] - time[inception_idx]) * 1000)
    fault_window = i_primary[inception_idx: inception_idx + cycle_n]

    # Peak fault current (primary side)
    peak_fault_current_a = float(np.max(np.abs(i_primary[inception_idx: extinction_idx + 1]))) if inception_idx < extinction_idx else float(np.max(np.abs(i_primary)))

    # di/dt max in fault window
    if len(fault_window) > 1:
        di_dt_max = float(np.max(np.abs(np.diff(fault_window))) * sr)
    else:
        di_dt_max = 0.0

    # THD of current at fault
    thd_percent = _thd_percent(fault_window, sr, freq) if len(fault_window) >= 4 else 0.0

    # Fault inception angle (FIA)
    fia_deg = 0.0
    if v_primary is not None and inception_idx < len(v_primary):
        v_peak = float(np.max(np.abs(v_primary[:inception_idx]))) if inception_idx > 0 else float(np.max(np.abs(v_primary)))
        if v_peak > 0:
            ratio = float(np.clip(v_primary[inception_idx] / v_peak, -1.0, 1.0))
            fia_deg = float(np.degrees(np.arcsin(ratio)))

    # Symmetrical components (zero-seq / pos-seq ratio)
    i0_i1_ratio = 0.0
    if ia is not None and ib is not None and ic is not None:
        seg_len = min(cycle_n, len(ia), len(ib), len(ic))
        s = inception_idx
        fa = ia[s: s + seg_len].astype(complex)
        fb = ib[s: s + seg_len].astype(complex)
        fc = ic[s: s + seg_len].astype(complex)
        if len(fa) >= 4:
            i0_rms, i1_rms, _ = _symmetrical_components(fa, fb, fc)
            i0_i1_ratio = float(i0_rms / i1_rms) if i1_rms > 0 else 0.0

    # Voltage sag features
    voltage_sag_depth_pu = 0.0
    voltage_phase_ratio_spread_pu = 0.0
    healthy_phase_voltage_ratio = 1.0
    v2_v1_ratio = 0.0
    voltage_thd_max_percent = 0.0

    v_channels = [(va, "A"), (vb, "B"), (vc, "C")]
    prefault_v_rms = []
    fault_v_rms = []
    for v_ch, _ in v_channels:
        if v_ch is None:
            continue
        pre_v = v_ch[:pre_end]
        fault_v = v_ch[inception_idx: inception_idx + cycle_n]
        if len(pre_v) > 1:
            prefault_v_rms.append(_rms_window(v_ch, 0, pre_end))
        if len(fault_v) > 1:
            fault_v_rms.append(_rms_window(v_ch, inception_idx, cycle_n))

    if prefault_v_rms and fault_v_rms:
        pre_mean = float(np.mean(prefault_v_rms))
        fault_min = float(np.min(fault_v_rms))
        if pre_mean > 0:
            voltage_sag_depth_pu = max(0.0, float((pre_mean - fault_min) / pre_mean))
        sag_ratios = [f / p if p > 0 else 1.0 for f, p in zip(fault_v_rms, prefault_v_rms)]
        voltage_phase_ratio_spread_pu = float(np.std(sag_ratios)) if len(sag_ratios) > 1 else 0.0
        healthy_phase_voltage_ratio = float(np.max(sag_ratios)) if sag_ratios else 1.0

    if va is not None and vb is not None and vc is not None:
        seg_len = min(cycle_n, len(va), len(vb), len(vc))
        s = inception_idx
        fva = va[s: s + seg_len].astype(complex)
        fvb = vb[s: s + seg_len].astype(complex)
        fvc = vc[s: s + seg_len].astype(complex)
        if len(fva) >= 4:
            _, v1_rms, v2_rms = _symmetrical_components(fva, fvb, fvc)
            v2_v1_ratio = float(v2_rms / v1_rms) if v1_rms > 0 else 0.0

    if fault_v_rms:
        v_thds = []
        for v_ch, _ in v_channels:
            if v_ch is None:
                continue
            seg = v_ch[inception_idx: inception_idx + cycle_n]
            v_thds.append(_thd_percent(seg, sr, freq))
        voltage_thd_max_percent = float(max(v_thds)) if v_thds else 0.0

    # Current step ratio: largest faulted-phase peak vs its own prefault RMS.
    # ~1.0–1.4 on steady load, ≫2 on a real fault. Used by the no-fault gate.
    peak_to_prefault_ratio = float(np.max(np.abs(i_primary)) / pre_rms) if pre_rms > 0 else 0.0

    digital = _digital_sequence_features(status_channels, time, inception_idx)

    # AR result — trust only the tightened detection in _digital_sequence_features.
    # The previous fallback (any AR-named channel containing a `1` ⇒ success) was
    # firing on AR CLOSE command / AR-ready / CB-position channels that are already
    # HIGH before fault inception (or polarity-inverted), producing false "berhasil"
    # verdicts even when the breaker never physically reclosed.
    ar_result = digital.get("digital_ar_status")

    # Ground fault detection (I0 > 20% of I1)
    is_ground = i0_i1_ratio > 0.2

    # Trip type from status channels
    trip_type_str = "unknown"
    for sch in status_channels:
        name = sch.get("name", "").upper()
        if not _status_any_on(sch.get("samples") or []):
            continue
        if "3PH" in name or "THREE" in name or "3P" in name or "THREE_POLE" in name:
            trip_type_str = "three_pole"
            break
        if "1PH" in name or "SINGLE" in name or "1P" in name or "SINGLE_POLE" in name:
            trip_type_str = "single_pole"
            break
    if digital.get("digital_trip_type"):
        trip_type_str = digital["digital_trip_type"]

    # Faulted phases — per-phase 3× pre-fault RMS threshold to avoid false 3Ph detection
    faulted_phases = []
    for ph_arr, phase in [(ia, "A"), (ib, "B"), (ic, "C")]:
        if ph_arr is None:
            continue
        pre_end_ph = min(2 * cycle_n, len(ph_arr) // 4)
        if pre_end_ph < 2:
            continue
        pre_rms_ph = float(np.sqrt(np.mean(ph_arr[:pre_end_ph] ** 2)))
        ph_thr = max(pre_rms_ph * 3.0, peak_fault_current_a * 0.10, 1.0)
        seg = ph_arr[inception_idx: inception_idx + cycle_n]
        if len(seg) > 0 and float(np.max(np.abs(seg))) > ph_thr:
            faulted_phases.append(phase)
    faulted_phases_str = "+".join(faulted_phases) if faulted_phases else "A"
    if len(digital.get("digital_startup_phases") or []) == 1:
        faulted_phases_str = digital["digital_startup_phases"][0]

    # Broad "did any protection actually operate?" scan. _digital_sequence_features
    # only catches phase-tagged trips/CB-opens; unphased operate bits like Op_Prot,
    # 87L.Op, 21Lx.Op, 78.Op, or BO "TRIP" channels must also count. We exclude
    # standing-status bits (.On/.Valid/.Ready/_OK/Pkp) which are HIGH in normal
    # service and a mere fault-detector pickup (Pkp) that resets itself.
    operate_active = False
    for sch in status_channels:
        nm = str(sch.get("name", "") or "").upper()
        if not _status_any_on(sch.get("samples") or []):
            continue
        if _is_operate_status(nm):
            operate_active = True
            break

    ct_anomaly = detect_ct_measurement_anomaly(
        {"A": ia, "B": ib, "C": ic},
        {"A": va, "B": vb, "C": vc},
        sr,
        freq,
        inception_idx,
        extinction_idx,
        fault_duration_ms,
    )

    # Zone from status channels
    zone_str = ""
    for sch in status_channels:
        name = sch.get("name", "").upper()
        for z in ("ZONE 1", "ZONE 2", "ZONE 3", "Z1", "Z2", "Z3"):
            if z in name and 1 in (sch.get("samples") or []):
                zone_str = z.replace(" ", "")
                break
    if digital.get("digital_zone"):
        zone_str = digital["digital_zone"]

    result = {
        "fault_duration_ms": round(max(fault_duration_ms, 0.0), 1),
        "fault_count": 1,
        "peak_fault_current_a": round(peak_fault_current_a, 2),
        "di_dt_max": round(di_dt_max, 2),
        "i0_i1_ratio": round(i0_i1_ratio, 3),
        "thd_percent": round(thd_percent, 2),
        "inception_angle_degrees": round(fia_deg, 1),
        "voltage_sag_depth_pu": round(voltage_sag_depth_pu, 3),
        "voltage_phase_ratio_spread_pu": round(voltage_phase_ratio_spread_pu, 3),
        "healthy_phase_voltage_ratio": round(healthy_phase_voltage_ratio, 3),
        "v2_v1_ratio": round(v2_v1_ratio, 3),
        "voltage_thd_max_percent": round(voltage_thd_max_percent, 2),
        "reclose_successful": ar_result,
        "is_ground_fault": is_ground,
        "trip_type": trip_type_str,
        "faulted_phases": faulted_phases_str,
        "zone_operated": zone_str,
        "ct_anomaly_detected": bool(ct_anomaly.get("detected")),
        "ct_anomaly_phase": str(ct_anomaly.get("phase", "") or ""),
        "ct_anomaly_evidence": str(ct_anomaly.get("evidence", "") or ""),
        "peak_to_prefault_ratio": round(peak_to_prefault_ratio, 2),
        "protection_operated": bool(operate_active),
    }
    result.update(digital)
    return result


def _empty_features() -> dict:
    row = {k: 0 for k in [
        "fault_duration_ms", "fault_count", "peak_fault_current_a", "di_dt_max",
        "i0_i1_ratio", "thd_percent", "inception_angle_degrees", "voltage_sag_depth_pu",
        "voltage_phase_ratio_spread_pu", "healthy_phase_voltage_ratio", "v2_v1_ratio",
        "voltage_thd_max_percent", "reclose_successful", "is_ground_fault",
        "trip_type", "faulted_phases", "zone_operated",
        "peak_to_prefault_ratio", "protection_operated",
    ]}
    row.update({
        "digital_trip_type": None,
        "digital_trip_phases": [],
        "digital_cb_open_phases": [],
        "digital_cb_close_phases": [],
        "digital_startup_phases": [],
        "digital_fault_phases": [],
        "digital_zone": "",
        "digital_ar_status": None,
        "digital_ar_attempted": False,
        "digital_ar_lockout": False,
        "digital_ar_not_ready": False,
        "digital_first_startup_ms": None,
        "digital_first_trip_ms": None,
        "digital_startup_to_trip_ms": None,
        "digital_first_cb_open_ms": None,
        "digital_first_cb_close_ms": None,
        "digital_ar_dead_time_ms": None,
        "digital_reclose_mode": None,
        "digital_topology_hint": None,
        "digital_cb_contact_breakers": [],
        "digital_cb_contact_open_phases": [],
        "digital_cb_contact_close_phases": [],
    })
    return row


_LABEL_MAP = {
    "PETIR":       "Petir / Lightning",
    "LAYANG":      "Layang-Layang / Kite",
    "POHON":       "Pohon / Vegetasi",
    "HEWAN":       "Hewan / Binatang",
    "BENDA_ASING": "Benda Asing",
    "KONDUKTOR":   "Konduktor / Tower",
    "PERALATAN":   "Peralatan Rusak / Anomali Proteksi",
}


def _build_narrative_evidence(row: dict, ranking: list, pred: str, confidence: float, margin: float) -> list[dict]:
    """Structured evidence — each item carries severity + weight so the UI can rank visually."""
    evidence: list[dict] = []
    phases_str = row.get("faulted_phases", "A")
    phases = [p for p in phases_str.split("+") if p]
    is_ground = bool(row.get("is_ground_fault", False))
    n_phases = len(phases)
    peak_a = row.get("peak_fault_current_a", 0.0)
    sag_pct = row.get("voltage_sag_depth_pu", 0.0) * 100

    if n_phases >= 3:
        type_name = "Gangguan Tiga Fasa"
    elif n_phases == 2 and is_ground:
        type_name = "Double Line to Ground"
    elif n_phases == 2:
        type_name = "Line to Line"
    elif is_ground:
        type_name = "Single Line to Ground"
    else:
        type_name = "Single Line"

    phases_label = "+".join(phases) + ("-N" if is_ground and n_phases < 3 else "")

    zone = row.get("zone_operated", "")
    trip = row.get("trip_type", "")
    dur = row.get("fault_duration_ms", 0.0)
    trip_label = trip.replace("_", " ") if trip and trip != "unknown" else ""

    fault_clauses = []
    if peak_a > 0:
        fault_clauses.append(f"kenaikan arus puncak mencapai {peak_a:.2f} A")
    if sag_pct > 0:
        fault_clauses.append(f"penurunan tegangan (voltage sag) sebesar {sag_pct:.1f}%")

    fault_sentence = f"Terdeteksi Gangguan Fasa {phases_label} ({type_name})"
    if fault_clauses:
        fault_sentence += " dengan " + " dan ".join(fault_clauses)
    fault_sentence += "."
    evidence.append(_ev(fault_sentence, "info", kind="physics"))

    prot_bits = []
    if zone:
        prot_bits.append(f"Proteksi Zona {zone} bekerja mentrigger TRIP")
        if trip_label:
            prot_bits[-1] += f" ({trip_label})"
    elif trip_label:
        prot_bits.append(f"Proteksi mentrigger TRIP ({trip_label}) namun zona proteksi tidak dapat diidentifikasi dari rekaman")
    elif dur > 0:
        prot_bits.append("Zona dan tipe trip proteksi tidak dapat diidentifikasi dari rekaman")
    if prot_bits:
        text = prot_bits[0]
        if dur > 0:
            text += f", dengan Fault Clearing Time (FCT) {dur:.0f} ms pada fasa terganggu {phases_label}"
        text += "."
        evidence.append(_ev(text, "info", kind="physics"))
    elif dur > 0:
        evidence.append(_ev(f"Fault Clearing Time (FCT) {dur:.0f} ms pada fasa terganggu {phases_label}.", "info", kind="physics"))

    if peak_a > 100_000:
        evidence.append(_ev(
            f"Catatan validasi rasio: I puncak terhitung {peak_a:.0f} A (>100 kA). "
            "Nilai setinggi ini sering mengindikasikan CT/VT ratio belum sesuai; verifikasi rasio COMTRADE, "
            "setting relay/RIO, atau input manual sebelum menyimpulkan besaran arus primer.",
            "warning", kind="physics",
        ))

    trip_phases = row.get("digital_trip_phases") or []
    cb_open_phases = row.get("digital_cb_open_phases") or []
    startup_phases = row.get("digital_startup_phases") or []
    startup_to_trip = row.get("digital_startup_to_trip_ms")
    topology_hint = row.get("digital_topology_hint")
    contact_breakers = row.get("digital_cb_contact_breakers") or []
    if trip_phases or cb_open_phases or startup_phases:
        details = []
        if startup_phases:
            details.append(f"startup fase {', '.join(startup_phases)}")
        if trip_phases:
            details.append(f"trip fase {', '.join(trip_phases)}")
        if cb_open_phases:
            details.append(f"CB open fase {', '.join(cb_open_phases)}")
        if startup_to_trip is not None:
            details.append(f"selang startup ke trip {startup_to_trip:.1f} ms")
        evidence.append(_ev("Pembacaan kanal digital menunjukkan " + "; ".join(details) + ".", "info", kind="physics"))

    if topology_hint == "one_and_half_breaker":
        breaker_text = "/".join(contact_breakers) if contact_breakers else "dua PMT"
        evidence.append(_ev(
            f"Kanal kontak {breaker_text} menunjukkan indikasi bay 1.5 breaker; "
            "status reclose dievaluasi dari urutan kontak PMT membuka lalu menutup kembali.",
            "info", kind="physics",
        ))

    if cb_open_phases and len(cb_open_phases) >= 3 and trip == "single_pole":
        evidence.append(_ev(
            "Catatan koreksi: meskipun loop gangguan terdeteksi satu fasa ke tanah, kanal digital menunjukkan "
            "PMT/CB tiga fasa membuka; trip operasi lebih tepat dibaca sebagai three-pole.",
            "warning", kind="physics",
        ))

    ar_ok = row.get("reclose_successful")
    ar_dead_ms = row.get("digital_ar_dead_time_ms")
    reclose_mode = row.get("digital_reclose_mode")
    cb_close_phases = row.get("digital_cb_close_phases") or []
    ar_attempted = bool(row.get("digital_ar_attempted"))
    mode_label = {"single_pole": "single-pole reclose", "three_pole": "three-pole reclose"}.get(reclose_mode or "", "")
    parts = []
    if ar_dead_ms is not None:
        parts.append(f"dead time ≈ {ar_dead_ms:.0f} ms")
    if mode_label:
        phase_str = "+".join(cb_close_phases) if cb_close_phases else ""
        parts.append(f"{mode_label}{f' (fase {phase_str})' if phase_str else ''}")
    ar_suffix = f" ({'; '.join(parts)})" if parts else ""

    if ar_ok is True:
        evidence.append(_ev(f"Auto Reclose (AR) berhasil — gangguan terkonfirmasi bersifat transien{ar_suffix}.", "notable", kind="physics"))
    elif ar_ok is False:
        evidence.append(_ev(f"Auto Reclose (AR) gagal — gangguan kemungkinan bersifat permanen{ar_suffix}.", "notable", kind="physics"))
    elif ar_attempted and cb_open_phases and not cb_close_phases:
        evidence.append(_ev(
            f"Urutan single-pole/open-pole terdeteksi pada kanal digital (fase {', '.join(cb_open_phases)}), "
            "tetapi kontak PMT menutup kembali tidak terekam dalam durasi file; status keberhasilan AR belum dapat dipastikan dari rekaman ini.",
            "info", kind="physics",
        ))
    else:
        evidence.append(_ev("Status Auto Reclose (AR) tidak teridentifikasi dari rekaman digital.", "info", kind="physics"))

    top_label = ranking[0]['label'] if ranking else "—"
    evidence.append(_ev(
        f"Berdasarkan analisis pola gelombang, AI mengklasifikasikan gangguan ini sebagai {top_label} "
        f"dengan tingkat keyakinan {confidence:.0%}.",
        "verdict", weight=min(1.0, confidence + 0.05), kind="model",
    ))

    if pred == "PETIR":
        subtype_line = _petir_subtype_description(row)
        if subtype_line:
            evidence.append(_ev(subtype_line, "notable", kind="physics"))

    if pred == "PETIR" and (
        row.get("digital_ar_lockout") or row.get("digital_ar_not_ready")
        or (cb_open_phases and len(cb_open_phases) >= 3)
        or (startup_to_trip is not None and startup_to_trip > 20)
    ):
        evidence.append(_ev(
            "Catatan interpretasi: label PETIR perlu dibaca hati-hati karena kanal digital menunjukkan "
            "evolusi pickup/trip dan/atau operasi tiga fasa/AR lockout; kandidat fisik seperti Hewan, "
            "Konduktor, atau Peralatan tetap perlu diverifikasi dari inspeksi lapangan.",
            "critical", kind="cap",
        ))

    fia = row.get("inception_angle_degrees", 0.0)
    if abs(fia) > 60:
        evidence.append(_ev(
            f"Fault Inception Angle (FIA) = {fia:.1f}° — gangguan terjadi dekat puncak tegangan, pola tipikal gangguan petir.",
            "notable", kind="physics",
        ))

    if margin < 0.15 and len(ranking) >= 2:
        evidence.append(_ev(
            f"Catatan: Selisih keyakinan ke kandidat kedua ({ranking[1]['label']}) hanya {margin * 100:.1f} pp — verifikasi lapangan tetap disarankan.",
            "warning", kind="model",
        ))

    return evidence


def _no_fault_gate(payload: dict) -> Optional[dict]:
    """Physics precondition: was there actually a fault to classify?

    Delegates the decision to the shared :func:`detect_fault_presence` (single
    source of truth used by relay-21/87L, electrical params, locus, and the PDF
    report) so every surface agrees. Returns a NONE/no-fault result dict to
    short-circuit the pipeline, or None to let normal classification proceed.
    """
    det = detect_fault_presence(payload)
    if det.is_fault:
        return None

    evidence = [
        _ev(
            "Tidak terdeteksi gangguan: " + "; ".join(det.reasons) + ".",
            "verdict", weight=0.95, kind="physics",
        ),
        _ev(
            "Rekaman kemungkinan ter-trigger oleh pickup/status non-proteksi "
            "(mis. FD pickup, sync/teleprotection/GPS/komunikasi) tanpa proteksi bekerja. "
            "Klasifikasi penyebab dan perhitungan impedansi/locus tidak dijalankan karena "
            "tidak ada gangguan untuk dianalisa.",
            "notable", kind="physics",
        ),
    ]
    return {
        "fault_type": "none",
        "cause_ranking": [],
        "overall_confidence": 0.0,
        "evidence": evidence,
        "no_fault": True,
        "tier1": {"fired": False},
        "raw_probabilities": None,
        "calibrated_probabilities": None,
        "applied_caps": [],
        "feature_vector_used": None,
    }


def run_ml_prediction(payload: dict, relay_type: str = "21") -> dict:
    """Run the LightGBM fault classifier on a session payload.

    Returns a dict matching the (now-extended) AIFaultResult schema. The
    response embeds enough provenance for downstream auditing:

      - ``raw_probabilities`` — pre-calibration LightGBM output
      - ``calibrated_probabilities`` — post-calibration, pre-cap
      - ``applied_caps`` — list of confidence caps that fired
      - ``feature_vector_used`` — exact features fed to the classifier
      - ``tier1`` — Tier 1 rule match (if any)
      - ``meta`` — model version, feature columns, calibration method, etc.

    Heavy model imports are deferred to here so server startup never fails.
    """
    try:
        from models.train import FEATURE_COLS  # noqa: F401
        from models.predict import (
            _build_feature_vector,
            _apply_transient_ambiguity_confidence_cap,
            _apply_equipment_caution_cap,
        )
    except Exception as e:
        return {
            "fault_type": "transient",
            "cause_ranking": [],
            "overall_confidence": 0.0,
            "evidence": [_ev(f"Model imports gagal: {e}", "critical", kind="model")],
            "meta": _model_metadata(None),
        }

    model_bundle = _load_model()
    meta = _model_metadata(model_bundle)

    # ------------------------------------------------------------------
    # Tier 0 — no-fault gate (physics precondition before any classification)
    # ------------------------------------------------------------------
    gate = _no_fault_gate(payload)
    if gate is not None:
        gate["meta"] = meta
        return gate

    row = extract_ml_features(payload, relay_type)
    # Inject SOE-derived loop context so Tier 1 sanity rules see the same
    # signals the CLI inference path has. SOE list is not built here yet, so
    # this is a no-op until a future change passes a real SOE through.
    row = _augment_row_with_soe_context(row, soe=None)

    # ------------------------------------------------------------------
    # Tier 1 — deterministic rules
    # ------------------------------------------------------------------
    tier1 = apply_rules(row)
    if tier1 is not None:
        # Translate rule label back to the cause_ranking format the UI uses.
        # Some Tier 1 rules emit labels that do not map cleanly to the 7-class
        # taxonomy (e.g. "GANGGUAN PERMANEN"); in that case we still surface
        # the rule's confidence on the matching class when possible.
        rule_to_cause = {
            "KONDUKTOR / TOWER": "KONDUKTOR",
            "PERALATAN RUSAK / ANOMALI PROTEKSI": "PERALATAN",
            "GANGGUAN PERMANEN": None,  # no direct cause class
        }
        cause_key = rule_to_cause.get(tier1.label)
        ranking: list[dict] = []
        if cause_key is not None:
            ranking.append({"cause": cause_key, "label": _LABEL_MAP.get(cause_key, tier1.label), "confidence": round(float(tier1.confidence), 3)})
            for k, v in _LABEL_MAP.items():
                if k == cause_key:
                    continue
                ranking.append({"cause": k, "label": v, "confidence": round((1.0 - tier1.confidence) / max(len(_LABEL_MAP) - 1, 1), 3)})
        else:
            # Permanent-fault rules — present a flat distribution but keep
            # KONDUKTOR slightly elevated since that is the modal cause.
            for k, v in _LABEL_MAP.items():
                base = 0.18 if k == "KONDUKTOR" else 0.05
                ranking.append({"cause": k, "label": v, "confidence": round(base, 3)})

        fault_type = "permanent"  # all current Tier 1 rules describe permanent / equipment issues

        evidence: list[dict] = [
            _ev(f"Tier 1 rule '{tier1.rule_name}' aktif — {tier1.label}.", "verdict", weight=float(tier1.confidence), kind="rule"),
            _ev(tier1.evidence, "critical", weight=float(tier1.confidence), kind="rule"),
            _ev(
                "Klasifikasi ini berasal dari aturan deterministik (rules.py), bukan model LightGBM. "
                "Verifikasi lapangan tetap disarankan.",
                "notable", kind="rule",
            ),
        ]

        return {
            "fault_type": fault_type,
            "cause_ranking": ranking,
            "overall_confidence": float(tier1.confidence),
            "evidence": evidence,
            "tier1": {
                "fired": True,
                "rule_name": tier1.rule_name,
                "label": tier1.label,
                "confidence": float(tier1.confidence),
                "evidence": tier1.evidence,
            },
            "raw_probabilities": None,
            "calibrated_probabilities": None,
            "applied_caps": [],
            "feature_vector_used": None,
            "meta": meta,
        }

    # ------------------------------------------------------------------
    # Tier 2 — LightGBM
    # ------------------------------------------------------------------
    if model_bundle is None:
        n = len(_LABEL_MAP)
        ranking = [{"cause": k, "label": v, "confidence": round(1 / n, 3)} for k, v in _LABEL_MAP.items()]
        return {
            "fault_type": "transient",
            "cause_ranking": ranking,
            "overall_confidence": round(1 / n, 3),
            "evidence": [_ev("Model fault_classifier.pkl tidak ditemukan — prediksi tidak tersedia.", "critical", kind="model")],
            "tier1": {"fired": False},
            "meta": meta,
        }

    clf = model_bundle["clf"]
    classes = list(getattr(clf, "classes_", model_bundle.get("classes", [])))
    feature_cols = list(model_bundle.get("feature_cols") or [])

    X = _build_feature_vector(row, feature_cols)
    pred = str(clf.predict(X)[0])
    raw_proba = np.asarray(clf.predict_proba(X)[0], dtype=float)

    calibrated_proba, calibration_method = _apply_calibrator(raw_proba, classes, X)

    applied_caps: list[dict] = []
    confidence = float(calibrated_proba.max())
    if confidence > 0.92:
        applied_caps.append({"name": "ceiling_92", "before": confidence, "after": 0.92, "reason": "Hard ceiling 92% untuk mencegah false confidence pada model klasifikasi 7-kelas dengan data terbatas."})
        confidence = 0.92

    sorted_p = np.sort(calibrated_proba)[::-1]
    margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) >= 2 else 1.0

    new_conf, ambiguity_note = _apply_transient_ambiguity_confidence_cap(
        confidence=confidence, pred_label=pred, proba_classes=classes,
        proba=calibrated_proba, margin=margin,
    )
    if new_conf != confidence:
        applied_caps.append({"name": "transient_ambiguity", "before": confidence, "after": new_conf, "reason": ambiguity_note.strip(" []")})
        confidence = new_conf

    new_conf, equipment_note = _apply_equipment_caution_cap(
        pred_label=pred, confidence=confidence,
        class_counts=model_bundle.get("class_counts"),
        soe=None,
        protection_name="DISTANCE" if relay_type == "21" else "DIFFERENTIAL",
    )
    if new_conf != confidence:
        applied_caps.append({"name": "equipment_caution", "before": confidence, "after": new_conf, "reason": equipment_note.strip(" []")})
        confidence = new_conf

    ranking = sorted(
        [{"cause": cls, "label": _LABEL_MAP.get(cls, cls), "confidence": round(float(p), 3)} for cls, p in zip(classes, calibrated_proba)],
        key=lambda x: x["confidence"], reverse=True,
    )

    # PETIR + caution-digital cap
    digital_cb_open_phases = row.get("digital_cb_open_phases") or []
    digital_startup_to_trip = row.get("digital_startup_to_trip_ms")
    digital_caution = (
        row.get("digital_ar_lockout") or row.get("digital_ar_not_ready")
        or len(digital_cb_open_phases) >= 3
        or (digital_startup_to_trip is not None and digital_startup_to_trip > 20)
    )
    if pred == "PETIR" and digital_caution:
        if confidence > 0.72:
            applied_caps.append({"name": "petir_digital_caution", "before": confidence, "after": 0.72, "reason": "AR lockout / startup-to-trip terlalu lama / CB tiga fasa membuka — meragukan label PETIR."})
        confidence = min(confidence, 0.72)
        for item in ranking:
            if item["cause"] == "PETIR":
                item["confidence"] = min(float(item["confidence"]), 0.72)
            elif item["cause"] == "HEWAN":
                item["confidence"] = max(float(item["confidence"]), 0.12)
            elif item["cause"] == "KONDUKTOR":
                item["confidence"] = max(float(item["confidence"]), 0.08)
            elif item["cause"] == "PERALATAN":
                item["confidence"] = max(float(item["confidence"]), 0.06)
        ranking.sort(key=lambda x: x["confidence"], reverse=True)

    top_class = ranking[0]["cause"] if ranking else pred
    fault_type = "transient" if top_class in _TRANSIENT else "permanent"

    evidence = _build_narrative_evidence(row, ranking, pred, confidence, margin)

    raw_dict = {cls: round(float(p), 4) for cls, p in zip(classes, raw_proba)}
    cal_dict = {cls: round(float(p), 4) for cls, p in zip(classes, calibrated_proba)}
    feature_vector_used = {col: float(X[0, i]) for i, col in enumerate(feature_cols)} if feature_cols else None

    meta = dict(meta)
    meta["calibration_method_used"] = calibration_method
    meta["margin"] = round(margin, 4)

    return {
        "fault_type": fault_type,
        "cause_ranking": ranking,
        "overall_confidence": confidence,
        "evidence": evidence,
        "tier1": {"fired": False},
        "raw_probabilities": raw_dict,
        "calibrated_probabilities": cal_dict,
        "applied_caps": applied_caps,
        "feature_vector_used": feature_vector_used,
        "meta": meta,
    }
