"""Bridge between COMTRADE session JSON and the existing fault_classifier.pkl model.

Extracts the same 17-feature vector that models/train.py uses, then runs the
LightGBM multi-class classifier with the same calibration and confidence caps
as models/predict.py.
"""

import pickle
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_PIPELINE_DIR = Path(__file__).parent.parent.parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from models.predict import _petir_subtype_description  # noqa: E402

_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "fault_classifier.pkl"

_LABEL_DISPLAY = {
    "PETIR":       "Petir / Lightning",
    "LAYANG":      "Layang-Layang / Kite",
    "POHON":       "Pohon / Vegetasi",
    "HEWAN":       "Hewan / Binatang",
    "BENDA_ASING": "Benda Asing",
    "KONDUKTOR":   "Konduktor / Tower",
    "PERALATAN":   "Peralatan / Proteksi",
}

_TRANSIENT = {"PETIR", "LAYANG", "HEWAN", "BENDA_ASING"}


def _load_model() -> Optional[dict]:
    if not _MODEL_PATH.exists():
        return None
    with open(_MODEL_PATH, "rb") as f:
        return pickle.load(f)


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


def _symmetrical_components(ia: np.ndarray, ib: np.ndarray, ic: np.ndarray):
    """Return (I0_rms, I1_rms, I2_rms) using one-sample approximation of seq."""
    a = np.exp(1j * 2 * np.pi / 3)
    i0 = (ia + ib + ic) / 3.0
    i1 = (ia + a * ib + (a ** 2) * ic) / 3.0
    i2 = (ia + (a ** 2) * ib + a * ic) / 3.0
    return float(np.sqrt(np.mean(np.abs(i0) ** 2))), float(np.sqrt(np.mean(np.abs(i1) ** 2))), float(np.sqrt(np.mean(np.abs(i2) ** 2)))


def _phase_from_status_name(name: str) -> Optional[str]:
    upper = name.upper()
    for phase in "ABC":
        patterns = [
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


def _status_any_on(samples: list) -> bool:
    return any(int(v) == 1 for v in samples or [])


def _digital_sequence_features(status_channels: list, time: np.ndarray, inception_idx: int) -> dict:
    start_idx = max(0, inception_idx - 2)
    trip_phases: dict[str, float] = {}
    cb_open_phases: dict[str, float] = {}
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

        is_trip = "TRIP" in name
        is_cb_open = "CB OPEN" in name or "PMT OPEN" in name or "BREAKER OPEN" in name or ("52" in name and "OPEN" in name)
        is_startup = "STARTUP" in name or "START UP" in name or "PICKUP" in name or "PICK UP" in name
        is_fault = "FAULT" in name and phase is not None

        if phase and first_ms is not None:
            if is_trip:
                trip_phases[phase] = min(first_ms, trip_phases.get(phase, first_ms))
            if is_cb_open:
                cb_open_phases[phase] = min(first_ms, cb_open_phases.get(phase, first_ms))
            if is_startup:
                startup_phases[phase] = min(first_ms, startup_phases.get(phase, first_ms))
            if is_fault:
                fault_phases[phase] = min(first_ms, fault_phases.get(phase, first_ms))

        if is_trip and "ZONE" in name and first_ms is not None:
            for zone in ("1", "2", "3", "4", "5"):
                if f"ZONE{zone}" in name.replace(" ", "") or f"Z{zone}" in name:
                    zone_times[f"Z{zone}"] = min(first_ms, zone_times.get(f"Z{zone}", first_ms))

        if "AR" in name or "RECLOS" in name or "RECLOSE" in name:
            if "LOCKOUT" in name or "LOCK OUT" in name:
                ar_flags["lockout"] = True
                ar_flags["failed"] = True
            if "NOT READY" in name or "NOTREADY" in name or "BLOCK" in name:
                ar_flags["not_ready" if "READY" in name else "block"] = True
                ar_flags["failed"] = True
            if "SUCCESS" in name or "CLOSE" in name or "RECLOS" in name:
                if "LOCK" not in name and "NOT READY" not in name and "BLOCK" not in name:
                    ar_flags["successful"] = True

    def sorted_phases(data: dict[str, float]) -> list[str]:
        return sorted(data.keys(), key=lambda ph: data[ph])

    trip_type = None
    if len(cb_open_phases) >= 3 or len(trip_phases) >= 3:
        trip_type = "three_pole"
    elif len(cb_open_phases) == 1 or len(trip_phases) == 1:
        trip_type = "single_pole"

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

    return {
        "digital_trip_type": trip_type,
        "digital_trip_phases": sorted_phases(trip_phases),
        "digital_cb_open_phases": sorted_phases(cb_open_phases),
        "digital_startup_phases": sorted_phases(startup_phases),
        "digital_fault_phases": sorted_phases(fault_phases),
        "digital_zone": min(zone_times, key=zone_times.get) if zone_times else "",
        "digital_ar_status": ar_status,
        "digital_ar_lockout": ar_flags["lockout"],
        "digital_ar_not_ready": ar_flags["not_ready"],
        "digital_first_startup_ms": round(first_startup_ms, 2) if first_startup_ms is not None else None,
        "digital_first_trip_ms": round(first_trip_ms, 2) if first_trip_ms is not None else None,
        "digital_startup_to_trip_ms": round(first_trip_ms - first_startup_ms, 2)
        if first_startup_ms is not None and first_trip_ms is not None else None,
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

    digital = _digital_sequence_features(status_channels, time, inception_idx)

    # AR result
    ar_result = None
    for sch in status_channels:
        name = sch.get("name", "").upper()
        if any(k in name for k in ("AR", "RECLOSE", "RECLUSE", "RECLOS")):
            if 1 in (sch.get("samples") or []):
                ar_result = True
            else:
                ar_result = False
            break
    if digital.get("digital_ar_status") is not None:
        ar_result = digital["digital_ar_status"]

    # Ground fault detection (I0 > 20% of I1)
    is_ground = i0_i1_ratio > 0.2

    # Trip type from status channels
    trip_type_str = "unknown"
    for sch in status_channels:
        name = sch.get("name", "").upper()
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
    ]}
    row.update({
        "digital_trip_type": None,
        "digital_trip_phases": [],
        "digital_cb_open_phases": [],
        "digital_startup_phases": [],
        "digital_fault_phases": [],
        "digital_zone": "",
        "digital_ar_status": None,
        "digital_ar_lockout": False,
        "digital_ar_not_ready": False,
        "digital_first_startup_ms": None,
        "digital_first_trip_ms": None,
        "digital_startup_to_trip_ms": None,
    })
    return row


def run_ml_prediction(payload: dict, relay_type: str = "21") -> dict:
    """
    Run the LightGBM fault classifier on session payload.
    Returns a dict matching the AIFaultResult schema.
    Heavy model imports are deferred to here so server startup never fails.
    """
    # Lazy imports — only executed when AI analysis is requested
    try:
        from models.train import FEATURE_COLS, encode_reclose, encode_trip_type, encode_zone, parse_phase_count  # noqa: F401
        from models.predict import (
            _calibrate_proba,
            _build_feature_vector,
            _apply_transient_ambiguity_confidence_cap,
            _apply_equipment_caution_cap,
        )
    except Exception as e:
        return {
            "fault_type": "transient",
            "cause_ranking": [],
            "overall_confidence": 0.0,
            "evidence": [f"Model imports gagal: {e}"],
        }

    model_bundle = _load_model()
    row = extract_ml_features(payload, relay_type)

    LABEL_MAP = {
        "PETIR":       "Petir / Lightning",
        "LAYANG":      "Layang-Layang / Kite",
        "POHON":       "Pohon / Vegetasi",
        "HEWAN":       "Hewan / Binatang",
        "BENDA_ASING": "Benda Asing",
        "KONDUKTOR":   "Konduktor / Tower",
        "PERALATAN":   "Peralatan / Proteksi",
    }

    if model_bundle is None:
        n = len(LABEL_MAP)
        ranking = [
            {"cause": k, "label": v, "confidence": round(1 / n, 3)}
            for k, v in LABEL_MAP.items()
        ]
        return {
            "fault_type": "transient",
            "cause_ranking": ranking,
            "overall_confidence": round(1 / n, 3),
            "evidence": ["Model fault_classifier.pkl tidak ditemukan — prediksi tidak tersedia."],
        }

    clf = model_bundle["clf"]
    classes = list(getattr(clf, "classes_", model_bundle.get("classes", [])))

    try:
        from models.train import FEATURE_COLS as _FC
    except Exception:
        _FC = model_bundle.get("feature_cols", [])

    X = _build_feature_vector(row, model_bundle.get("feature_cols", _FC))
    pred = str(clf.predict(X)[0])
    proba = clf.predict_proba(X)[0]

    proba = _calibrate_proba(proba, temperature=1.5)
    confidence = float(proba.max())
    if confidence > 0.92:
        confidence = 0.92

    sorted_p = np.sort(proba)[::-1]
    margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) >= 2 else 1.0

    confidence, _ = _apply_transient_ambiguity_confidence_cap(
        confidence=confidence,
        pred_label=pred,
        proba_classes=classes,
        proba=proba,
        margin=margin,
    )
    confidence, _ = _apply_equipment_caution_cap(
        pred_label=pred,
        confidence=confidence,
        class_counts=model_bundle.get("class_counts"),
        soe=None,
        protection_name="DISTANCE" if relay_type == "21" else "DIFFERENTIAL",
    )

    ranking = sorted(
        [
            {
                "cause": cls,
                "label": LABEL_MAP.get(cls, cls),
                "confidence": round(float(p), 3),
            }
            for cls, p in zip(classes, proba)
        ],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    digital_trip_phases = row.get("digital_trip_phases") or []
    digital_cb_open_phases = row.get("digital_cb_open_phases") or []
    digital_startup_to_trip = row.get("digital_startup_to_trip_ms")
    digital_caution = (
        row.get("digital_ar_lockout")
        or row.get("digital_ar_not_ready")
        or len(digital_cb_open_phases) >= 3
        or (digital_startup_to_trip is not None and digital_startup_to_trip > 20)
    )
    if pred == "PETIR" and digital_caution:
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

    # Fault type: driven purely by ML top cause nature, not by AR result
    top_class = ranking[0]["cause"] if ranking else pred
    fault_type = "transient" if top_class in _TRANSIENT else "permanent"

    ar_ok = row.get("reclose_successful")

    # --- Rich narrative evidence ---
    evidence = []

    # 1. Fault detection summary
    phases_str = row.get("faulted_phases", "A")
    phases = [p for p in phases_str.split("+") if p]
    is_ground = bool(row.get("is_ground_fault", False))
    n_phases = len(phases)
    peak_a = row.get("peak_fault_current_a", 0.0)
    sag_pct = row.get("voltage_sag_depth_pu", 0.0) * 100

    if n_phases >= 3:
        fault_code, type_name = "3Ph", "Gangguan Tiga Fasa"
    elif n_phases == 2 and is_ground:
        fault_code, type_name = "DLG", "Double Line to Ground"
    elif n_phases == 2:
        fault_code, type_name = "LL", "Line to Line"
    elif is_ground:
        fault_code, type_name = "SLG", "Single Line to Ground"
    else:
        fault_code, type_name = "SL", "Single Line"

    phases_label = "+".join(phases) + ("-N" if is_ground and n_phases < 3 else "")

    if peak_a > 0 and sag_pct > 0:
        evidence.append(
            f"Terdeteksi Gangguan Fasa {phases_label} ({type_name}) "
            f"dengan kenaikan arus puncak mencapai {peak_a:.2f} A "
            f"dan penurunan tegangan (voltage sag) sebesar {sag_pct:.1f}%."
        )
    elif peak_a > 0:
        evidence.append(
            f"Terdeteksi Gangguan Fasa {phases_label} ({type_name}) "
            f"dengan kenaikan arus puncak mencapai {peak_a:.2f} A."
        )

    # 2. Protection operation
    zone = row.get("zone_operated", "")
    trip = row.get("trip_type", "")
    dur = row.get("fault_duration_ms", 0.0)
    if zone or (trip and trip != "unknown"):
        zone_str = f"Zona {zone}" if zone else "zona tidak teridentifikasi"
        trip_str = trip.replace("_", " ") if trip and trip != "unknown" else "tidak teridentifikasi"
        evidence.append(
            f"Fungsi proteksi {zone_str} bekerja mentrigger TRIP ({trip_str}) "
            f"dengan Fault Clearing Time (FCT) {dur:.0f} ms."
        )
    elif dur > 0:
        evidence.append(f"Fault Clearing Time (FCT) {dur:.0f} ms.")

    trip_phases = row.get("digital_trip_phases") or []
    cb_open_phases = row.get("digital_cb_open_phases") or []
    startup_phases = row.get("digital_startup_phases") or []
    startup_to_trip = row.get("digital_startup_to_trip_ms")
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
        evidence.append("Pembacaan kanal digital menunjukkan " + "; ".join(details) + ".")

    if cb_open_phases and len(cb_open_phases) >= 3 and trip == "single_pole":
        evidence.append(
            "Catatan koreksi: meskipun loop gangguan terdeteksi satu fasa ke tanah, kanal digital menunjukkan "
            "PMT/CB tiga fasa membuka; trip operasi lebih tepat dibaca sebagai three-pole."
        )

    # 3. AR status
    if ar_ok is True:
        evidence.append("Auto Reclose (AR) berhasil — gangguan terkonfirmasi bersifat transien.")
    elif ar_ok is False:
        evidence.append("Auto Reclose (AR) gagal — gangguan kemungkinan bersifat permanen.")
    else:
        evidence.append("Status Auto Reclose (AR) tidak teridentifikasi dari rekaman digital.")

    # 4. AI classification conclusion
    evidence.append(
        f"Berdasarkan analisis pola gelombang, AI mengklasifikasikan gangguan ini sebagai "
        f"{ranking[0]['label']} dengan tingkat keyakinan {confidence:.0%}."
    )

    # 4a. PETIR sub-mechanism (Shielding Failure vs Back-Flashover)
    if pred == "PETIR":
        subtype_line = _petir_subtype_description(row)
        if subtype_line:
            evidence.append(subtype_line)

    if pred == "PETIR" and (
        row.get("digital_ar_lockout")
        or row.get("digital_ar_not_ready")
        or (cb_open_phases and len(cb_open_phases) >= 3)
        or (startup_to_trip is not None and startup_to_trip > 20)
    ):
        evidence.append(
            "Catatan interpretasi: label PETIR perlu dibaca hati-hati karena kanal digital menunjukkan "
            "evolusi pickup/trip dan/atau operasi tiga fasa/AR lockout; kandidat fisik seperti Hewan, "
            "Konduktor, atau Peralatan tetap perlu diverifikasi dari inspeksi lapangan."
        )

    # 5. FIA note
    fia = row.get("inception_angle_degrees", 0.0)
    if abs(fia) > 60:
        evidence.append(
            f"Fault Inception Angle (FIA) = {fia:.1f}° — gangguan terjadi dekat puncak tegangan, "
            f"pola tipikal gangguan petir."
        )

    # 6. Thin margin warning
    if margin < 0.15 and len(ranking) >= 2:
        evidence.append(
            f"Catatan: Selisih keyakinan ke kandidat kedua ({ranking[1]['label']}) "
            f"hanya {margin * 100:.1f} pp — verifikasi lapangan tetap disarankan."
        )

    return {
        "fault_type": fault_type,
        "cause_ranking": ranking,
        "overall_confidence": confidence,
        "evidence": evidence,
    }
