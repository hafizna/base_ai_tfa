"""
Multi-class Fault Cause Classifier
====================================
Trains a LightGBM candidate on labeled_features_v2.csv to classify the
physical cause of transmission line faults into 7 categories:

    PETIR       — lightning (direct strike or induced overvoltage)
    LAYANG      — kite (layang-layang)
    POHON       — tree / vegetation contact
    HEWAN       — animal (ular, binatang, burung, babi, tikus, etc.)
    BENDA_ASING — non-living foreign object (aluminium foil, terpal, etc.)
    KONDUKTOR   — conductor / tower structural failure
    PERALATAN   — equipment / protection / telecom-origin failure

Training now uses the shared live-analysis dataset and a paired grouped-CV
promotion gate. This compatibility entrypoint writes only a candidate; see
TRAINING_PIPELINE.md and python -m models.retrain --help for promotion.

Run from the repo root:
    python models/train.py
"""

import sys
import argparse
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FEATURES_CSV = Path(__file__).parent.parent / "data" / "features" / "labeled_features_v2.csv"
MODEL_OUT    = Path(__file__).parent / "candidates" / "fault_classifier.pkl"

ALL_CLASSES = ["PETIR", "LAYANG", "POHON", "HEWAN", "BENDA_ASING", "KONDUKTOR", "PERALATAN"]

# ── Feature set ──────────────────────────────────────────────────────────────
# Each feature maps a physical phenomenon to a discriminating signal:
#   fault_duration_ms      PETIR = brief (20-100ms), KONDUKTOR = long (>200ms)
#   fault_count            PETIR = 1, LAYANG/KONDUKTOR may repeat
#   peak_fault_current_a   magnitude; log-scaled for dynamic range
#   di_dt_max              wavefront steepness; lightning = very fast; log-scaled
#   i0_i1_ratio            zero-seq dominance → ground fault (HEWAN, PETIR)
#   thd_percent            harmonic distortion; less relevant for lightning
#   inception_angle_deg    lightning strikes at voltage peak (~90°)
#   voltage_sag_depth_pu   severity of voltage dip
#   voltage_phase_ratio_spread_pu  phase-to-phase voltage asymmetry during fault
#   healthy_phase_voltage_ratio    whether one phase stayed nearly healthy
#   v2_v1_ratio          negative-sequence voltage unbalance
#   voltage_thd_max_percent  voltage waveform distortion in the early fault window
#   reclose_enc            0=failed, 0.5=not attempted/unknown, 1=successful
#   is_ground_enc          1 if ground fault, 0 if phase fault
#   trip_type_enc          0=unknown, 1=single_pole, 2=three_pole
#   phase_count            number of faulted phases (1, 2, or 3)
#   zone_enc               distance zone (1, 2, 3, 0=unknown)
from models.feature_schema import (
    FEATURE_COLS, FEATURE_VERSION, build_feature_frame, encode_reclose,
    encode_trip_type, encode_zone, parse_phase_count,
)


# ── Tier 1 rule check (must stay in sync with rules.py) ─────────────────────

def is_tier1_handled(row) -> bool:
    """Return True if a Tier 1 rule fires — exclude from Tier 2 training."""
    from models.rules import apply_rules
    values = {k: v for k, v in dict(row).items() if not (isinstance(v, float) and np.isnan(v))}
    for key in ("reclose_successful", "ct_anomaly_detected", "soe_phase_mismatch"):
        value = values.get(key)
        if value in ("True", "true"):
            values[key] = True
        elif value in ("False", "false"):
            values[key] = False
    return apply_rules(values) is not None



# ── Feature engineering helpers ──────────────────────────────────────────────

def load_and_prepare(csv_path: Path):
    df = pd.read_csv(csv_path)

    # Quality filter
    df = df[df["scaling_ok"].astype(str).str.lower() == "true"].copy()
    df = df[df["duration_ok"].astype(str).str.lower() == "true"].copy()

    # Keep only known classes
    df = df[df["label"].isin(ALL_CLASSES)].copy()

    # Exclude rows already handled by Tier 1
    df = df[~df.apply(is_tier1_handled, axis=1)].copy()

    print(f"After quality + Tier-1 filter: {len(df)} rows")
    counts = df["label"].value_counts()
    for cls in ALL_CLASSES:
        n = counts.get(cls, 0)
        bar = "#" * n + "." * max(0, 40 - n)
        print(f"  {cls:<15} {n:>4}  {bar[:40]}")
    print()

    # ── Engineered features ──────────────────────────────────────────────────
    X = build_feature_frame(df.to_dict("records"))
    X.index = df.index
    return X, df["label"], df



def _balanced_class_weight(y_series: pd.Series) -> dict:
    """Compute sklearn-like balanced class weights from current training labels."""
    counts = y_series.value_counts().to_dict()
    n_samples = float(len(y_series))
    n_classes = float(len(counts)) if counts else 1.0
    return {cls: n_samples / (n_classes * float(cnt)) for cls, cnt in counts.items() if cnt > 0}


def build_class_weight(
    y_series: pd.Series,
    focus_transient_lines: bool = True,
    petir_multiplier: float = 0.80,
    non_petir_transient_multiplier: float = 1.35,
) -> dict:
    """
    Build class weights with optional transmission-line transient focus.

    Starting from balanced weights, we down-weight PETIR slightly and up-weight
    LAYANG/HEWAN/BENDA_ASING to reduce PETIR over-call when signatures overlap.
    """
    weights = _balanced_class_weight(y_series)
    if not focus_transient_lines:
        return weights

    if "PETIR" in weights:
        weights["PETIR"] *= float(petir_multiplier)
    for cls in ("LAYANG", "HEWAN", "BENDA_ASING"):
        if cls in weights:
            weights[cls] *= float(non_petir_transient_multiplier)
    return weights


# ── Training ─────────────────────────────────────────────────────────────────

def train(
    csv_path: Path = FEATURES_CSV,
    model_out: Path = MODEL_OUT,
    focus_transient_lines: bool = True,
    petir_multiplier: float = 0.80,
    non_petir_transient_multiplier: float = 1.35,
):
    from models.retrain import evaluate_and_train
    return evaluate_and_train(
        csv_path, model_out,
        focus_transient_lines=focus_transient_lines,
        petir_multiplier=petir_multiplier,
        non_petir_transient_multiplier=non_petir_transient_multiplier,
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train Tier-2 multiclass model for transmission-line fault causes."
    )
    parser.add_argument("--csv", type=Path, default=FEATURES_CSV, help="Path to rebuilt labeled_features_v2.csv")
    parser.add_argument("--out", type=Path, default=MODEL_OUT, help="Output model path")
    parser.add_argument(
        "--no-focus-transient-lines",
        action="store_true",
        help="Disable extra up-weighting for non-PETIR transient classes.",
    )
    parser.add_argument(
        "--petir-multiplier",
        type=float,
        default=0.80,
        help="Multiplier applied to PETIR class weight (default: 0.80).",
    )
    parser.add_argument(
        "--non-petir-transient-multiplier",
        type=float,
        default=1.35,
        help="Multiplier applied to LAYANG/HEWAN/BENDA_ASING class weights (default: 1.35).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        csv_path=args.csv,
        model_out=args.out,
        focus_transient_lines=not args.no_focus_transient_lines,
        petir_multiplier=args.petir_multiplier,
        non_petir_transient_multiplier=args.non_petir_transient_multiplier,
    )
