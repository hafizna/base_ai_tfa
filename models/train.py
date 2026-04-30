"""
Multi-class Fault Cause Classifier
====================================
Trains a LightGBM classifier on labeled_features.csv to classify the
physical cause of transmission line faults into 7 categories:

    PETIR       — lightning (direct strike or induced overvoltage)
    LAYANG      — kite (layang-layang)
    POHON       — tree / vegetation contact
    HEWAN       — animal (ular, binatang, burung, babi, tikus, etc.)
    BENDA_ASING — non-living foreign object (aluminium foil, terpal, etc.)
    KONDUKTOR   — conductor / tower structural failure
    PERALATAN   — equipment / protection / telecom-origin failure

Design rationale
----------------
Promoted from pipeline-lgbm after 3-way CV showed LightGBM outperforms
Random Forest on the metrics that matter for imbalanced multi-class:

    F1 macro    0.407 (LGBM) vs 0.352 (RF)   — primary metric
    F1 weighted 0.757 (LGBM) vs 0.738 (RF)
    Accuracy    0.778 (LGBM) vs 0.800 (RF)   — less relevant here

Previous design was a binary PETIR vs rest tree — only viable with the
old tiny dataset (83 rows, 84% PETIR). With the expanded dataset
(~450+ rows across 7 classes), LightGBM is now used as the main Tier 2
multiclass model while Random Forest remains the comparison baseline.

  - Tier 1 deterministic rules are still applied first (rules.py) for
    high-confidence KONDUKTOR and failed-reclose cases.
  - Tier 2 (this model) handles the remaining ambiguous events.
  - LightGBM with class_weight='balanced' handles imbalance natively
    (no SMOTE needed).
  - Features cover waveform physics (di/dt, peak current, THD, inception
    angle), fault context (duration, ground ratio, zone, reclose), and
    protection metadata (trip type, phase count).
  - Stratified 5-fold cross-validation with per-class F1 report.
  - Model is saved as pipeline/models/fault_classifier.pkl (joblib/pickle).

For ongoing RF vs LGBM comparison as data grows, run compare_models.py
from the project root — it always benchmarks RF as a standalone baseline.

Run from the pipeline/ directory:
    python models/train.py
"""

import sys
import argparse
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

FEATURES_CSV = Path(__file__).parent.parent / "data" / "features" / "labeled_features.csv"
MODEL_OUT    = Path(__file__).parent / "fault_classifier.pkl"
# Keep old binary model path so existing webapp code still loads during transition
MODEL_OUT_LEGACY = Path(__file__).parent / "petir_tree.pkl"

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
FEATURE_COLS = [
    "fault_duration_ms",
    "fault_count",
    "peak_fault_current_a",      # log-scaled in prep
    "di_dt_max",                  # log-scaled in prep
    "i0_i1_ratio",
    "thd_percent",
    "inception_angle_degrees",
    "voltage_sag_depth_pu",
    "voltage_phase_ratio_spread_pu",
    "healthy_phase_voltage_ratio",
    "v2_v1_ratio",
    "voltage_thd_max_percent",
    "reclose_enc",
    "is_ground_enc",
    "trip_type_enc",
    "phase_count",
    "zone_enc",
]


# ── Tier 1 rule check (must stay in sync with rules.py) ─────────────────────

def is_tier1_handled(row) -> bool:
    """Return True if a Tier 1 rule fires — exclude from Tier 2 training."""
    fault_count    = int(row.get("fault_count", 1))
    faulted_phases = str(row.get("faulted_phases", ""))
    duration_ms    = float(row.get("fault_duration_ms", 0))
    reclose_ok     = row.get("reclose_successful")
    trip_type      = str(row.get("trip_type", ""))
    peak_i         = float(row.get("peak_fault_current_a", 0) or 0)

    phase_count       = faulted_phases.count("+") + 1 if faulted_phases else 1
    reclose_failed    = (reclose_ok is False or str(reclose_ok) == "False")
    reclose_succeeded = (reclose_ok is True  or str(reclose_ok) == "True")

    if (fault_count >= 2 and fault_count <= 20 and phase_count == 2
            and duration_ms > 80 and not reclose_succeeded):
        return True
    if reclose_failed and trip_type == "three_pole" and peak_i > 50:
        return True
    if reclose_failed and duration_ms > 10 and peak_i > 100:
        return True
    return False


# ── Feature engineering helpers ──────────────────────────────────────────────

def encode_reclose(val) -> float:
    if val is True or str(val).lower() == "true":
        return 1.0
    if val is False or str(val).lower() == "false":
        return 0.0
    return 0.5


def encode_trip_type(val) -> int:
    s = str(val).lower()
    if "single" in s or "1" in s:
        return 1
    if "three" in s or "3" in s:
        return 2
    return 0


def encode_zone(val) -> int:
    s = str(val).upper().strip()
    for z in ("Z1", "Z2", "Z3"):
        if z in s:
            return int(z[1])
    return 0


def parse_phase_count(val) -> int:
    s = str(val)
    if not s or s == "nan":
        return 1
    return s.count("+") + 1


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
    df["reclose_enc"]     = df["reclose_successful"].apply(encode_reclose)
    df["is_ground_enc"]   = df["is_ground_fault"].astype(str).str.lower().map(
                                {"true": 1, "false": 0}).fillna(0).astype(int)
    df["trip_type_enc"]   = df["trip_type"].apply(encode_trip_type)
    df["phase_count"]     = df["faulted_phases"].apply(parse_phase_count)
    df["zone_enc"]        = df["zone_operated"].apply(encode_zone)

    # Log-scale large-range features
    df["di_dt_max"]            = np.log1p(df["di_dt_max"].fillna(0).clip(lower=0))
    df["peak_fault_current_a"] = np.log1p(df["peak_fault_current_a"].fillna(0).clip(lower=0))

    # Backward compatibility: older CSVs may not yet have the newest features.
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(df[col].median())

    y = df["label"]
    X = df[FEATURE_COLS]
    return X, y, df


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
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found — run batch_extract.py first")
        sys.exit(1)

    X, y, df = load_and_prepare(csv_path)
    class_counts = y.value_counts()

    # Need at least 2 samples per class for stratified CV
    trainable = [c for c in ALL_CLASSES if class_counts.get(c, 0) >= 2]
    sparse    = [c for c in ALL_CLASSES if 0 < class_counts.get(c, 0) < 2]
    missing   = [c for c in ALL_CLASSES if class_counts.get(c, 0) == 0]

    if sparse:
        print(f"WARNING: classes with only 1 sample (excluded from CV): {sparse}")
    if missing:
        print(f"WARNING: classes with no samples (model cannot predict): {missing}")

    mask = y.isin(trainable)
    X_tr, y_tr = X[mask], y[mask]

    print(f"Training on {len(X_tr)} rows across {len(trainable)} classes: {trainable}\n")

    class_weight = build_class_weight(
        y_tr,
        focus_transient_lines=focus_transient_lines,
        petir_multiplier=petir_multiplier,
        non_petir_transient_multiplier=non_petir_transient_multiplier,
    )
    print("Class weight (effective):")
    for cls in sorted(class_weight):
        print(f"  {cls:<15} {class_weight[cls]:.3f}")
    print()

    clf = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    # Stratified CV — adapt n_splits to smallest class size
    min_class = min(y_tr.value_counts())
    n_splits  = max(2, min(5, min_class))

    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_results = cross_validate(
            clf, X_tr, y_tr, cv=cv,
            scoring=["f1_macro", "f1_weighted", "accuracy"],
            return_train_score=False,
        )
        print(f"Cross-validation ({n_splits}-fold):")
        print(f"  accuracy     {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}")
        print(f"  F1 macro     {cv_results['test_f1_macro'].mean():.3f} ± {cv_results['test_f1_macro'].std():.3f}")
        print(f"  F1 weighted  {cv_results['test_f1_weighted'].mean():.3f} ± {cv_results['test_f1_weighted'].std():.3f}")
        print()
    else:
        print("Too few samples per class for cross-validation — skipping CV\n")

    # Full-data fit
    clf.fit(X_tr, y_tr)

    # Training-set report (optimistic — shows model capacity, not generalisation)
    y_pred = clf.predict(X_tr)
    print("Training-set classification report:")
    print(classification_report(y_tr, y_pred, labels=trainable, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_tr, y_pred, labels=trainable)
    print("Confusion matrix (rows=actual, cols=predicted):")
    header = "".join(f"{c[:6]:>8}" for c in trainable)
    print(f"{'':>12}{header}")
    for i, cls in enumerate(trainable):
        row = "".join(f"{cm[i,j]:>8}" for j in range(len(trainable)))
        print(f"  {cls:<12}{row}")
    print()

    # Feature importances
    print("Feature importances:")
    for feat, imp in sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1]):
        bar = "#" * int(imp * 40)
        print(f"  {feat:<30} {imp:.3f}  {bar}")
    print()

    # Save
    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "clf": clf,
        "feature_cols": FEATURE_COLS,
        "classes": list(getattr(clf, "classes_", trainable)),
        "all_classes": ALL_CLASSES,
        "class_counts": dict(class_counts),
        "model_type": "multiclass_lightgbm",
        "training_profile": {
            "focus_transient_lines": bool(focus_transient_lines),
            "petir_multiplier": float(petir_multiplier),
            "non_petir_transient_multiplier": float(non_petir_transient_multiplier),
            "class_weight": class_weight,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    with open(model_out, "wb") as f:
        pickle.dump(payload, f)
    print(f"Model saved -> {model_out}")

    # Also write to legacy path so webapp still works until it's updated
    with open(MODEL_OUT_LEGACY, "wb") as f:
        pickle.dump(payload, f)
    print(f"Legacy path -> {MODEL_OUT_LEGACY}")

    return clf


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train Tier-2 multiclass model for transmission-line fault causes."
    )
    parser.add_argument("--csv", type=Path, default=FEATURES_CSV, help="Path to labeled_features.csv")
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
