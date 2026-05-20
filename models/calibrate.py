"""
Probability Calibration
=======================
Fit a probability calibrator on a held-out validation split, separately
from the main training run. Produces ``models/proba_calibrator.pkl``.

The webapp inference path in ``webapp/api/ml_predict.py`` prefers this
fitted calibrator over the default temperature scaling (T=1.5) whenever
the pickle is present, so the calibrator can be regenerated independently
of the classifier (no need to re-train LightGBM).

Methods
-------
- ``sigmoid`` (Platt scaling): logistic regression on logits. Robust on small
  validation sets (~50–200 samples per class) and fast. Default.
- ``isotonic``: non-parametric monotonic regression. More flexible but needs
  meaningfully more held-out data (~300+ samples per class) to avoid
  overfitting. Use only when classes are well-represented.

Usage
-----
    python models/calibrate.py
    python models/calibrate.py --method isotonic --test-size 0.25
    python models/calibrate.py --features data/features/labeled_features_87l.csv

Notes
-----
- Reads the same labeled_features.csv that ``models/train.py`` consumes.
- Uses ``CalibratedClassifierCV(cv='prefit')`` so the underlying LightGBM
  is never re-trained — only the calibrator is fitted on the validation
  fold. This preserves the model artefact's identity (SHA-256 unchanged).
- Output bundle: ``{calibrator, classes_, method, fitted_at_utc, n_samples,
  classifier_sha, training_features}``.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_PIPELINE_DIR = Path(__file__).parent.parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from models.predict import _build_feature_vector
from models.train import FEATURE_COLS

warnings.filterwarnings("ignore")

DEFAULT_FEATURES_CSV = Path(__file__).parent.parent / "data" / "features" / "labeled_features.csv"
DEFAULT_MODEL_PATH = Path(__file__).parent / "fault_classifier.pkl"
DEFAULT_OUT_PATH = Path(__file__).parent / "proba_calibrator.pkl"


def _row_to_dict(series: pd.Series) -> dict:
    return {col: series.get(col) for col in series.index}


def _build_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    rows = []
    for _, series in df.iterrows():
        vec = _build_feature_vector(_row_to_dict(series), feature_cols)
        rows.append(vec.reshape(-1))
    return np.vstack(rows)


def fit_and_save(
    features_csv: Path = DEFAULT_FEATURES_CSV,
    model_path: Path = DEFAULT_MODEL_PATH,
    out_path: Path = DEFAULT_OUT_PATH,
    method: str = "sigmoid",
    test_size: float = 0.20,
    label_col: str = "label",
    random_state: int = 42,
) -> None:
    if not features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained classifier not found: {model_path} — run models/train.py first")
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError(f"Unknown method '{method}' (use 'sigmoid' or 'isotonic')")

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    clf = bundle["clf"]
    feature_cols = list(bundle.get("feature_cols") or FEATURE_COLS)
    trained_classes = list(getattr(clf, "classes_", bundle.get("classes", [])))

    df = pd.read_csv(features_csv)
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in {features_csv} (cols: {list(df.columns)[:6]}…)")

    df = df.dropna(subset=[label_col])
    df = df[df[label_col].isin(trained_classes)]
    if df.empty:
        raise RuntimeError("After filtering to trained classes the dataset is empty — check label column values.")

    y = df[label_col].astype(str).values
    X = _build_matrix(df, feature_cols)

    # Stratified hold-out — calibrator fits only on the held-out fold so the
    # base classifier never sees these rows at calibration time.
    X_train_unused, X_holdout, y_train_unused, y_holdout = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )

    print(f"Loaded {len(df)} labeled rows. Held-out fold for calibration: {len(X_holdout)} rows.")
    counts = pd.Series(y_holdout).value_counts().to_dict()
    print(f"Held-out class distribution: {counts}")

    if method == "isotonic":
        min_per_class = min(counts.values()) if counts else 0
        if min_per_class < 30:
            print(f"⚠ Warning: isotonic method ideally needs >=30 samples per class, "
                  f"got min={min_per_class}. Consider --method sigmoid.")

    calibrator = CalibratedClassifierCV(estimator=clf, cv="prefit", method=method)
    calibrator.fit(X_holdout, y_holdout)

    # Quick before/after diagnostic on the held-out fold.
    raw_proba = clf.predict_proba(X_holdout)
    cal_proba = calibrator.predict_proba(X_holdout)
    raw_top = raw_proba.max(axis=1).mean()
    cal_top = cal_proba.max(axis=1).mean()
    raw_acc = (np.array(trained_classes)[raw_proba.argmax(axis=1)] == y_holdout).mean()
    cal_acc = (np.array(list(calibrator.classes_))[cal_proba.argmax(axis=1)] == y_holdout).mean()
    print(f"Mean top-1 confidence  raw: {raw_top:.3f}   calibrated: {cal_top:.3f}")
    print(f"Held-out accuracy      raw: {raw_acc:.3f}   calibrated: {cal_acc:.3f}")

    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
    out_bundle = {
        "calibrator": calibrator,
        "classes_": list(calibrator.classes_),
        "method": method,
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(X_holdout)),
        "classifier_sha": sha,
        "training_features": feature_cols,
        "holdout_class_counts": counts,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out_bundle, f)
    print(f"Calibrator saved -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Fit a held-out probability calibrator for fault_classifier.pkl")
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES_CSV, help="Path to labeled_features.csv")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to fault_classifier.pkl")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="Where to save proba_calibrator.pkl")
    p.add_argument("--method", choices=("sigmoid", "isotonic"), default="sigmoid", help="Calibration method")
    p.add_argument("--test-size", type=float, default=0.20, help="Fraction held out for calibrator fitting")
    p.add_argument("--label-col", default="label", help="Column name holding the cause label")
    p.add_argument("--seed", type=int, default=42, help="Random state for reproducible splits")
    args = p.parse_args()
    fit_and_save(
        features_csv=args.features, model_path=args.model, out_path=args.out,
        method=args.method, test_size=args.test_size, label_col=args.label_col, random_state=args.seed,
    )


if __name__ == "__main__":
    main()
