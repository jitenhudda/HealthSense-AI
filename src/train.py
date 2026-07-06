"""
Automated training pipeline for HealthSense AI.

Every CSV inside data/ is treated as one disease dataset. For each dataset
this module will:
    1. Auto-detect the target column and feature columns.
    2. Build a preprocessing pipeline (imputation + scaling / encoding).
    3. Train Logistic Regression, Random Forest and XGBoost in parallel
       (joblib.Parallel) and evaluate them on a held-out test split.
    4. Persist only the best-performing pipeline (preprocessor + model)
       plus a metadata JSON file describing the schema, used later by the
       Streamlit app to render forms and run predictions.

Run directly:  python -m src.train
"""

import glob
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import DATA_DIR, MODEL_DIR, TARGET_KEYWORDS

warnings.filterwarnings("ignore")


def detect_target(df: pd.DataFrame) -> str:
    """Find the target column: prefer known keywords, else fall back to
    the last binary-ish column, else the last column."""
    for col in df.columns:
        if col.strip().lower() in TARGET_KEYWORDS:
            return col
    for col in reversed(df.columns.tolist()):
        if df[col].nunique() <= 2:
            return col
    return df.columns[-1]


def build_preprocessor(df: pd.DataFrame, feature_cols: list):
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    preprocessor = ColumnTransformer(transformers)
    return preprocessor, numeric_cols, categorical_cols


def _fit_and_score(name, model, preprocessor, X_train, y_train, X_test, y_test):
    """Fit a single candidate pipeline and return its evaluation metrics.
    Designed to be safely called inside joblib.Parallel worker processes."""
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    auc = acc
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X_test)
            if proba.shape[1] == 2 and y_test.nunique() > 1:
                auc = roc_auc_score(y_test, proba[:, 1])
        except Exception:
            pass

    metrics = {"accuracy": float(acc), "f1_score": float(f1), "roc_auc": float(auc)}
    return name, pipe, metrics


def _is_up_to_date(disease_name: str, csv_path: str) -> bool:
    """True if a trained model already exists and is newer than its CSV,
    so retraining can be skipped for speed."""
    model_path = os.path.join(MODEL_DIR, f"{disease_name}.joblib")
    meta_path = os.path.join(MODEL_DIR, f"{disease_name}_meta.json")
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        return False
    return os.path.getmtime(model_path) >= os.path.getmtime(csv_path)


def train_disease(csv_path: str, force: bool = False) -> dict:
    disease_name = os.path.splitext(os.path.basename(csv_path))[0].lower()

    if not force and _is_up_to_date(disease_name, csv_path):
        with open(os.path.join(MODEL_DIR, f"{disease_name}_meta.json")) as f:
            meta = json.load(f)
        return {
            "disease": disease_name, "best_model": meta["best_model"],
            "metrics": meta["metrics"], "skipped": True,
        }

    start = time.time()

    df = pd.read_csv(csv_path)
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.contains("^unnamed", case=False)]

    target_col = detect_target(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != target_col]
    y_raw = df[target_col]
    if y_raw.dtype == object or str(y_raw.dtype) == "category":
        y = y_raw.astype("category").cat.codes
        class_labels = list(y_raw.astype("category").cat.categories)
    else:
        y = y_raw.astype(int)
        class_labels = sorted(y.unique().tolist())

    X = df[feature_cols]
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(df, feature_cols)

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, n_jobs=-1),
        "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            eval_metric="logloss", random_state=42, n_jobs=-1,
            tree_method="hist",
        ),
    }

    # Train all candidate models in parallel for fast experimentation.
    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(_fit_and_score)(name, model, preprocessor, X_train, y_train, X_test, y_test)
        for name, model in candidates.items()
    )

    best_name, best_pipe, best_metrics = max(results, key=lambda r: r[2]["roc_auc"])

    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(best_pipe, os.path.join(MODEL_DIR, f"{disease_name}.joblib"))

    meta = {
        "disease": disease_name,
        "display_name": disease_name.replace("_", " ").title(),
        "target": target_col,
        "class_labels": [str(c) for c in class_labels],
        "features": feature_cols,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "best_model": best_name,
        "metrics": best_metrics,
        "all_results": {n: m for n, _, m in results},
        "feature_ranges": {
            c: {
                "min": float(df[c].min()),
                "max": float(df[c].max()),
                "mean": float(df[c].mean()),
                "is_binary": bool(df[c].dropna().nunique() <= 2),
            }
            for c in numeric_cols
        },
        "feature_categories": {
            c: sorted(df[c].dropna().astype(str).unique().tolist())
            for c in categorical_cols
        },
        "n_samples": int(len(df)),
        "n_features": len(feature_cols),
        "n_classes": int(y.nunique()),
        "training_seconds": round(time.time() - start, 2),
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(MODEL_DIR, f"{disease_name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return {"disease": disease_name, "best_model": best_name, "metrics": best_metrics}


def remove_orphan_models() -> list:
    """Delete any trained model/metadata whose source CSV no longer exists
    in data/. Keeps the model registry in sync with the dataset folder so
    deleted diseases immediately disappear from the app without a retrain."""
    if not os.path.isdir(MODEL_DIR):
        return []
    valid_names = {
        os.path.splitext(os.path.basename(p))[0].lower()
        for p in glob.glob(os.path.join(DATA_DIR, "*.csv"))
    }
    removed = []
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".joblib"):
            name = fname[: -len(".joblib")]
        elif fname.endswith("_meta.json"):
            name = fname[: -len("_meta.json")]
        else:
            continue
        if name not in valid_names:
            os.remove(os.path.join(MODEL_DIR, fname))
            removed.append(name)
    return sorted(set(removed))


def train_all(progress_callback=None, force: bool = False) -> list:
    """Train a model for every CSV found in the data directory. Datasets
    that haven't changed since their last successful training run are
    skipped automatically unless force=True.

    progress_callback, if provided, is called as
    progress_callback(current_index, total, disease_name) before each fit.
    """
    remove_orphan_models()
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    summary = []
    for i, csv_path in enumerate(csv_files, start=1):
        disease_name = os.path.splitext(os.path.basename(csv_path))[0]
        if progress_callback:
            progress_callback(i, len(csv_files), disease_name)
        try:
            summary.append(train_disease(csv_path, force=force))
        except Exception as exc:  # keep training other diseases even if one fails
            summary.append({"disease": disease_name, "error": str(exc)})
    return summary


if __name__ == "__main__":
    for r in train_all():
        print(r)
