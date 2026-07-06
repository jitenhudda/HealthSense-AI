"""Model loading and inference utilities for HealthSense AI."""

import json
import os

import pandas as pd
import streamlit as st
from joblib import load

from src.config import (
    MODEL_DIR, get_appointment_suggestion, get_diagnostic_tests, get_specialist,
    risk_level,
)


def list_diseases() -> list:
    """Return sorted disease keys that currently have a trained model."""
    if not os.path.isdir(MODEL_DIR):
        return []
    return sorted({
        f[:-len(".joblib")]
        for f in os.listdir(MODEL_DIR)
        if f.endswith(".joblib")
    })


@st.cache_resource(show_spinner=False)
def load_model(disease: str):
    return load(os.path.join(MODEL_DIR, f"{disease}.joblib"))


@st.cache_data(show_spinner=False)
def load_meta(disease: str) -> dict:
    with open(os.path.join(MODEL_DIR, f"{disease}_meta.json")) as f:
        return json.load(f)


def build_default_inputs(meta: dict) -> dict:
    """Sensible auto-fill values for features not shown in Patient mode:
    the training-set mean for numeric columns, the most common category
    for categorical ones."""
    defaults = {}
    for feature, info in meta.get("feature_ranges", {}).items():
        if info.get("is_binary"):
            defaults[feature] = 0
        else:
            mean = info.get("mean", 0)
            min_value = info.get("min", mean)
            max_value = info.get("max", mean)
            is_int = float(min_value).is_integer() and float(max_value).is_integer()
            defaults[feature] = int(round(mean)) if is_int else round(float(mean), 1)
    for feature, categories in meta.get("feature_categories", {}).items():
        if categories:
            defaults[feature] = categories[0]
    return defaults


def _coerce_feature_value(value, feature: str, meta: dict):
    if value is None:
        return None
    if pd.isna(value):
        return None
    if feature in meta.get("categorical_features", []):
        categories = meta.get("feature_categories", {}).get(feature, [])
        if categories:
            text = str(value).strip()
            for category in categories:
                if str(category).strip().lower() == text.lower():
                    return category
            if text:
                return categories[0]
        return value

    try:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.lower() in {"", "none", "na", "n/a", "null"}:
                return None
            if cleaned.lower() in {"yes", "true", "y"}:
                return 1
            if cleaned.lower() in {"no", "false", "n"}:
                return 0
            return float(cleaned)
        return float(value)
    except (TypeError, ValueError):
        return None


def predict(disease: str, input_dict: dict) -> dict:
    """Run inference for a single patient record. Returns risk probability,
    model confidence and the binary prediction. Designed to complete in
    well under a second thanks to cached model loading."""
    model = load_model(disease)
    meta = load_meta(disease)

    defaults = build_default_inputs(meta)
    row = {}
    for feature in meta["features"]:
        raw_value = input_dict.get(feature)
        if raw_value is None:
            raw_value = defaults.get(feature)
        row[feature] = _coerce_feature_value(raw_value, feature, meta)

    for feature, value in row.items():
        if value is None:
            row[feature] = defaults.get(feature)

    df = pd.DataFrame([row])

    proba = model.predict_proba(df)[0]
    if len(proba) > 1:
        risk = float(proba[1])
    else:
        risk = float(proba[0])
    confidence = float(max(proba))
    level = risk_level(risk)

    return {
        "risk": risk,
        "confidence": confidence,
        "prediction": int(risk >= 0.5),
        "model_used": meta["best_model"],
        "risk_level": level,
        "recommended_specialist": get_specialist(disease),
        "appointment_suggestion": get_appointment_suggestion(level),
        "suggested_tests": get_diagnostic_tests(disease),
    }
