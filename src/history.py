"""
Patient Report and Doctor Report persistence for HealthSense AI.

Patient Reports and Doctor Reports are stored SEPARATELY (two different
CSV files) as required by the product spec. A Doctor Report always links
back to the Patient Report it extends via `patient_report_id`, but the two
stores are independent — deleting/clearing one never touches the other.
"""

import datetime
import os
import uuid

import pandas as pd

from src.config import DOCTOR_REPORTS_FILE, HISTORY_DIR, PATIENT_REPORTS_FILE

PATIENT_REPORT_COLUMNS = [
    "patient_report_id", "timestamp", "disease", "patient_name", "age", "gender",
    "risk_percent", "confidence_percent", "risk_level", "prediction",
    "model_used", "recommended_specialist", "appointment_suggestion", "status",
]

DOCTOR_REPORT_COLUMNS = [
    "doctor_report_id", "patient_report_id", "timestamp", "disease", "patient_name",
    "ai_risk_percent", "ai_confidence_percent", "ai_risk_level",
    "diagnostic_tests_suggested", "test_observations", "final_diagnosis",
    "demo_prescription", "lifestyle_advice", "follow_up_date", "doctor_notes",
]


def _read_history_frame(file_path: str, expected_columns: list) -> pd.DataFrame:
    """Load a history CSV safely, skipping malformed lines and preserving any extra columns."""
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=expected_columns)

    try:
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(file_path, engine="python", on_bad_lines="skip", dtype=str, keep_default_na=False)

    if df.empty:
        return pd.DataFrame(columns=expected_columns)

    for col in expected_columns:
        if col not in df.columns:
            df[col] = ""

    extra_cols = [c for c in df.columns if c not in expected_columns]
    ordered_cols = expected_columns + extra_cols
    return df.reindex(columns=ordered_cols, fill_value="")


def _write_history_frame(file_path: str, df: pd.DataFrame, expected_columns: list) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    ordered = [col for col in expected_columns if col in df.columns]
    ordered.extend([col for col in df.columns if col not in expected_columns])
    df = df.reindex(columns=ordered, fill_value="")
    df.to_csv(file_path, index=False)


# ----------------------------------------------------------------------------
# PATIENT REPORTS
# ----------------------------------------------------------------------------

def save_patient_report(disease: str, patient_name: str, symptoms: dict, result: dict) -> dict:
    """Persist a new Patient Report (generated in Patient Mode) and return it
    as a dict, including the freshly-assigned patient_report_id."""
    os.makedirs(HISTORY_DIR, exist_ok=True)

    level = result["risk_level"]
    record = {
        "patient_report_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "disease": disease,
        "patient_name": patient_name or "Anonymous Patient",
        "age": symptoms.get("_age", ""),
        "gender": symptoms.get("_gender", ""),
        "risk_percent": round(result["risk"] * 100, 2),
        "confidence_percent": round(result["confidence"] * 100, 2),
        "risk_level": level,
        "prediction": "High Risk" if result["prediction"] == 1 else "Low Risk",
        "model_used": result["model_used"],
        "recommended_specialist": result["recommended_specialist"],
        "appointment_suggestion": result["appointment_suggestion"],
        "status": "Pending Doctor Review",
    }
    # Store the plain-English symptom answers (labels only — never raw ML
    # feature/column names) so they can be shown back in the report.
    record.update({f"symptom_{k}": v for k, v in symptoms.items() if not k.startswith("_")})

    existing_df = _read_history_frame(PATIENT_REPORTS_FILE, PATIENT_REPORT_COLUMNS)
    new_df = pd.concat([existing_df, pd.DataFrame([record])], ignore_index=True)
    _write_history_frame(PATIENT_REPORTS_FILE, new_df, PATIENT_REPORT_COLUMNS)
    return record


def load_patient_reports() -> pd.DataFrame:
    df = _read_history_frame(PATIENT_REPORTS_FILE, PATIENT_REPORT_COLUMNS)
    for col in ["risk_percent", "confidence_percent"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.sort_values("timestamp", ascending=False)


def get_patient_report(patient_report_id: str) -> dict | None:
    df = load_patient_reports()
    match = df[df["patient_report_id"] == patient_report_id]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def mark_patient_report_reviewed(patient_report_id: str) -> None:
    if not os.path.exists(PATIENT_REPORTS_FILE):
        return
    df = _read_history_frame(PATIENT_REPORTS_FILE, PATIENT_REPORT_COLUMNS)
    df.loc[df["patient_report_id"] == patient_report_id, "status"] = "Reviewed by Doctor"
    _write_history_frame(PATIENT_REPORTS_FILE, df, PATIENT_REPORT_COLUMNS)


# ----------------------------------------------------------------------------
# DOCTOR REPORTS
# ----------------------------------------------------------------------------

def save_doctor_report(patient_report: dict, review: dict) -> dict:
    """Persist a new Doctor Report that extends a given Patient Report."""
    os.makedirs(HISTORY_DIR, exist_ok=True)

    record = {
        "doctor_report_id": uuid.uuid4().hex[:12],
        "patient_report_id": patient_report["patient_report_id"],
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "disease": patient_report["disease"],
        "patient_name": patient_report["patient_name"],
        "ai_risk_percent": patient_report["risk_percent"],
        "ai_confidence_percent": patient_report["confidence_percent"],
        "ai_risk_level": patient_report["risk_level"],
        "diagnostic_tests_suggested": "; ".join(review.get("tests_ordered", [])),
        "test_observations": review.get("test_observations", ""),
        "final_diagnosis": review.get("final_diagnosis", ""),
        "demo_prescription": review.get("demo_prescription", ""),
        "lifestyle_advice": review.get("lifestyle_advice", ""),
        "follow_up_date": review.get("follow_up_date", ""),
        "doctor_notes": review.get("doctor_notes", ""),
    }

    existing_df = _read_history_frame(DOCTOR_REPORTS_FILE, DOCTOR_REPORT_COLUMNS)
    new_df = pd.concat([existing_df, pd.DataFrame([record])], ignore_index=True)
    _write_history_frame(DOCTOR_REPORTS_FILE, new_df, DOCTOR_REPORT_COLUMNS)

    mark_patient_report_reviewed(patient_report["patient_report_id"])
    return record


def load_doctor_reports() -> pd.DataFrame:
    df = _read_history_frame(DOCTOR_REPORTS_FILE, DOCTOR_REPORT_COLUMNS)
    for col in ["ai_risk_percent", "ai_confidence_percent"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.sort_values("timestamp", ascending=False)


def clear_all_reports() -> None:
    for f in (PATIENT_REPORTS_FILE, DOCTOR_REPORTS_FILE):
        if os.path.exists(f):
            os.remove(f)
