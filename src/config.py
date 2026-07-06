"""Central configuration and shared constants for HealthSense AI."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
HISTORY_DIR = os.path.join(DATA_DIR, "history")
PATIENT_REPORTS_FILE = os.path.join(HISTORY_DIR, "patient_reports.csv")
DOCTOR_REPORTS_FILE = os.path.join(HISTORY_DIR, "doctor_reports.csv")

TARGET_KEYWORDS = [
    "target", "outcome", "diagnosis", "result", "label",
    "class", "disease", "status", "risk",
]

# Keyword -> UI group mapping used to auto-organize prediction form inputs.
GROUP_KEYWORDS = {
    "Patient Info": [
        "age", "gender", "sex", "height", "weight", "bmi", "pregnan",
    ],
    "Lifestyle": [
        "smok", "alcohol", "exercise", "activity", "diet", "sleep", "physical",
    ],
    "Medical History": [
        "family", "history", "hypertension", "heart", "chest", "stroke",
        "medication", "surgery", "condition", "disease",
    ],
    "Lab Values": [
        "glucose", "cholesterol", "pressure", "bp", "insulin", "hba1c",
        "creatinine", "triglyceride", "hemoglobin", "count", "level",
        "rate", "score", "sugar",
    ],
}

GROUP_ORDER = ["Patient Info", "Lifestyle", "Medical History", "Lab Values"]

GROUP_ICONS = {
    "Patient Info": "🧍",
    "Lifestyle": "🏃",
    "Medical History": "📋",
    "Lab Values": "🧪",
}

# Ordered canonical questions shown in Patient (simple) mode. Each maps to a
# small set of keywords used to find the matching column in a disease's
# feature list. At most one feature is surfaced per category, keeping the
# patient-facing form to roughly 7-10 plain-English questions regardless of
# how many raw columns the underlying dataset actually has.
PATIENT_CATEGORIES = [
    ("Age", ["age"]),
    ("Gender", ["gender", "sex"]),
    ("Height", ["height"]),
    ("Weight / BMI", ["bmi", "weight"]),
    ("Smoking", ["smok"]),
    ("Alcohol", ["alcohol"]),
    ("Exercise / Activity", ["exercise", "activity", "physical"]),
    ("Family History", ["family"]),
    ("Blood Pressure", ["pressure", "bp", "hypertension"]),
    ("Blood Sugar", ["glucose", "sugar", "hba1c", "insulin"]),
]

DISEASE_QUESTIONNAIRES = {
    "breast_cancer": [
        {
            "id": "breast_lump",
            "question": "Have you noticed a new lump or thickening in a breast or under the arm?",
            "feature": "concavity_mean",
            "widget": "select",
            "options": ["No", "A small change", "A noticeable change"],
            "value_map": {"No": 0.0, "A small change": 0.25, "A noticeable change": 0.7},
        },
        {
            "id": "skin_change",
            "question": "Has the skin over the area become red, dimpled, or pulled inward?",
            "feature": "texture_mean",
            "widget": "select",
            "options": ["No", "Sometimes", "Yes"],
            "value_map": {"No": 0.0, "Sometimes": 0.3, "Yes": 0.75},
        },
        {
            "id": "nipple_change",
            "question": "Have you noticed any nipple discharge or unusual changes?",
            "feature": "compactness_mean",
            "widget": "select",
            "options": ["No", "Not sure", "Yes"],
            "value_map": {"No": 0.0, "Not sure": 0.2, "Yes": 0.8},
        },
        {
            "id": "pain",
            "question": "Is there any persistent pain or discomfort in the area?",
            "feature": "radius_mean",
            "widget": "select",
            "options": ["No", "Occasionally", "Often"],
            "value_map": {"No": 0.0, "Occasionally": 0.25, "Often": 0.7},
        },
        {
            "id": "family_history",
            "question": "Has a close family member ever had breast cancer?",
            "feature": "fractal_dimension_mean",
            "widget": "select",
            "options": ["No", "Not sure", "Yes"],
            "value_map": {"No": 0.0, "Not sure": 0.2, "Yes": 0.8},
        },
        {
            "id": "size_change",
            "question": "Has the area become noticeably larger or more obvious over time?",
            "feature": "area_mean",
            "widget": "select",
            "options": ["No", "Slightly", "Clearly"],
            "value_map": {"No": 0.0, "Slightly": 0.3, "Clearly": 0.75},
        },
        {
            "id": "swelling",
            "question": "Do you have any swelling, warmth, or hardness in the breast area?",
            "feature": "smoothness_mean",
            "widget": "select",
            "options": ["No", "Mild", "Moderate or more"],
            "value_map": {"No": 0.0, "Mild": 0.3, "Moderate or more": 0.7},
        },
    ],
    "diabetes": [
        {
            "id": "age",
            "question": "How old are you?",
            "feature": "age",
            "widget": "slider",
            "min": 18,
            "max": 90,
            "step": 1,
        },
        {
            "id": "gender",
            "question": "What best describes your gender?",
            "feature": "gender",
            "widget": "select",
            "options": ["Female", "Male", "Other"],
            "value_map": {"Female": 0.0, "Male": 1.0, "Other": 2.0},
        },
        {
            "id": "thirst",
            "question": "Have you been unusually thirsty or needing to urinate more often lately?",
            "feature": "glucose",
            "widget": "select",
            "options": ["No", "A little", "Yes, often"],
            "value_map": {"No": 90.0, "A little": 120.0, "Yes, often": 160.0},
        },
        {
            "id": "fatigue",
            "question": "Do you often feel tired, weak, or unusually sleepy during the day?",
            "feature": "hba1c",
            "widget": "select",
            "options": ["No", "Sometimes", "Often"],
            "value_map": {"No": 4.5, "Sometimes": 6.0, "Often": 8.0},
        },
        {
            "id": "weight_change",
            "question": "Have you noticed unexplained weight loss or weight gain?",
            "feature": "bmi",
            "widget": "select",
            "options": ["No", "A little", "Yes"],
            "value_map": {"No": 22.0, "A little": 26.0, "Yes": 31.0},
        },
        {
            "id": "family_history",
            "question": "Has a close family member been diagnosed with diabetes?",
            "feature": "family_history_diabetes",
            "widget": "select",
            "options": ["No", "Not sure", "Yes"],
            "value_map": {"No": 0.0, "Not sure": 0.5, "Yes": 1.0},
        },
        {
            "id": "exercise",
            "question": "How often do you exercise or stay physically active each week?",
            "feature": "physical_activity",
            "widget": "select",
            "options": ["Rarely", "1-3 times", "4 or more times"],
            "value_map": {"Rarely": 0.0, "1-3 times": 1.0, "4 or more times": 2.0},
        },
        {
            "id": "smoking",
            "question": "Do you smoke cigarettes or use tobacco?",
            "feature": "smoking",
            "widget": "select",
            "options": ["No", "Occasionally", "Yes"],
            "value_map": {"No": 0.0, "Occasionally": 0.5, "Yes": 1.0},
        },
    ],
    "heart_disease": [
        {
            "id": "chest_pain",
            "question": "Do you get chest discomfort or pressure when you exert yourself?",
            "feature": "chest_pain",
            "widget": "select",
            "options": ["No", "Sometimes", "Often"],
            "value_map": {"No": 0.0, "Sometimes": 1.0, "Often": 2.0},
        },
        {
            "id": "shortness_of_breath",
            "question": "Do you feel short of breath during routine activity?",
            "feature": "resting_bp",
            "widget": "select",
            "options": ["No", "Occasionally", "Yes"],
            "value_map": {"No": 110.0, "Occasionally": 125.0, "Yes": 140.0},
        },
        {
            "id": "family_history",
            "question": "Has a close family member had heart disease or a stroke?",
            "feature": "family_history_heart_disease",
            "widget": "select",
            "options": ["No", "Not sure", "Yes"],
            "value_map": {"No": 0.0, "Not sure": 0.5, "Yes": 1.0},
        },
        {
            "id": "hypertension",
            "question": "Have you been told you have high blood pressure?",
            "feature": "hypertension",
            "widget": "select",
            "options": ["No", "Not sure", "Yes"],
            "value_map": {"No": 0.0, "Not sure": 0.5, "Yes": 1.0},
        },
        {
            "id": "cholesterol",
            "question": "Have you had high cholesterol or a recent lipid test that was abnormal?",
            "feature": "cholesterol",
            "widget": "select",
            "options": ["No", "Not sure", "Yes"],
            "value_map": {"No": 180.0, "Not sure": 210.0, "Yes": 250.0},
        },
        {
            "id": "exercise",
            "question": "How often do you get regular exercise or brisk walking?",
            "feature": "exercise_activity",
            "widget": "select",
            "options": ["Rarely", "1-3 times", "4 or more times"],
            "value_map": {"Rarely": 0.0, "1-3 times": 1.0, "4 or more times": 2.0},
        },
        {
            "id": "smoking",
            "question": "Do you smoke cigarettes or vape regularly?",
            "feature": "smoking",
            "widget": "select",
            "options": ["No", "Sometimes", "Yes"],
            "value_map": {"No": 0.0, "Sometimes": 0.5, "Yes": 1.0},
        },
    ],
    "kidney_disease": [
        {
            "id": "swelling",
            "question": "Have you noticed swelling in your feet, ankles, or around your eyes?",
            "feature": "bp",
            "widget": "select",
            "options": ["No", "Sometimes", "Yes"],
            "value_map": {"No": 120.0, "Sometimes": 135.0, "Yes": 155.0},
        },
        {
            "id": "urine_change",
            "question": "Have you noticed foamy urine, blood in urine, or a change in urination?",
            "feature": "sg",
            "widget": "select",
            "options": ["No", "A little", "Yes"],
            "value_map": {"No": 1.02, "A little": 1.01, "Yes": 1.0},
        },
        {
            "id": "fatigue",
            "question": "Do you often feel weak, tired, or low on energy?",
            "feature": "hemo",
            "widget": "select",
            "options": ["No", "Sometimes", "Often"],
            "value_map": {"No": 14.0, "Sometimes": 11.5, "Often": 9.0},
        },
        {
            "id": "high_bp",
            "question": "Have you been told your blood pressure is high?",
            "feature": "htn",
            "widget": "select",
            "options": ["No", "Not sure", "Yes"],
            "value_map": {"No": 0.0, "Not sure": 0.5, "Yes": 1.0},
        },
        {
            "id": "diabetes",
            "question": "Have you been diagnosed with diabetes or high blood sugar?",
            "feature": "dm",
            "widget": "select",
            "options": ["No", "Not sure", "Yes"],
            "value_map": {"No": 0.0, "Not sure": 0.5, "Yes": 1.0},
        },
        {
            "id": "appetite",
            "question": "Has your appetite changed or do you feel nauseated often?",
            "feature": "appet",
            "widget": "select",
            "options": ["Normal", "Slightly reduced", "Reduced"],
            "value_map": {"Normal": 1.0, "Slightly reduced": 0.5, "Reduced": 0.0},
        },
        {
            "id": "swelling2",
            "question": "Have you had recent swelling, puffiness, or discomfort around the face or hands?",
            "feature": "sc",
            "widget": "select",
            "options": ["No", "Sometimes", "Yes"],
            "value_map": {"No": 0.5, "Sometimes": 1.0, "Yes": 1.5},
        },
    ],
}

CLINICAL_PLAN_TEMPLATES = {
    "breast_cancer": {
        "probable_diagnosis": "Possible breast tissue abnormality that requires specialist assessment and imaging.",
        "demo_prescription": "Supportive care, symptom tracking and specialist referral; confirm with a licensed clinician.",
        "lifestyle_advice": [
            "Keep a symptom diary and note any increase in lump size or skin changes.",
            "Avoid unnecessary delays and follow up with a breast specialist promptly.",
        ],
        "follow_up": "Within 7 days if symptoms are new or worsening.",
        "emergency_priority": "Urgent review if there is sudden pain, skin redness, discharge, or rapid growth.",
        "admission_recommendation": "Outpatient specialist review is usually appropriate; urgent admission only if severe symptoms develop.",
    },
    "diabetes": {
        "probable_diagnosis": "Possible metabolic risk that warrants confirmatory blood sugar and HbA1c testing.",
        "demo_prescription": "Lifestyle-focused care plan with glucose monitoring and clinician review.",
        "lifestyle_advice": [
            "Prioritize balanced meals and regular activity.",
            "Track blood sugar readings and hydration closely.",
        ],
        "follow_up": "Within 2 weeks for repeat screening if symptoms persist.",
        "emergency_priority": "Urgent care if there is severe confusion, vomiting, or very high blood sugar symptoms.",
        "admission_recommendation": "Outpatient review is usually sufficient unless emergency symptoms occur.",
    },
    "heart_disease": {
        "probable_diagnosis": "Possible cardiovascular risk requiring cardiac evaluation and investigation.",
        "demo_prescription": "Cardiac risk reduction plan, activity guidance and medical follow-up.",
        "lifestyle_advice": [
            "Reduce salt, saturated fats and smoking exposure.",
            "Increase gentle activity and monitor blood pressure regularly.",
        ],
        "follow_up": "Within 7 to 14 days for urgent symptoms or abnormal screening results.",
        "emergency_priority": "Emergency care if chest pain is severe, persistent, or associated with fainting.",
        "admission_recommendation": "Urgent admission may be needed for severe chest pain or unstable symptoms.",
    },
    "kidney_disease": {
        "probable_diagnosis": "Possible renal concern that needs urine and kidney function review.",
        "demo_prescription": "Supportive care plan with hydration guidance and renal follow-up.",
        "lifestyle_advice": [
            "Monitor fluid intake and blood pressure carefully.",
            "Avoid unnecessary NSAIDs and follow up with a nephrologist if symptoms worsen.",
        ],
        "follow_up": "Within 1 to 2 weeks if abnormal symptoms continue.",
        "emergency_priority": "Immediate medical review for severe swelling, reduced urine, or confusion.",
        "admission_recommendation": "Admission may be appropriate if kidney symptoms are rapidly worsening.",
    },
    "default": {
        "probable_diagnosis": "Possible clinical concern that requires physician review and confirmatory testing.",
        "demo_prescription": "Supportive treatment plan and follow-up with a qualified clinician.",
        "lifestyle_advice": ["Maintain a healthy routine and monitor symptoms closely."],
        "follow_up": "Within 2 to 3 weeks depending on severity.",
        "emergency_priority": "Seek immediate medical attention for severe or rapidly worsening symptoms.",
        "admission_recommendation": "Outpatient review is usually preferred unless symptoms become unstable.",
    },
}


def categorize_feature(name: str) -> str:
    """Assign a raw feature column name to one of the UI input groups."""
    lname = name.lower()
    for group in GROUP_ORDER:
        keywords = GROUP_KEYWORDS[group]
        if any(k in lname for k in keywords):
            return group
    return "Lab Values"


# ----------------------------------------------------------------------------
# CLINICAL DECISION-SUPPORT MAPPINGS (demo only)
#
# These map a disease/dataset name to a recommended specialist and a short
# list of standard diagnostic tests. They exist purely to make the Patient
# Report and Doctor Mode workflow feel complete for a disease-prediction
# demo — they are NOT a substitute for professional medical judgement, and
# no dosing/prescription content is ever auto-generated (that stays a free
# text field the reviewing doctor fills in themselves).
# ----------------------------------------------------------------------------

SPECIALIST_MAP = {
    "heart": "Cardiologist",
    "cardio": "Cardiologist",
    "diabet": "Endocrinologist",
    "glucose": "Endocrinologist",
    "thyroid": "Endocrinologist",
    "breast": "Oncologist",
    "cancer": "Oncologist",
    "tumor": "Oncologist",
    "kidney": "Nephrologist",
    "renal": "Nephrologist",
    "liver": "Hepatologist",
    "hepat": "Hepatologist",
    "lung": "Pulmonologist",
    "respiratory": "Pulmonologist",
    "asthma": "Pulmonologist",
    "stroke": "Neurologist",
    "brain": "Neurologist",
    "neuro": "Neurologist",
    "skin": "Dermatologist",
}
DEFAULT_SPECIALIST = "General Physician"

DIAGNOSTIC_TESTS_MAP = {
    "heart": ["ECG (Electrocardiogram)", "Echocardiogram", "Lipid Profile", "Troponin Test", "Stress Test"],
    "cardio": ["ECG (Electrocardiogram)", "Echocardiogram", "Lipid Profile", "Troponin Test", "Stress Test"],
    "diabet": ["Fasting Blood Sugar", "HbA1c", "Oral Glucose Tolerance Test", "Lipid Panel"],
    "glucose": ["Fasting Blood Sugar", "HbA1c", "Oral Glucose Tolerance Test"],
    "breast": ["Mammogram", "Breast Ultrasound", "Biopsy", "MRI Breast"],
    "cancer": ["Biopsy", "Imaging (CT/MRI)", "Tumor Marker Blood Test"],
    "kidney": ["Serum Creatinine", "eGFR", "Urinalysis", "Renal Ultrasound"],
    "liver": ["Liver Function Test (LFT)", "Ultrasound Abdomen", "Hepatitis Panel"],
    "lung": ["Chest X-Ray", "Spirometry", "CT Chest"],
    "stroke": ["CT Brain", "MRI Brain", "Carotid Doppler", "Lipid Profile"],
}
DEFAULT_DIAGNOSTIC_TESTS = ["Complete Blood Count (CBC)", "Basic Metabolic Panel", "Physical Examination"]


def _match_by_keyword(disease: str, mapping: dict, default):
    lname = disease.lower()
    for keyword, value in mapping.items():
        if keyword in lname:
            return value
    return default


def get_specialist(disease: str) -> str:
    """Recommended specialist for a given disease/dataset name."""
    return _match_by_keyword(disease, SPECIALIST_MAP, DEFAULT_SPECIALIST)


def get_diagnostic_tests(disease: str) -> list:
    """Standard demo diagnostic tests a doctor might order for this disease."""
    return _match_by_keyword(disease, DIAGNOSTIC_TESTS_MAP, DEFAULT_DIAGNOSTIC_TESTS)


def risk_level(risk: float) -> str:
    """Bucket a 0-1 risk probability into a plain-English risk level."""
    if risk >= 0.65:
        return "High"
    if risk >= 0.35:
        return "Moderate"
    return "Low"


def get_appointment_suggestion(level: str) -> str:
    """Plain-English appointment urgency suggestion based on risk level."""
    return {
        "High": "See a specialist within the next 2-3 days.",
        "Moderate": "Schedule a check-up within the next 2-3 weeks.",
        "Low": "Routine annual check-up is sufficient.",
    }.get(level, "Consult a healthcare professional for guidance.")


def pretty(name: str) -> str:
    """Turn a snake_case column name into a human readable label."""
    return name.replace("_", " ").strip().title()


def get_patient_features(features: list) -> list:
    """Pick the subset of features to surface in Patient (simple) mode.

    Returns an ordered list of (question_label, feature_name) tuples, one
    per canonical category that has a match in this dataset's feature list.
    Every remaining feature is left out of the patient form entirely and is
    instead auto-filled with a sensible default at prediction time.
    """
    used = set()
    matched = []
    for label, keywords in PATIENT_CATEGORIES:
        for feature in features:
            if feature in used:
                continue
            lname = feature.lower()
            if any(k in lname for k in keywords):
                matched.append((label, feature))
                used.add(feature)
                break
    return matched


def get_patient_questionnaire(disease: str, meta: dict | None = None) -> list:
    """Return the disease-specific questionnaire if available for the given model metadata."""
    disease_key = (disease or "").lower().replace(" ", "_")
    questions = DISEASE_QUESTIONNAIRES.get(disease_key, [])
    if not meta:
        return questions

    allowed_features = set(meta.get("features", []))
    filtered = []
    for question in questions:
        feature = question.get("feature")
        feature_names = feature if isinstance(feature, list) else [feature]
        if any(name in allowed_features for name in feature_names):
            filtered.append(question)
    return filtered or questions


def get_clinical_plan(disease: str, risk_level: str, confidence: float | None = None) -> dict:
    """Generate a polished doctor-facing plan from the disease, risk level and confidence."""
    disease_key = (disease or "").lower().replace(" ", "_")
    template = CLINICAL_PLAN_TEMPLATES.get(disease_key, CLINICAL_PLAN_TEMPLATES["default"])
    risk_label = (risk_level or "Moderate").lower()
    if risk_label == "high":
        urgency = "High"
    elif risk_label == "moderate":
        urgency = "Moderate"
    else:
        urgency = "Low"

    return {
        "specialist": get_specialist(disease),
        "diagnostic_tests": get_diagnostic_tests(disease),
        "probable_diagnosis": template["probable_diagnosis"],
        "demo_prescription": template["demo_prescription"],
        "lifestyle_advice": template["lifestyle_advice"],
        "follow_up": template["follow_up"],
        "emergency_priority": template["emergency_priority"],
        "admission_recommendation": template["admission_recommendation"],
        "urgency": urgency,
        "confidence": confidence,
    }
