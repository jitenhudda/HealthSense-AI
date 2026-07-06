"""
HealthSense AI — Multi-disease risk prediction platform.

Streamlit entry point. Automatically discovers trained models produced by
src/train.py (one per CSV in data/), renders a plain-English Patient
Screening form, and lets a doctor review the resulting AI prediction and
extend it into a Doctor Report. Disease prediction (Logistic Regression,
SVM, Random Forest, XGBoost) remains the core feature throughout — Doctor
Mode only reviews and annotates AI predictions, it never becomes a
hospital ERP (no pharmacy, billing, ambulance or ward management).
"""

import os
import time

import pandas as pd
import streamlit as st

from src.config import DATA_DIR, MODEL_DIR, get_clinical_plan, get_diagnostic_tests, pretty
from src.history import (
    load_doctor_reports, load_patient_reports, save_doctor_report,
    save_patient_report,
)
from src.pdf_report import (
    generate_doctor_report_pdf, generate_patient_report_pdf, get_recommendations,
)
from src.predict import list_diseases, load_meta, load_model, predict
from src.train import train_all
from src.ui import (
    app_header, disease_distribution_chart, glass_end, glass_start,
    history_trend_chart, inject_theme, metric_card, model_comparison_chart,
    render_patient_form, risk_gauge,
)

st.set_page_config(
    page_title="HealthSense AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()


# ----------------------------------------------------------------------------
# AUTO-SYNC: detect new/removed CSVs and (re)train affected models once per
# session. train_all() itself skips datasets that haven't changed and
# removes models for datasets that were deleted from data/, so this stays
# fast even with several diseases registered.
# ----------------------------------------------------------------------------

def _data_fingerprint() -> tuple:
    if not os.path.isdir(DATA_DIR):
        return ()
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    return tuple((f, os.path.getmtime(os.path.join(DATA_DIR, f))) for f in files)


fingerprint = _data_fingerprint()
if st.session_state.get("_data_fingerprint") != fingerprint:
    with st.spinner("Syncing datasets and models..."):
        sync_summary = train_all()
    st.session_state["_data_fingerprint"] = fingerprint
    st.session_state["_last_sync_summary"] = sync_summary
    load_meta.clear()
    load_model.clear()


# ----------------------------------------------------------------------------
# CACHED DATA HELPERS
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def count_datasets() -> int:
    if not os.path.isdir(DATA_DIR):
        return 0
    return len([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])


def get_diseases() -> list:
    return list_diseases()


RISK_BADGE_CLASS = {"High": "risk-high", "Moderate": "risk-moderate", "Low": "risk-low"}


# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.2rem;">'
        '<span style="font-size:1.8rem;">🩺</span>'
        '<span style="font-size:1.35rem;font-weight:800;color:#F4F8FF;">HealthSense AI</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("AI-powered multi-disease risk screening")
    st.write("")

    page = st.radio(
        "Navigate",
        ["📊  Dashboard", "🙂  Patient Screening", "🧑‍⚕️  Doctor Review",
         "📁  Reports", "⚙️  Model Manager"],
        label_visibility="collapsed",
    )

    st.write("")
    st.markdown("---")
    diseases = get_diseases()
    st.caption(f"**{len(diseases)}** trained model(s) available")
    for d in diseases:
        st.markdown(f"&nbsp;&nbsp;🔹 {pretty(d)}")


# ----------------------------------------------------------------------------
# DASHBOARD
# ----------------------------------------------------------------------------

if page.startswith("📊"):
    app_header("Dashboard", "Overview of your AI disease-prediction system")

    patient_df = load_patient_reports()
    doctor_df = load_doctor_reports()
    diseases = get_diseases()

    c1, c2, c3, c4 = st.columns(4)
    metric_card("Datasets Loaded", count_datasets(), c1)
    metric_card("Diseases Supported", len(diseases), c2)
    metric_card("Patient Screenings", len(patient_df), c3)
    metric_card("Doctor Reviews Completed", len(doctor_df), c4)

    st.write("")
    left, right = st.columns([1.3, 1])

    with left:
        glass_start()
        st.markdown('<div class="section-title">📈 Risk Prediction Trend</div>', unsafe_allow_html=True)
        if patient_df.empty:
            st.info("No patient screenings yet. Run one from Patient Screening to see trends here.")
        else:
            st.plotly_chart(history_trend_chart(patient_df), use_container_width=True)
        glass_end()

    with right:
        glass_start()
        st.markdown('<div class="section-title">🥧 Screenings by Disease</div>', unsafe_allow_html=True)
        if patient_df.empty:
            st.info("No screening history available yet.")
        else:
            st.plotly_chart(disease_distribution_chart(patient_df), use_container_width=True)
        glass_end()

    st.write("")
    glass_start()
    st.markdown('<div class="section-title">🧠 Model Performance by Disease</div>', unsafe_allow_html=True)
    if not diseases:
        st.warning("No trained models found. Go to **Model Manager** to train models from your CSV files.")
    else:
        cols = st.columns(min(3, len(diseases)))
        for i, disease in enumerate(diseases):
            meta = load_meta(disease)
            with cols[i % len(cols)]:
                st.markdown(f"**{pretty(disease)}**")
                st.caption(f"Best model: {meta['best_model']}  ·  {meta['n_samples']} samples")
                st.plotly_chart(model_comparison_chart(meta["all_results"]), use_container_width=True)
    glass_end()

    if not patient_df.empty:
        st.write("")
        glass_start()
        pending = int((patient_df["status"] == "Pending Doctor Review").sum())
        st.markdown('<div class="section-title">🩺 Doctor Review Queue</div>', unsafe_allow_html=True)
        if pending:
            st.info(f"**{pending}** patient report(s) are waiting for doctor review. Head to **Doctor Review**.")
        else:
            st.success("All patient reports have been reviewed by a doctor.")
        glass_end()


# ----------------------------------------------------------------------------
# PATIENT SCREENING — the only way a patient interacts with the model.
# Asks 7-10 plain-English questions, never shows raw dataset columns/ML
# features, and produces a Patient Report.
# ----------------------------------------------------------------------------

elif page.startswith("🙂"):
    app_header("Patient Screening", "Answer a few simple questions to get your AI risk assessment")

    diseases = get_diseases()
    if not diseases:
        st.warning("No trained models available yet. Add a CSV to the data/ folder and reload the app.")
    else:
        disease = st.selectbox("What would you like to be screened for?", diseases, format_func=pretty)
        meta = load_meta(disease)

        glass_start()
        st.markdown(
            f'<div class="section-title">📝 A Few Quick Questions — {pretty(disease)}</div>',
            unsafe_allow_html=True,
        )
        st.caption("These are everyday symptom questions so the screening feels like a clinical conversation, not a form.")
        patient_name = st.text_input("Your Name (optional)", key=f"{disease}_patient_name")
        full_inputs, symptom_answers = render_patient_form(meta, disease)
        run = st.button("🔍  Get My AI Risk Assessment", use_container_width=False)
        glass_end()

        if run:
            with st.spinner("Analyzing your answers..."):
                result = predict(disease, full_inputs)
                age = symptom_answers.get("Age", "")
                gender = symptom_answers.get("Gender", "")
                symptom_answers["_age"] = age
                symptom_answers["_gender"] = gender
                patient_report = save_patient_report(disease, patient_name, symptom_answers, result)
                symptom_answers.pop("_age", None)
                symptom_answers.pop("_gender", None)

            level = patient_report["risk_level"]
            st.write("")
            res_left, res_right = st.columns([1, 1.2])

            with res_left:
                glass_start()
                badge_class = RISK_BADGE_CLASS.get(level, "risk-moderate")
                st.markdown(
                    f'<span class="risk-badge {badge_class}">{level.upper()} RISK</span>',
                    unsafe_allow_html=True,
                )
                st.write("")
                st.plotly_chart(risk_gauge(patient_report["risk_percent"]), use_container_width=True)
                m1, m2 = st.columns(2)
                m1.metric("Confidence", f"{patient_report['confidence_percent']}%")
                m2.metric("Risk Level", level)
                glass_end()

            with res_right:
                glass_start()
                st.markdown('<div class="section-title">🩺 Patient Report Summary</div>', unsafe_allow_html=True)
                st.markdown(f"**Recommended Specialist:** {patient_report['recommended_specialist']}")
                st.markdown(f"**Appointment Suggestion:** {patient_report['appointment_suggestion']}")
                st.write("")
                st.markdown("**What this means:**")
                st.caption(
                    "Your answers were reviewed by the AI model to estimate disease risk. "
                    "This is a screening tool and should be reviewed by a qualified clinician."
                )
                st.markdown("**General Guidance:**")
                for rec in get_recommendations(patient_report["prediction"] == "High Risk"):
                    st.markdown(f"- {rec}")
                st.write("")
                pdf_bytes = generate_patient_report_pdf(disease, patient_report, symptom_answers)
                st.download_button(
                    "📄  Download Patient Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"patient_report_{patient_report['patient_report_id']}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.caption(
                    f"Report ID **{patient_report['patient_report_id']}** — share this with your "
                    "doctor, or ask them to open it under Doctor Review."
                )
                glass_end()


# ----------------------------------------------------------------------------
# DOCTOR REVIEW — opens a patient's report, reviews the AI prediction,
# suggests diagnostic tests (demo), records observations (demo), and
# produces a Doctor Report that extends the AI prediction. It never
# collects raw ML features directly and never becomes a hospital ERP.
# ----------------------------------------------------------------------------

elif page.startswith("🧑‍⚕️"):
    app_header("Doctor Review", "Open a patient's AI report, review it, and finalize your assessment")

    patient_df = load_patient_reports()
    if patient_df.empty:
        st.info("No patient reports yet. Ask the patient to complete a screening under **Patient Screening**.")
    else:
        show_all = st.checkbox("Show already-reviewed reports too", value=False)
        visible_df = patient_df if show_all else patient_df[patient_df["status"] == "Pending Doctor Review"]

        if visible_df.empty:
            st.success("No pending reports right now — all patient reports have been reviewed.")
        else:
            def _label(row_id):
                row = visible_df[visible_df["patient_report_id"] == row_id].iloc[0]
                return (f"{row['patient_name']} — {pretty(row['disease'])} — "
                        f"{row['risk_level']} Risk — {row['status']}")

            selected_id = st.selectbox(
                "Select a patient report to open",
                visible_df["patient_report_id"].tolist(),
                format_func=_label,
            )
            report = patient_df[patient_df["patient_report_id"] == selected_id].iloc[0].to_dict()
            disease = report["disease"]

            glass_start()
            st.markdown(f'<div class="section-title">🧾 Patient Report — {report["patient_name"]}</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("AI Risk Prediction", f"{report['risk_percent']}%")
            c2.metric("Risk Level", report["risk_level"])
            c3.metric("Confidence", f"{report['confidence_percent']}%")
            c4.metric("Model Used", report["model_used"])
            st.caption(
                f"Disease: **{pretty(disease)}**  ·  Recommended Specialist: **{report['recommended_specialist']}**  ·  "
                f"Appointment Suggestion: {report['appointment_suggestion']}"
            )
            st.caption("The doctor view loads the pending AI screening automatically so the review can start immediately.")
            symptom_cols = [c for c in report if str(c).startswith("symptom_") and pd.notna(report[c])]
            if symptom_cols:
                with st.expander("View patient-reported symptoms & details"):
                    for c in symptom_cols:
                        st.markdown(f"- **{c[len('symptom_'):]}:** {report[c]}")
            glass_end()

            st.write("")
            glass_start()
            st.markdown('<div class="section-title">🧪 Diagnostic Tests (Demo)</div>', unsafe_allow_html=True)
            suggested = get_diagnostic_tests(disease)
            tests_ordered = st.multiselect(
                "Tests to order for this patient", options=suggested, default=suggested,
                key=f"tests_{selected_id}",
            )
            test_observations = st.text_area(
                "Record test observations (demo)",
                placeholder="e.g. ECG shows normal sinus rhythm; lipid profile mildly elevated...",
                key=f"obs_{selected_id}",
            )
            glass_end()

            st.write("")
            glass_start()
            st.markdown('<div class="section-title">📝 Doctor Report</div>', unsafe_allow_html=True)
            clinical_plan = get_clinical_plan(disease, report["risk_level"], report["confidence_percent"])
            final_diagnosis = st.text_area(
                "Final Diagnosis",
                value=clinical_plan["probable_diagnosis"],
                placeholder=f"e.g. Confirmed {pretty(disease)} risk / Ruled out / Needs further testing...",
                key=f"diag_{selected_id}",
            )
            demo_prescription = st.text_area(
                "Demo Prescription (placeholder only — not a real prescription)",
                value=clinical_plan["demo_prescription"],
                placeholder="e.g. Medication name, dosage, frequency — to be completed by the reviewing physician",
                key=f"presc_{selected_id}",
            )
            default_lifestyle = "\n".join(f"- {r}" for r in clinical_plan["lifestyle_advice"])
            lifestyle_advice = st.text_area(
                "Lifestyle Advice", value=default_lifestyle, key=f"life_{selected_id}",
            )
            follow_up_date = st.date_input("Follow-up Date", key=f"fu_{selected_id}")
            doctor_notes = st.text_area(
                "Doctor Notes",
                value=(
                    f"Suggested priority: {clinical_plan['urgency']}\n"
                    f"Emergency priority: {clinical_plan['emergency_priority']}\n"
                    f"Admission recommendation: {clinical_plan['admission_recommendation']}"
                ),
                placeholder="Any additional notes for the patient record...",
                key=f"notes_{selected_id}",
            )

            submit = st.button("✅  Generate Doctor Report", use_container_width=False)
            glass_end()

            if submit:
                review = {
                    "tests_ordered": tests_ordered,
                    "test_observations": test_observations,
                    "final_diagnosis": final_diagnosis,
                    "demo_prescription": demo_prescription,
                    "lifestyle_advice": lifestyle_advice,
                    "follow_up_date": str(follow_up_date),
                    "doctor_notes": doctor_notes,
                }
                doctor_report = save_doctor_report(report, review)
                st.success(f"Doctor Report **{doctor_report['doctor_report_id']}** saved and linked to this patient report.")
                pdf_bytes = generate_doctor_report_pdf(report, doctor_report)
                st.download_button(
                    "📄  Download Doctor Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"doctor_report_{doctor_report['doctor_report_id']}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )


# ----------------------------------------------------------------------------
# REPORTS — Patient Reports and Doctor Reports, stored and shown separately.
# ----------------------------------------------------------------------------

elif page.startswith("📁"):
    app_header("Reports", "Patient Reports and Doctor Reports are stored separately")

    tab1, tab2 = st.tabs(["🙂 Patient Reports", "🧑‍⚕️ Doctor Reports"])

    with tab1:
        patient_df = load_patient_reports()
        if patient_df.empty:
            st.info("No patient reports recorded yet.")
        else:
            diseases_opt = ["All"] + sorted(patient_df["disease"].unique().tolist())
            col_a, col_b = st.columns([1, 3])
            with col_a:
                filter_disease = st.selectbox(
                    "Filter by disease", diseases_opt,
                    format_func=lambda d: pretty(d) if d != "All" else d, key="pf_disease",
                )
            filtered = patient_df if filter_disease == "All" else patient_df[patient_df["disease"] == filter_disease]

            c1, c2, c3 = st.columns(3)
            metric_card("Total Reports", len(filtered), c1)
            metric_card("Avg Risk %", f"{filtered['risk_percent'].mean():.1f}%" if len(filtered) else "—", c2)
            metric_card("Pending Review", int((filtered["status"] == "Pending Doctor Review").sum()), c3)

            st.write("")
            glass_start()
            st.markdown('<div class="section-title">📋 Patient Report Records</div>', unsafe_allow_html=True)
            display_cols = ["patient_report_id", "timestamp", "disease", "patient_name",
                             "risk_percent", "confidence_percent", "risk_level",
                             "recommended_specialist", "status"]
            st.dataframe(
                filtered[display_cols].assign(disease=lambda d: d["disease"].map(pretty)),
                use_container_width=True, hide_index=True,
            )
            glass_end()

    with tab2:
        doctor_df = load_doctor_reports()
        if doctor_df.empty:
            st.info("No doctor reports recorded yet. Complete a review under Doctor Review.")
        else:
            diseases_opt = ["All"] + sorted(doctor_df["disease"].unique().tolist())
            col_a, col_b = st.columns([1, 3])
            with col_a:
                filter_disease = st.selectbox(
                    "Filter by disease", diseases_opt,
                    format_func=lambda d: pretty(d) if d != "All" else d, key="df_disease",
                )
            filtered = doctor_df if filter_disease == "All" else doctor_df[doctor_df["disease"] == filter_disease]

            glass_start()
            st.markdown('<div class="section-title">📋 Doctor Report Records</div>', unsafe_allow_html=True)
            display_cols = ["doctor_report_id", "patient_report_id", "timestamp", "disease",
                             "patient_name", "ai_risk_percent", "ai_risk_level", "final_diagnosis",
                             "follow_up_date"]
            st.dataframe(
                filtered[display_cols].assign(disease=lambda d: d["disease"].map(pretty)),
                use_container_width=True, hide_index=True,
            )
            glass_end()


# ----------------------------------------------------------------------------
# MODEL MANAGER
# ----------------------------------------------------------------------------

elif page.startswith("⚙️"):
    app_header("Model Manager", "Datasets are auto-detected, trained and kept in sync")

    trained = get_diseases()

    glass_start()
    st.markdown('<div class="section-title">🔄 Auto-Sync Status</div>', unsafe_allow_html=True)
    st.caption(
        "Every CSV placed in data/ is automatically preprocessed, compared across "
        "Logistic Regression, SVM, Random Forest and XGBoost, and the best model is saved. "
        "Removing a CSV automatically removes its model — nothing untrained or stale "
        "is ever shown in the app."
    )
    last_summary = st.session_state.get("_last_sync_summary", [])
    newly_trained = [r for r in last_summary if not r.get("skipped") and "error" not in r]
    if newly_trained:
        for r in newly_trained:
            st.markdown(
                f"✅ **{pretty(r['disease'])}** trained → best model: **{r['best_model']}** "
                f"(ROC-AUC: {r['metrics']['roc_auc']:.3f})"
            )
    else:
        st.info("All models are already up to date with their source datasets.")

    if st.button("🔁  Force Retrain All Models", use_container_width=False):
        progress = st.progress(0.0, text="Starting training...")

        def _cb(i, total, name):
            progress.progress(i / total, text=f"Training model {i}/{total}: {pretty(name)}")

        with st.spinner("Training in progress..."):
            summary = train_all(progress_callback=_cb, force=True)
        progress.progress(1.0, text="Training complete!")

        st.session_state["_last_sync_summary"] = summary
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    glass_end()

    if trained:
        st.write("")
        glass_start()
        st.markdown('<div class="section-title">📊 Trained Model Details</div>', unsafe_allow_html=True)
        for disease in trained:
            meta = load_meta(disease)
            with st.expander(f"{pretty(disease)} — {meta['best_model']}"):
                cols = st.columns(4)
                cols[0].metric("Accuracy", f"{meta['metrics']['accuracy']*100:.1f}%")
                cols[1].metric("ROC-AUC", f"{meta['metrics']['roc_auc']:.3f}")
                cols[2].metric("F1 Score", f"{meta['metrics']['f1_score']:.3f}")
                cols[3].metric("Samples", meta["n_samples"])
                st.caption(f"Trained at {meta['trained_at']} · {meta['training_seconds']}s")
                st.plotly_chart(model_comparison_chart(meta["all_results"]), use_container_width=True)
        glass_end()
