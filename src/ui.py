"""Reusable UI building blocks: theme CSS, dynamic form widgets and charts."""

import plotly.graph_objects as go
import streamlit as st

from src.config import (
    GROUP_ICONS, GROUP_ORDER, categorize_feature, get_patient_features,
    get_patient_questionnaire, pretty,
)

# ----------------------------------------------------------------------------
# THEME
# ----------------------------------------------------------------------------

def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .stApp {
            background: radial-gradient(circle at 15% 0%, #10233f 0%, #060c18 55%, #030710 100%);
            color: #E6ECF5;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1a30 0%, #060e1c 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        section[data-testid="stSidebar"] * { color: #E6ECF5 !important; }

        h1, h2, h3, h4 { color: #F4F8FF !important; font-weight: 700 !important; }
        p, span, label, li { color: #C9D6E8; }

        /* Glass card */
        .glass-card {
            background: rgba(255, 255, 255, 0.045);
            border: 1px solid rgba(255, 255, 255, 0.09);
            border-radius: 20px;
            padding: 1.4rem 1.6rem;
            backdrop-filter: blur(14px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.35);
            margin-bottom: 1.1rem;
        }

        .metric-card {
            background: linear-gradient(145deg, rgba(56,132,255,0.14), rgba(56,132,255,0.02));
            border: 1px solid rgba(90,150,255,0.25);
            border-radius: 18px;
            padding: 1.3rem 1.4rem;
            text-align: left;
        }
        .metric-value { font-size: 2.1rem; font-weight: 800; color: #ffffff; line-height: 1.1; }
        .metric-label { font-size: 0.85rem; color: #9FB2CE; margin-top: 0.35rem; text-transform: uppercase; letter-spacing: 0.06em; }

        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #7FB1FF;
            margin: 0.4rem 0 0.9rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .app-title {
            font-size: 1.9rem;
            font-weight: 800;
            background: linear-gradient(90deg, #7FB1FF, #C8E0FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .app-subtitle { color: #8FA3C2; font-size: 0.92rem; margin-top: -0.4rem; }

        .risk-badge {
            display: inline-block;
            padding: 0.55rem 1.3rem;
            border-radius: 999px;
            font-weight: 800;
            font-size: 1.05rem;
            letter-spacing: 0.03em;
        }
        .risk-high { background: rgba(214,69,69,0.18); color: #FF7676; border: 1px solid rgba(255,110,110,0.4); }
        .risk-moderate { background: rgba(230,160,40,0.18); color: #FFC864; border: 1px solid rgba(255,193,84,0.4); }
        .risk-low { background: rgba(39,174,96,0.16); color: #4FE38A; border: 1px solid rgba(79,227,138,0.4); }

        div[data-testid="stButton"] button {
            background: linear-gradient(90deg, #2C6BFF, #5A9CFF);
            color: white; border: none; border-radius: 12px;
            padding: 0.6rem 1.4rem; font-weight: 700; font-size: 0.95rem;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            box-shadow: 0 4px 18px rgba(44,107,255,0.35);
        }
        div[data-testid="stButton"] button:hover { transform: translateY(-1px); box-shadow: 0 6px 24px rgba(44,107,255,0.5); }

        div[data-testid="stDownloadButton"] button {
            background: linear-gradient(90deg, #1D8348, #27AE60);
            color: white; border: none; border-radius: 12px; font-weight: 700;
        }

        div[data-baseweb="radio"] label, div[data-baseweb="select"] { color: #E6ECF5 !important; }

        [data-testid="stMetricValue"] { color: #ffffff; }

        hr { border-color: rgba(255,255,255,0.08); }

        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-thumb { background: rgba(127,177,255,0.35); border-radius: 8px; }

        /* Larger, more readable base text + form labels */
        .stMarkdown, .stCaption, div[data-testid="stMarkdownContainer"] p { font-size: 1rem; }
        label[data-testid="stWidgetLabel"] p { font-size: 1rem !important; font-weight: 600 !important; color: #DCE6F5 !important; }

        /* Mode toggle pill */
        .mode-pill {
            display: inline-flex; gap: 0.4rem; padding: 0.3rem;
            background: rgba(255,255,255,0.05); border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.09);
        }

        /* Expander styling for Doctor mode collapsible sections */
        div[data-testid="stExpander"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            margin-bottom: 0.7rem;
        }
        div[data-testid="stExpander"] summary { font-size: 1.05rem; font-weight: 700; color: #CFE0FF; }

        /* Responsive layout tweaks for narrow / mobile viewports */
        @media (max-width: 768px) {
            .metric-value { font-size: 1.6rem; }
            .app-title { font-size: 1.5rem; }
            .glass-card { padding: 1rem 1.1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def app_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="app-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.write("")


def metric_card(label: str, value, col) -> None:
    col.markdown(
        f"""<div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def glass_start():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)


def glass_end():
    st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# DYNAMIC FORM WIDGETS
# ----------------------------------------------------------------------------

def group_features(meta: dict) -> dict:
    """Bucket every feature into its UI group, preserving column order."""
    groups = {g: [] for g in GROUP_ORDER}
    for feature in meta["features"]:
        groups[categorize_feature(feature)].append(feature)
    return {g: cols for g, cols in groups.items() if cols}


def render_field(feature: str, meta: dict, key_prefix: str, label_override: str = None):
    """Render the appropriate widget for one feature and return its value."""
    key = f"{key_prefix}_{feature}"
    label = label_override or pretty(feature)

    if feature in meta.get("categorical_features", []):
        options = meta["feature_categories"].get(feature, [])
        if len(options) == 2:
            choice = st.radio(label, options, horizontal=True, key=key)
        else:
            choice = st.selectbox(label, options, key=key)
        return choice

    info = meta["feature_ranges"].get(feature, {"min": 0, "max": 1, "mean": 0, "is_binary": False})

    if info.get("is_binary"):
        choice = st.radio(label, ["No", "Yes"], horizontal=True, key=key)
        return 1 if choice == "Yes" else 0

    lo, hi, mean = info["min"], info["max"], info["mean"]
    is_int = float(lo).is_integer() and float(hi).is_integer()
    if is_int:
        return st.slider(label, int(lo), int(hi), int(round(mean)), key=key)
    return st.slider(label, float(lo), float(hi), float(round(mean, 1)), step=0.1, key=key)


def render_patient_form(meta: dict, disease: str) -> tuple:
    """Render a disease-specific questionnaire in everyday language and map
    the answers into the trained model features without exposing the ML schema.
    """
    from src.predict import build_default_inputs

    questions = get_patient_questionnaire(disease, meta)
    if not questions:
        questions = [{"question": "How are you feeling today?", "feature": meta["features"][0], "widget": "select", "options": ["No", "Yes"], "value_map": {"No": 0, "Yes": 1}}]

    key_prefix = f"{meta['disease']}_patient"
    inputs = {}
    symptom_answers = {}

    cols = st.columns(2)
    for i, question in enumerate(questions):
        feature = question.get("feature")
        if not feature:
            continue
        with cols[i % 2]:
            widget = question.get("widget", "select")
            if widget == "select":
                options = question.get("options", [])
                selected = st.selectbox(question["question"], options, key=f"{key_prefix}_{question['id']}")
                value = question.get("value_map", {}).get(selected, selected)
            elif widget == "radio":
                options = question.get("options", [])
                selected = st.radio(question["question"], options, horizontal=True, key=f"{key_prefix}_{question['id']}")
                value = question.get("value_map", {}).get(selected, selected)
            else:
                min_value = question.get("min", 0)
                max_value = question.get("max", 100)
                step = question.get("step", 1)
                default = question.get("default", int((min_value + max_value) / 2))
                value = st.slider(
                    question["question"],
                    min_value=min_value,
                    max_value=max_value,
                    value=default,
                    step=step,
                    key=f"{key_prefix}_{question['id']}",
                )

            inputs[feature] = value
            answer_text = str(value)
            if question.get("widget") in {"select", "radio"} and "value_map" in question:
                answer_text = str(selected)
            symptom_answers[question["question"]] = answer_text

            if feature == "age":
                symptom_answers["Age"] = answer_text
                symptom_answers["_age"] = value
            if feature == "gender":
                symptom_answers["Gender"] = answer_text
                symptom_answers["_gender"] = value

    full_inputs = build_default_inputs(meta)
    full_inputs.update(inputs)
    return full_inputs, symptom_answers


# ----------------------------------------------------------------------------
# CHARTS
# ----------------------------------------------------------------------------

def risk_gauge(risk_pct: float) -> go.Figure:
    color = "#FF6B6B" if risk_pct >= 50 else "#4FE38A"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={"suffix": "%", "font": {"size": 42, "color": "#ffffff"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8FA3C2"},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(79,227,138,0.18)"},
                {"range": [40, 70], "color": "rgba(255,193,84,0.18)"},
                {"range": [70, 100], "color": "rgba(255,107,107,0.18)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E6ECF5"},
        height=280,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


def model_comparison_chart(all_results: dict) -> go.Figure:
    names = list(all_results.keys())
    aucs = [all_results[n]["roc_auc"] for n in names]
    fig = go.Figure(go.Bar(
        x=aucs, y=names, orientation="h",
        marker=dict(color="#5A9CFF", line=dict(color="#7FB1FF", width=1)),
        text=[f"{v:.3f}" for v in aucs], textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E6ECF5"}, height=260,
        xaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.08)", title="ROC-AUC"),
        margin=dict(l=10, r=30, t=20, b=30),
    )
    return fig


def history_trend_chart(df) -> go.Figure:
    fig = go.Figure()
    for disease in df["disease"].unique():
        sub = df[df["disease"] == disease].sort_values("timestamp")
        fig.add_trace(go.Scatter(
            x=sub["timestamp"], y=sub["risk_percent"], mode="lines+markers",
            name=pretty(disease),
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E6ECF5"}, height=320,
        legend=dict(orientation="h", y=1.15),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Time"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Risk %"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def disease_distribution_chart(df) -> go.Figure:
    counts = df["disease"].value_counts()
    fig = go.Figure(go.Pie(
        labels=[pretty(d) for d in counts.index], values=counts.values, hole=0.55,
        marker=dict(colors=["#5A9CFF", "#7FB1FF", "#2C6BFF", "#9FC6FF", "#3B82F6"]),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E6ECF5"}, height=300,
        showlegend=True, legend=dict(orientation="h", y=-0.1),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig
