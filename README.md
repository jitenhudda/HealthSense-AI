# HealthSense AI

<div align="center">

# HealthSense AI
### AI-Powered Disease Prediction & Clinical Decision Support System

A production-oriented Machine Learning application that predicts multiple diseases using structured medical datasets and provides intelligent clinical decision support through an interactive Streamlit dashboard.

Built with Python, Scikit-learn, XGBoost and Streamlit.

</div>

---

## Overview

HealthSense AI is an end-to-end Machine Learning healthcare application designed to predict multiple diseases from patient information using trained classification models.

The application provides an intuitive patient-friendly interface that collects symptoms through simple questionnaires instead of exposing technical dataset features. After prediction, AI generates a patient report containing disease risk, confidence score and specialist recommendations.

A dedicated doctor workspace reviews patient reports and automatically generates diagnostic recommendations, probable diagnosis, suggested laboratory tests, demo prescriptions, lifestyle advice and follow-up recommendations.

The project demonstrates the complete Machine Learning lifecycle including:

- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Model persistence
- Real-time prediction
- Clinical report generation
- Interactive dashboard

---

# Features

### Machine Learning

- Multi-disease prediction system
- Automatic dataset preprocessing
- Multiple classification algorithms
- Automatic best model selection
- Saved trained models for fast inference
- Feature scaling and encoding pipeline
- Parallel model training
- High-speed prediction

---

### Supported Diseases

- Diabetes
- Heart Disease
- Kidney Disease
- Liver Disease
- Hypertension
- Stroke
- Parkinson's Disease
- Thyroid Disease
- Breast Cancer

---

### Patient Module

- Simple symptom-based questionnaire
- Disease-specific questions
- Patient-friendly interface
- AI disease prediction
- Risk percentage
- Confidence score
- Specialist recommendation
- Patient report generation

---

### Doctor Module

- Review patient reports
- AI-assisted clinical decision support
- Recommended diagnostic tests
- Probable diagnosis
- Demo prescriptions
- Lifestyle recommendations
- Follow-up recommendations
- Doctor report generation

---

### Dashboard

- Premium healthcare dashboard
- Disease prediction interface
- Prediction history
- Interactive visualizations
- Model status
- Patient reports
- Doctor reports
- Fast navigation

---

## Machine Learning Models

The application trains multiple algorithms for every disease and automatically selects the best-performing model.

| Model | Purpose |
|--------|----------|
| Logistic Regression | Baseline Classification |
| Support Vector Machine (SVM) | Non-linear Classification |
| Random Forest | Ensemble Learning |
| XGBoost | Gradient Boosting |

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Frontend | Streamlit |
| Machine Learning | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Model Storage | Joblib |
| Database | SQLite |

---

# Project Structure

```text
HealthSense-AI/
│
├── app.py
├── requirements.txt
│
├── data/
│   └── *.csv
│
├── models/
│   ├── *.joblib
│   └── metadata
│
└── src/
    ├── train.py
    ├── preprocess.py
    ├── predict.py
    ├── config.py
    ├── database.py
    ├── reports.py
    └── utils.py
```

---

# Project Workflow

```text
                   Medical CSV Dataset
                           │
                           ▼
                 Data Cleaning & Validation
                           │
                           ▼
                  Feature Engineering
                           │
                           ▼
                 Data Preprocessing Pipeline
                           │
                           ▼
                Train Multiple ML Models
      ┌──────────────┬──────────────┬──────────────┬──────────────┐
      │ Logistic Reg │     SVM      │ RandomForest │   XGBoost    │
      └──────────────┴──────────────┴──────────────┴──────────────┘
                           │
                           ▼
                 Model Performance Evaluation
                           │
                           ▼
                 Best Model Automatically Saved
                           │
                           ▼
                  Streamlit Web Application
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       Patient Questionnaire      Doctor Review
              │                         │
              ▼                         ▼
         AI Prediction          Clinical Suggestions
              │                         │
              └────────────┬────────────┘
                           ▼
                 Report Generation
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/your-username/HealthSense-AI.git
```

Move into the project directory

```bash
cd HealthSense-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Train Models

```bash
python -m src.train
```

---

# Run Application

```bash
streamlit run app.py
```

---

# Model Training Pipeline

1. Load medical datasets
2. Clean missing values
3. Perform preprocessing
4. Split train/test data
5. Train multiple models
6. Evaluate model performance
7. Select best-performing model
8. Save trained model
9. Load model during prediction

---

# Prediction Workflow

```text
Patient Input
      │
      ▼
Questionnaire
      │
      ▼
Feature Conversion
      │
      ▼
Saved ML Model
      │
      ▼
Prediction
      │
      ▼
Confidence Score
      │
      ▼
Risk Assessment
      │
      ▼
Patient Report
```

---

# Key Highlights

- Production-ready project architecture
- End-to-end Machine Learning pipeline
- Automatic preprocessing
- Automatic model selection
- Multiple disease prediction
- AI-generated clinical reports
- Interactive dashboard
- Modular codebase
- Fast prediction using saved models
- Easily extendable for new diseases

---

# Future Enhancements

- Deep Learning models
- Explainable AI (SHAP/LIME)
- PDF report export
- Appointment scheduling
- Electronic Health Records integration
- Cloud deployment
- REST API
- Multi-language support

---

# Author

**Jiten Hudda**

Machine Learning | Data Science | Artificial Intelligence

---

## License

This project is developed for educational, research, and portfolio purposes.

---
```