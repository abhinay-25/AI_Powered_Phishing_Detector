import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Internal imports use repo logic for consistent feature ordering
try:
    from src.predict import load_artifacts as load_model_artifacts
    from src.predict import predict_url, predict_email
except Exception:
    load_model_artifacts = None
    predict_url = None
    predict_email = None

st.set_page_config(page_title="Phishing Detector • Comparative Analysis", layout="wide")
st.title("AI Powered Phishing Detector — Comparative Analysis & Demo")
st.caption("Loads saved metrics and artifacts, shows model comparisons, and lets you demo predictions using the best model. No retraining.")

MODELS_DIR = "models"
REPORTS_DIR = os.path.join(MODELS_DIR, "reports")

@st.cache_data(show_spinner=False)
def load_metrics(models_dir: str = MODELS_DIR):
    metrics_json_path = os.path.join(models_dir, "metrics.json")
    metrics_table_path = os.path.join(models_dir, "reports", "metrics_table.csv")
    metrics_json = None
    metrics_df = None
    if os.path.exists(metrics_json_path):
        try:
            with open(metrics_json_path, "r", encoding="utf-8") as f:
                metrics_json = json.load(f)
        except Exception as e:
            st.warning(f"Could not read metrics.json: {e}")
    if os.path.exists(metrics_table_path):
        try:
            metrics_df = pd.read_csv(metrics_table_path)
        except Exception as e:
            st.warning(f"Could not read metrics_table.csv: {e}")
    return metrics_json, metrics_df

@st.cache_resource(show_spinner=False)
def load_pipeline(models_dir: str = MODELS_DIR):
    path = os.path.join(models_dir, "phishing_pipeline.joblib")
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Could not load pipeline: {e}")
    return None

metrics_json, metrics_df = load_metrics()
pipeline = load_pipeline()

# Derive best model name
best_model_name = None
if metrics_json and isinstance(metrics_json, dict):
    best_model_name = metrics_json.get("best_model")

# Overview cards
with st.container():
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Artifacts Folder", MODELS_DIR)
    c2.metric("Reports Folder", REPORTS_DIR)
    c3.metric("Best Model", best_model_name or "Unknown")
    if metrics_json and best_model_name and "reports" in metrics_json and best_model_name in metrics_json["reports"]:
        rep = metrics_json["reports"][best_model_name]
        c4.metric("Best F1 (weighted)", f"{rep.get('f1_weighted', 0):.3f}")
    else:
        c4.metric("Best F1 (weighted)", "—")

st.markdown("---")

# Comparative analysis section
st.subheader("Model Comparison")
if metrics_df is not None and len(metrics_df) > 0:
    st.dataframe(metrics_df, use_container_width=True)

    # Bar charts: F1 and ROC-AUC
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    try:
        sns.barplot(ax=axes[0], data=metrics_df, x="model", y="f1_weighted")
        axes[0].set_title("F1 (weighted)")
        axes[0].tick_params(axis='x', rotation=20)
    except Exception:
        axes[0].text(0.5, 0.5, "No F1 data", ha='center')

    try:
        if "roc_auc" in metrics_df.columns and metrics_df["roc_auc"].notna().any():
            sns.barplot(ax=axes[1], data=metrics_df, x="model", y="roc_auc")
            axes[1].set_title("ROC-AUC")
            axes[1].tick_params(axis='x', rotation=20)
        else:
            axes[1].text(0.5, 0.5, "No ROC-AUC available", ha='center')
    except Exception:
        axes[1].text(0.5, 0.5, "No ROC-AUC data", ha='center')

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
else:
    st.warning("metrics_table.csv not found or empty. Train and save artifacts first.")

# Confusion matrix for the best model if available
if metrics_json and best_model_name and "reports" in metrics_json and best_model_name in metrics_json["reports"]:
    st.subheader(f"Confusion Matrix — {best_model_name}")
    rep = metrics_json["reports"][best_model_name]
    cm = rep.get("confusion_matrix")
    if cm is not None:
        cm = np.array(cm)
        fig_cm, ax_cm = plt.subplots(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('True')
        st.pyplot(fig_cm, clear_figure=True)
    else:
        st.info("No confusion matrix in metrics.")

st.markdown("---")

# Demo tabs
tab_url, tab_email, tab_artifacts = st.tabs(["URL Demo", "Email Demo", "Artifacts"])

# Try loading model+scaler artifacts via repo helper
model = scaler = feature_columns = None
if load_model_artifacts:
    try:
        model, scaler, feature_columns = load_model_artifacts(MODELS_DIR)
    except Exception as e:
        st.warning(f"Could not load model+scaler artifacts: {e}")

with tab_url:
    st.subheader("Predict on a URL")
    demo_url = st.text_input("Enter URL", value="http://secure-login-update.example.com/verify?acc=123")
    if st.button("Predict (URL)", type="primary"):
        if model is not None and predict_url is not None and feature_columns is not None:
            pred_label, prob, feats = predict_url(demo_url, model, scaler, feature_columns)
            label_txt = "Phishing 🚨" if int(pred_label) == 1 else "Legitimate ✅"
            st.success(f"Prediction: {label_txt}")
            if prob is not None:
                st.write(f"Confidence (prob phishing): {prob:.3f}")
            with st.expander("Feature Signals"):
                st.json(feats)
        elif pipeline is not None:
            # Fallback: pipeline-only inference (URL only)
            try:
                proba = pipeline.predict_proba(pd.DataFrame({'url':[demo_url]}))[:,1][0]
                label = int(proba >= 0.5)
                label_txt = "Phishing 🚨" if label == 1 else "Legitimate ✅"
                st.success(f"Prediction: {label_txt} (prob phishing={proba:.3f})")
            except Exception as e:
                st.error(f"Pipeline prediction failed: {e}")
        else:
            st.error("No artifacts available. Train and save artifacts first.")

with tab_email:
    st.subheader("Predict on an Email")
    subject = st.text_input("Subject", value="URGENT: Verify your account now")
    body = st.text_area("Body", height=180, value="Dear user, your account will be suspended. Click here to verify: http://bit.ly/verify-now")
    if st.button("Predict (Email)"):
        if model is not None and predict_email is not None and feature_columns is not None:
            pred_label, prob, feats = predict_email(subject, body, model, scaler, feature_columns)
            label_txt = "Phishing 🚨" if int(pred_label) == 1 else "Legitimate ✅"
            st.success(f"Prediction: {label_txt}")
            if prob is not None:
                st.write(f"Confidence (prob phishing): {prob:.3f}")
            with st.expander("Feature Signals"):
                st.json(feats)
        else:
            st.error("Email demo requires model+scaler artifacts from repo training path.")

with tab_artifacts:
    st.subheader("Artifacts")
    st.write("Best effort to show what's present under models/…")
    files = []
    for root, _, fs in os.walk(MODELS_DIR):
        for f in fs:
            files.append(os.path.relpath(os.path.join(root, f), start=MODELS_DIR))
    if files:
        st.code("\n".join(sorted(files)))
    else:
        st.info("No files under models/ yet.")
