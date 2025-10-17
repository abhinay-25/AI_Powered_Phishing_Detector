import json
import os
import pandas as pd
import streamlit as st
import joblib

from src.features import url_features, email_features

st.set_page_config(page_title="AI Powered Phishing Detector", layout="centered")

st.title("AI Powered Phishing Detector")
st.caption("Classify URLs or emails as Phishing (1) or Legitimate (0)")

@st.cache_resource(show_spinner=False)
def load_artifacts(models_dir: str = "models"):
	model_path = os.path.join(models_dir, "phishing_detector_model.pkl")
	scaler_path = os.path.join(models_dir, "scaler.pkl")
	if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
		st.warning("Trained model not found. Please run training first to generate models.")
		return None, None
	model = joblib.load(model_path)
	scaler = joblib.load(scaler_path)
	return model, scaler

model, scaler = load_artifacts()

tab_url, tab_email = st.tabs(["URL", "Email"])

with tab_url:
	url = st.text_input("Enter URL", placeholder="https://example.com/login")
	if st.button("Predict (URL)"):
		if model is None:
			st.stop()
		feats = url_features(url)
		X = pd.DataFrame([feats])
		X_scaled = scaler.transform(X)
		y = model.predict(X_scaled)[0]
		prob = model.predict_proba(X_scaled)[0, 1] if hasattr(model, "predict_proba") else None
		label = "Phishing 🚨" if int(y) == 1 else "Legitimate ✅"
		st.subheader(label)
		if prob is not None:
			st.write(f"Confidence (prob phishing): {prob:.3f}")
		st.markdown("### Feature Explanation")
		st.json(feats)

with tab_email:
	subject = st.text_input("Email Subject", placeholder="URGENT: Verify your account")
	body = st.text_area("Email Body", height=200, placeholder="Dear user, your account will be suspended...")
	if st.button("Predict (Email)"):
		if model is None:
			st.stop()
		feats = email_features(subject, body)
		X = pd.DataFrame([feats])
		X_scaled = scaler.transform(X)
		y = model.predict(X_scaled)[0]
		prob = model.predict_proba(X_scaled)[0, 1] if hasattr(model, "predict_proba") else None
		label = "Phishing 🚨" if int(y) == 1 else "Legitimate ✅"
		st.subheader(label)
		if prob is not None:
			st.write(f"Confidence (prob phishing): {prob:.3f}")
		st.markdown("### Feature Explanation")
		st.json(feats)
