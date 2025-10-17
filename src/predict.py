import argparse
import json
import os
import sys

import joblib
import pandas as pd

from .features import url_features, email_features


def load_artifacts(models_dir: str = "models"):
    model_path = os.path.join(models_dir, "phishing_detector_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model/scaler not found. Train first.")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_url(url: str, model, scaler):
    feats = url_features(url)
    X = pd.DataFrame([feats])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    prob = (
        model.predict_proba(X_scaled)[0, 1] if hasattr(model, "predict_proba") else None
    )
    return int(pred), prob, feats


def predict_email(subject: str, body: str, model, scaler):
    feats = email_features(subject, body)
    X = pd.DataFrame([feats])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    prob = (
        model.predict_proba(X_scaled)[0, 1] if hasattr(model, "predict_proba") else None
    )
    return int(pred), prob, feats


def main():
    ap = argparse.ArgumentParser(description="Predict phishing for URL or email.")
    ap.add_argument("--mode", choices=["url", "email"], required=True)
    ap.add_argument("--url", type=str, help="URL to classify")
    ap.add_argument("--subject", type=str, help="Email subject")
    ap.add_argument("--body", type=str, help="Email body")
    args = ap.parse_args()

    model, scaler = load_artifacts()

    if args.mode == "url":
        if not args.url:
            raise SystemExit("--url is required when --mode url")
        pred, prob, feats = predict_url(args.url, model, scaler)
    else:
        pred, prob, feats = predict_email(args.subject or "", args.body or "", model, scaler)

    label = "Phishing" if pred == 1 else "Legitimate"
    print(json.dumps({"label": label, "prob": prob, "features": feats}, indent=2))


if __name__ == "__main__":
    main()
