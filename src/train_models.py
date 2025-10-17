import argparse
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. Provide a valid path via --data."
        )
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the dataset.")
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop("label", axis=1)
    y = df["label"]
    return X, y


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=None),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    reports = {}
    best_name = None
    best_model = None
    best_f1 = -1.0

    for name, model in models.items():
        # Use scaled features for linear/SVM models; tree-based models can also use scaled safely
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        try:
            # For ROC-AUC we need probability estimates; fall back to 0 if not available
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            else:
                roc_auc = None
        except Exception:
            roc_auc = None
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True)

        reports[name] = {
            "accuracy": acc,
            "f1_weighted": f1,
            "precision_weighted": prec,
            "recall_weighted": rec,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist(),
            "classification_report": cr,
        }

        print(f"=== {name} ===")
        print("Accuracy:", acc)
        print("Precision (weighted):", prec)
        print("Recall (weighted):", rec)
        print("F1 (weighted):", f1)
        if roc_auc is not None:
            print("ROC-AUC:", roc_auc)
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("\n" + "-" * 60 + "\n")

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    return best_name, best_model, scaler, reports


def save_artifacts(best_name: str, model, scaler, reports: dict, out_dir: str = "models"):
    os.makedirs(out_dir, exist_ok=True)
    reports_dir = os.path.join(out_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "phishing_detector_model.pkl")
    scaler_path = os.path.join(out_dir, "scaler.pkl")
    metrics_path = os.path.join(out_dir, "metrics.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    summary = {
        "best_model": best_name,
        "saved_model_path": model_path,
        "saved_scaler_path": scaler_path,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "reports": reports,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved best model: {best_name} -> {model_path}")
    print(f"Saved scaler -> {scaler_path}")
    print(f"Saved metrics -> {metrics_path}")

    # Save a simple metrics table as CSV for easy comparison
    import pandas as pd
    rows = []
    for m, r in reports.items():
        rows.append({
            "model": m,
            "accuracy": r.get("accuracy"),
            "precision_weighted": r.get("precision_weighted"),
            "recall_weighted": r.get("recall_weighted"),
            "f1_weighted": r.get("f1_weighted"),
            "roc_auc": r.get("roc_auc"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(reports_dir, "metrics_table.csv"), index=False)
    print(f"Saved metrics table -> {os.path.join(reports_dir, 'metrics_table.csv')}")

    # Bar plot comparison
    plt.figure(figsize=(9, 5))
    sns.barplot(x="model", y="f1_weighted", data=df)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "f1_comparison.png"))
    plt.close()
    print(f"Saved F1 comparison plot -> {os.path.join(reports_dir, 'f1_comparison.png')}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train multiple models for phishing detection and save the best."
    )
    p.add_argument(
        "--data",
        type=str,
        default=os.path.join("data", "cleaned_phishing_dataset.csv"),
        help="Path to the cleaned dataset CSV (expects a 'label' column)",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="models",
        help="Directory to save model, scaler, and metrics",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f"Loading dataset from: {args.data}")
    df = load_dataset(args.data)
    X, y = split_features_target(df)

    best_name, best_model, scaler, reports = train_and_evaluate(
        X, y, random_state=args.random_state
    )

    save_artifacts(best_name, best_model, scaler, reports, out_dir=args.out_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
