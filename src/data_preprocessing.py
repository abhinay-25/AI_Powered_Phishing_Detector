import argparse
import os
import re
from typing import List

import pandas as pd
from .features import url_features, email_features


def load_url_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Heuristic: try to find URL-like column and label column
    url_col = None
    label_col = None
    for c in df.columns:
        if re.search(r"url|domain|link", c, re.IGNORECASE):
            url_col = c
        if re.search(r"label|target|class", c, re.IGNORECASE):
            label_col = c
    if url_col is None:
        raise ValueError("Could not detect URL column in URL dataset. Please rename accordingly.")
    if label_col is None:
        raise ValueError("Could not detect label column in URL dataset. Please rename accordingly.")

    feat_rows = []
    for _, row in df.iterrows():
        feats = url_features(str(row[url_col]))
        feats["label"] = int(row[label_col])
        feat_rows.append(feats)
    return pd.DataFrame(feat_rows)


def load_email_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Heuristic: subject/body and label columns
    subject_col = None
    body_col = None
    label_col = None
    for c in df.columns:
        if re.search(r"subject", c, re.IGNORECASE):
            subject_col = c
        if re.search(r"body|text|content", c, re.IGNORECASE):
            body_col = c
        if re.search(r"label|target|class|is_phishing", c, re.IGNORECASE):
            label_col = c
    if body_col is None:
        raise ValueError("Could not detect email body/content column. Please rename accordingly.")
    if label_col is None:
        raise ValueError("Could not detect label column in email dataset. Please rename accordingly.")

    feat_rows = []
    for _, row in df.iterrows():
        feats = email_features(str(row.get(subject_col, "")), str(row.get(body_col, "")))
        feats["label"] = int(row[label_col])
        feat_rows.append(feats)
    return pd.DataFrame(feat_rows)


def main():
    ap = argparse.ArgumentParser(description="Preprocess URL and Email datasets and output unified CSV.")
    ap.add_argument("--url-csv", type=str, help="Path to URL phishing dataset CSV", required=False)
    ap.add_argument("--email-csv", type=str, help="Path to email phishing dataset CSV", required=False)
    ap.add_argument("--out", type=str, default=os.path.join("data", "cleaned_phishing_dataset.csv"))
    args = ap.parse_args()

    frames: List[pd.DataFrame] = []
    if args.url_csv and os.path.exists(args.url_csv):
        print(f"Processing URL dataset: {args.url_csv}")
        frames.append(load_url_dataset(args.url_csv))
    if args.email_csv and os.path.exists(args.email_csv):
        print(f"Processing Email dataset: {args.email_csv}")
        frames.append(load_email_dataset(args.email_csv))

    if not frames:
        raise SystemExit("No datasets provided. Use --url-csv and/or --email-csv.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    combined = pd.concat(frames, axis=0, ignore_index=True).fillna(0)
    combined.to_csv(args.out, index=False)
    print(f"Saved cleaned dataset -> {args.out} with shape {combined.shape}")


if __name__ == "__main__":
    main()
