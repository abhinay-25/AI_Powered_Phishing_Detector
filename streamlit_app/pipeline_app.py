import os
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Phishing Detector (Pipeline)", layout="centered")
st.title("Phishing Detector (Pipeline)")
st.caption("Loads a pre-trained pipeline. No retraining at runtime.")

@st.cache_resource(show_spinner=False)
def load_pipeline(models_dir: str = "models"):
    path = os.path.join(models_dir, "phishing_pipeline.joblib")
    if not os.path.exists(path):
        st.error(f"Pipeline not found at {path}. Run the end-to-end notebook to generate it.")
        return None
    return joblib.load(path)

pipe = load_pipeline()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Single URL")
    url = st.text_input("Enter URL", value="https://example.com/login")
    if st.button("Predict", type="primary"):
        if pipe is None:
            st.stop()
        proba = pipe.predict_proba(pd.DataFrame({"url": [url]}))[:, 1][0]
        label = int(proba >= 0.5)
        st.write(f"Prediction: {'Phishing 🚨' if label == 1 else 'Legitimate ✅'} (prob={proba:.3f})")

with col2:
    st.subheader("Batch CSV Upload")
    file = st.file_uploader("Upload CSV with a 'url' column", type=["csv"])
    if file and pipe is not None:
        df_in = pd.read_csv(file)
        if "url" not in df_in.columns:
            st.error("CSV must contain a 'url' column")
        else:
            probs = pipe.predict_proba(df_in[["url"]])[:, 1]
            out = df_in.copy()
            out["proba_phishing"] = probs
            out["label_pred"] = (probs >= 0.5).astype(int)
            st.dataframe(out.head(20))
            st.download_button("Download Predictions", out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
