# AI_Powered_Phishing_Detector

A scaffolded AI/ML project for detecting phishing using classical ML/XGBoost and a Streamlit app.

## Folder Structure
- `data/`           Raw and processed datasets (gitignored by default)
- `notebooks/`      Experiments and EDA
- `models/`         Saved models and artifacts (gitignored patterns included)
- `src/`            Python package code (feature engineering, training, inference)
- `streamlit_app/`  UI app using Streamlit

## Quickstart
1. Create venv (Windows PowerShell):
```
python -m venv venv
./venv/Scripts/Activate.ps1
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Initialize Git and push (after you create the repo on GitHub):
```
git init
git add .
git commit -m "chore: initial scaffolding"
git branch -M main
# Replace <your-username> with your GitHub username
git remote add origin https://github.com/<your-username>/AI_Powered_Phishing_Detector.git
git push -u origin main
```

## Colab
- See `notebooks/colab_setup.ipynb` for cloning the repo, checking GPU, and installing dependencies in Colab.

## Streamlit
Run the app (once you add code to `streamlit_app/`):
```
streamlit run streamlit_app/app.py
```

## Notes
- Large datasets and model binaries are ignored by default; consider using Git LFS for large assets.
- Adjust `.gitignore` rules to your needs.
