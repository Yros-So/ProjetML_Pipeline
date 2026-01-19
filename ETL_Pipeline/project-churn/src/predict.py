import pandas as pd
import joblib
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models/best_model.joblib"
FEATURES_PATH = PROJECT_DIR / "models/features.json"
TASK_PATH = PROJECT_DIR / "models/task.json"

pipeline = joblib.load(MODEL_PATH)
FEATURES = json.load(open(FEATURES_PATH))
TASK = json.load(open(TASK_PATH))["task"]

def predict(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Ajouter colonnes manquantes
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0

    # Garder uniquement features
    df = df[FEATURES]

    # Prédiction
    df["prediction"] = pipeline.predict(df)

    # Probabilités classification
    if TASK == "classification" and hasattr(pipeline.named_steps["model"], "predict_proba"):
        df["probability"] = pipeline.predict_proba(df)[:, 1]

    return df
