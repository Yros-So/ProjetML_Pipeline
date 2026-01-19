from pathlib import Path

# Racine du projet : project-churn/
ROOT = Path(__file__).resolve().parents[1]

# Répertoires essentiels
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Fichiers modèles
BEST_MODEL = MODELS_DIR / "best_model.joblib"
SCALER = MODELS_DIR / "scaler.joblib"
FEATURES = MODELS_DIR / "features.json"
