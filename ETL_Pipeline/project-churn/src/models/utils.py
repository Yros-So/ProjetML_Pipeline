import joblib
from pathlib import Path

def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)
