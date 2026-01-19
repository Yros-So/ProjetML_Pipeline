import streamlit as st
import pandas as pd
from pathlib import Path
import sys

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("Churn Predictor - Test UI")

# --- Import des modules internes ---
try:
    sys.path.append(str(Path(__file__).resolve().parents[0] / "src"))
    from models.utils import load_model
    from features import build_feature_matrix
except Exception as e:
    st.warning("Impossible d'importer les modules internes. Exécutez depuis la racine du projet.")
    st.write(e)

# --- Sidebar options ---
st.sidebar.header("Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV (clients)", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample data (data/raw/sample_customers.csv)", value=True)
model_choice = st.sidebar.selectbox(
    "Choose model file",
    ["models/xgb.joblib", "models/rf.joblib", "models/logreg.joblib"]
)

# --- Charger les données ---
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif use_sample:
    sample = Path("data/raw/sample_customers.csv")
    if sample.exists():
        df = pd.read_csv(sample)
    else:
        st.error("Sample file not found. Upload a CSV.")
        st.stop()
else:
    st.info("Upload a CSV or enable 'Use sample data'.")
    st.stop()

st.subheader("Preview data")
st.dataframe(df.head())

# --- Charger le modèle ---
model_path = Path(model_choice)
if not model_path.exists():
    st.warning(f"Model {model_choice} not found. Train models first with `make train` or run train script.")
    st.stop()

try:
    model = load_model(model_path.name)
except Exception as e:
    st.error("Erreur lors du chargement du modèle: " + str(e))
    st.stop()

# --- Build features ---
X = build_feature_matrix(df)
st.write("Shape features:", X.shape)

# --- Prédictions ---
if st.button("Predict churn"):
    try:
        proba = model.predict_proba(X)[:, 1]
        preds = (proba > 0.5).astype(int)
        out = df.copy()
        out["churn_pred"] = preds
        out["churn_prob"] = proba

        st.success("Prédictions générées")
        st.dataframe(out.head())

        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger prédictions CSV",
            data=csv,
            file_name="predictions_streamlit.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error("Erreur lors de la prédiction: " + str(e))