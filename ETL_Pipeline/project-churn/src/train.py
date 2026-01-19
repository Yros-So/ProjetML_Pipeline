import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_model.joblib"
FEATURES_PATH = MODEL_DIR / "features.json"
TASK_PATH = MODEL_DIR / "task.json"


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🧠 Entraînement d’un Modèle ML (Sélection manuelle des colonnes)")

uploaded = st.file_uploader("📂 Charger votre dataset (CSV)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Aperçu du dataset")
    st.dataframe(df.head())

    # =============================================================
    # 1. Sélection de la TARGET
    # =============================================================
    st.subheader("🎯 Sélection de la cible (Y)")
    target_col = st.selectbox("Colonne cible", df.columns)

    # =============================================================
    # 2. Sélection des FEATURES X
    # =============================================================
    st.subheader("📌 Sélection des features (X)")
    feature_cols = st.multiselect(
        "Colonnes explicatives",
        df.columns.drop(target_col),
        default=list(df.columns.drop(target_col))
    )

    # =============================================================
    # 3. Sélection des colonnes numériques et catégorielles
    # =============================================================
    st.subheader("🔧 Définition manuelle des types de colonnes")

    num_features = st.multiselect(
        "Colonnes numériques",
        feature_cols,
        default=[c for c in feature_cols if df[c].dtype != "object"]
    )

    cat_features = st.multiselect(
        "Colonnes catégorielles",
        feature_cols,
        default=[c for c in feature_cols if df[c].dtype == "object"]
    )

    # =============================================================
    # 4. Choix du modèle ML
    # =============================================================
    st.subheader("🤖 Choix du modèle")

    model_choice = st.selectbox(
        "Sélectionner un algorithme",
        [
            "RandomForest (Régression)",
            "RandomForest (Classification)",
            "Régression Linéaire",
            "Logistic Regression"
        ]
    )

    # =============================================================
    # TRAIN BUTTON
    # =============================================================
    if st.button("🚀 Entraîner le modèle"):

        if len(feature_cols) == 0:
            st.error("⚠️ Vous devez sélectionner au moins une feature X.")
            st.stop()

        if len(num_features) + len(cat_features) != len(feature_cols):
            st.error("⚠️ Toutes les features X doivent être soit numériques soit catégorielles.")
            st.stop()

        X = df[feature_cols]
        y = df[target_col]

        # Preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
            ]
        )

        # MODELES
        if model_choice == "RandomForest (Régression)":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            task = "regression"

        elif model_choice == "RandomForest (Classification)":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            task = "classification"

        elif model_choice == "Régression Linéaire":
            model = LinearRegression()
            task = "regression"

        elif model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=200)
            task = "classification"

        # PIPELINE
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # TRAIN SPLIT
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline.fit(X_train, y_train)

        # SAVE MODEL
        joblib.dump(pipeline, MODEL_PATH)

        # Save features
        with open(FEATURES_PATH, "w") as f:
            json.dump(feature_cols, f)

        # Save task type
        with open(TASK_PATH, "w") as f:
            json.dump({"task": task}, f)

        st.success("✅ Modèle entraîné et sauvegardé avec succès !")
        st.info(f"Tâche : {task}")
        st.info(f"Features utilisées : {feature_cols}")