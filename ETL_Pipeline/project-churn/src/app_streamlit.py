import shap
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, r2_score, mean_squared_error, accuracy_score
)
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve


# Colonnes interdites (IDs, index, timestamp)
BLACKLIST_COLS = ["id", "ID", "index", "timestamp", "date"]

# -------------------------
# Fonction : détection automatique de la cible
# -------------------------

def auto_detect_target(df):
    priority_names = ["target", "label", "y", "class", "churn", "outcome"]

    #  1️⃣ Vérifie les noms prioritaires
    for col in df.columns:
        if col.lower() in priority_names and col not in BLACKLIST_COLS:
            return col

    # 2️⃣ Colonne avec peu de valeurs uniques
    candidates = [
        col for col in df.columns
        if df[col].nunique() < max(20, len(df) * 0.05) and col not in BLACKLIST_COLS
    ]
    if candidates:
        return candidates[-1]

    # 3️⃣ Dernière colonne non blacklistée
    for col in reversed(df.columns):
        if col not in BLACKLIST_COLS:
            return col

    # 4️⃣ Fallback
    return df.columns[0]


def detect_task(y):
    if y.dtype == "object" or y.nunique() <= 20:
        return "classification"
    return "regression"

# --------------------------------------------------------
# CONFIG 
# --------------------------------------------------------
st.set_page_config(page_title="ML Dashboard Universel", layout="wide")
st.title("📊 Dashboard ML Universel – Classification & Régression")

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models/best_model.joblib"
FEATURES_PATH = PROJECT_DIR / "models/features.json"
TASK_PATH = PROJECT_DIR / "models/task.json"



# --------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Aller vers :",
    [
        "🏠 Explorer Dataset",
        "⚙️ Entraîner un modèle",
        "🔮 Prédiction CSV",
        "📊 Visualisation Modèle",
        "🏭 KPI & Rapport Métier"
    ]
)

# ========================================================
# PAGE 1 — EXPLORATION DATASET
# ========================================================
if page == "🏠 Explorer Dataset":
    uploaded = st.file_uploader("📂 Charger un CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        st.write(df.describe(include="all"))



# ========================================================
# PAGE 2 — ENTRAINEMENT DU MODELE
# ========================================================

elif page == "⚙️ Entraîner un modèle":
    import json, joblib
    import pandas as pd
    import streamlit as st
    from sklearn.preprocessing import RobustScaler, OneHotEncoder, PolynomialFeatures
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import roc_auc_score, roc_curve


    st.header("⚙️ Entraîner un modèle ML")

    uploaded = st.file_uploader("📂 Charger un dataset", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        # ===============================
        # 1️⃣ Sélection cible & features
        # ===============================
        st.subheader("🎯 Sélection des colonnes")
        
        # Modification
        # target = st.selectbox("Colonne cible (Y)", df.columns)
        target = auto_detect_target(df)
        y = df[target]
        task = detect_task(y)
        
        st.info(f"🧠 Colonne cible détectée automatiquement : **{target}**")
        st.info(f"🧠 Type de problème détecté automatiquement : **{task.upper()}**")
        
        if task == "classification" and y.nunique() < 2:
            st.error("❌ La cible n’a qu’une seule classe")
            st.stop()

        if task == "classification" and y.nunique() > 20:
            st.warning("⚠️ Trop de classes → classification risquée")

        st.info(f"🧠 Type de problème détecté automatiquement : **{task.upper()}**")
        
        
        target = st.selectbox(
            "Colonne cible détectée automatiquement",
            df.columns,
            index=list(df.columns).index(target)
        )
        
        # -------------------------
        # Features : tout sauf la cible et blacklist
        # -------------------------
        feature_cols = [c for c in df.columns if c != target and c not in BLACKLIST_COLS]
        X = df[feature_cols]
        
        
        num_features = X.select_dtypes(exclude=["object"]).columns.tolist()
        cat_features = X.select_dtypes(include=["object"]).columns.tolist()

        # Ca s'arrete la : 
        
        feature_cols = st.multiselect(
            "Colonnes explicatives (X)",
            df.columns.drop(target),
            default=list(df.columns.drop(target))
        )

        if len(feature_cols) == 0:
            st.warning("Sélectionnez au moins une feature")
            st.stop()

        X = df[feature_cols]
        y = df[target]

        # ===============================
        # 2 Num / Cat
        # ===============================
        
        degree = st.slider("Degré du polynôme (num features)", 1, 3, 1) 
        
        # Systeme de detection Colonne : 
        num_features = [c for c in feature_cols if df[c].dtype != "object"]
        cat_features = [c for c in feature_cols if df[c].dtype == "object"]
        
        st.success(f"✅ {len(feature_cols)} features détectées automatiquement : {len(num_features)} numériques, {len(cat_features)} catégorielles")
        # Création de la liste des transformations pour ColumnTransformer
        transformers_list = []

        if len(num_features) > 0:
            if task == "regression" and degree > 1:
                # Appliquer PolynomialFeatures + Scaler
                transformers_list.append(
                    ("num", Pipeline([
                        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                        ("scaler", RobustScaler())
                    ]), num_features)
                )
            else:
                # Juste scaler
                transformers_list.append(
                    ("num", RobustScaler(), num_features)
                )

        if len(cat_features) > 0:
            transformers_list.append(
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
            )

        # ColumnTransformer final
        preprocess = ColumnTransformer(transformers=transformers_list)
                
        preprocess = ColumnTransformer(
            transformers=[
                ("num", RobustScaler(), num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
            ]
        )

        # ===============================
        # 3️⃣ Train / Test split
        # ===============================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ===============================
        # 4️⃣ 🧪 COMPLEXITY PROBE
        # ===============================
        st.subheader("🧪 Complexity Probe – Decision Tree")

        depth_range = range(1, 21)
        train_auc, test_auc = [], []

        for d in depth_range:
            tree = DecisionTreeClassifier(max_depth=d, random_state=42)

            probe_pipeline = Pipeline([
                ("preprocess", preprocess),
                ("model", tree)
            ])

            probe_pipeline.fit(X_train, y_train)

            y_train_vec = y_train.copy().squeeze()
            y_test_vec = y_test.copy().squeeze()

            train_probs = probe_pipeline.predict_proba(X_train)
            test_probs = probe_pipeline.predict_proba(X_test)

            if task == "classification" and len(y_train_vec.unique()) == 2:
                # Binaire : prendre la colonne positive 
                train_auc.append(roc_auc_score(y_train_vec, train_probs[:, 1], multi_class="ovr"))
                test_auc.append(roc_auc_score(y_test_vec, test_probs[:, 1], multi_class="ovr"))
            else:
                # Multi-classe
                train_auc.append(roc_auc_score(y_train_vec, train_probs, multi_class="ovr"))
                test_auc.append(roc_auc_score(y_test_vec, test_probs, multi_class="ovr"))

        auc_df = pd.DataFrame({
            "max_depth": depth_range,
            "Train AUC": train_auc,
            "Test AUC": test_auc
        })

        st.line_chart(auc_df.set_index("max_depth"))

        best_depth = auc_df.loc[auc_df["Test AUC"].idxmax(), "max_depth"]
        gap_probe = max(train_auc) - max(test_auc)

        st.success(f"🎯 Profondeur optimale détectée : {best_depth}")

        if gap_probe > 0.1:
            st.warning("⚠️ Overfitting structurel détecté")

        # ===============================
        # 5️⃣ Choix du modèle final
        # ===============================
        st.subheader("🤖 Choix du modèle")

        model_choice = st.selectbox(
            "Modèle",
            [
                "RandomForest (Classification)",
                "RandomForest (Régression)",
                "Logistic Regression",
                "Régression Linéaire"
            ]
        )

        if st.button("🚀 Entraîner le modèle final"):

            if model_choice == "RandomForest (Classification)":
                model = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=best_depth,
                    min_samples_leaf=5,
                    random_state=42
                )
                task = "classification"

            elif model_choice == "RandomForest (Régression)":
                model = RandomForestRegressor(
                    n_estimators=300,
                    max_depth=best_depth,
                    min_samples_leaf=5,
                    random_state=42
                )
                task = "regression"

            elif model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=500)
                task = "classification"

            else:
                model = LinearRegression()
                task = "regression"

            pipeline = Pipeline([
                ("preprocess", preprocess),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)

            train_score = pipeline.score(X_train, y_train)
            test_score = pipeline.score(X_test, y_test)

            st.metric("Train Score", round(train_score, 3))
            st.metric("Test Score", round(test_score, 3))
            st.metric("Gap", round(train_score - test_score, 3))

            # ===============================
            # 6️⃣ Sauvegarde
            # ===============================
            joblib.dump(pipeline, MODEL_PATH)
            json.dump(feature_cols, open(FEATURES_PATH, "w"))
            json.dump({"task": task}, open(TASK_PATH, "w"))

            st.success("🎉 Modèle entraîné et sauvegardé !")


# ========================================================
# PAGE 3 — PREDICTION CSV
# ========================================================
elif page == "🔮 Prédiction CSV":
    from predict import predict

    uploaded = st.file_uploader("📂 Charger un CSV", type=["csv"], key="pred")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        try:
            df_pred = predict(df)
            st.success("🎉 Prédictions générées !")
            st.dataframe(df_pred.head())

            st.download_button(
                "⬇ Télécharger les prédictions",
                df_pred.to_csv(index=False),
                "predictions.csv",
                "text/csv"
            )
        except Exception as e:
            st.error(f"❌ Erreur : {e}")


# ========================================================
# PAGE 4 — VISUALISATION MODELE (STABLE & PRO)
# ========================================================

elif page == "📊 Visualisation Modèle":

    # ====================================================
    # 1️⃣ Chargement modèle & métadonnées
    # ====================================================
    
    
    if not MODEL_PATH.exists():
        st.error("❌ Aucun modèle trouvé. Entraînez un modèle d'abord.")
        st.stop()

    pipeline = joblib.load(MODEL_PATH)
    FEATURES = json.load(open(FEATURES_PATH))
    META = json.load(open(TASK_PATH))

    TASK = META["task"]
    TARGET = META.get("target")

    st.header("🧠 Modèle entraîné")
    st.write(pipeline.named_steps["model"])

    st.subheader("📌 Features utilisées")
    st.write(FEATURES)

    st.subheader("🎯 Type de tâche")
    st.success(TASK.upper())

    # ====================================================
    # 2️⃣ Upload dataset d’évaluation
    # ====================================================
    uploaded = st.file_uploader(
        "📂 Charger un dataset pour analyse & évaluation",
        type=["csv"],
        key="eval"
    )

    if not uploaded:
        st.stop()

    try:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, sep=None, engine="python")
        if df.empty:
            raise ValueError("Dataset vide")
    except Exception as e:
        st.error(f"❌ Erreur lecture CSV : {e}")
        st.stop()

    st.subheader("👀 Aperçu du dataset")
    st.dataframe(df.head())

    # ====================================================
    # 3️⃣ Vérification cible & features
    # ====================================================
    if TARGET is None:
        TARGET = st.text_input("Nom de la colonne cible")

    if not TARGET or TARGET not in df.columns:
        st.error(f"❌ La colonne cible '{TARGET}' est absente du dataset")
        st.stop()

    missing_features = set(FEATURES) - set(df.columns)
    if missing_features:
        st.error(f"❌ Features manquantes : {missing_features}")
        st.stop()

    X_eval = df[FEATURES].reindex(columns=FEATURES)
    y_true = df[TARGET]
    y_pred = pipeline.predict(X_eval)

    # ====================================================
    # 4️⃣ ÉVALUATION DU MODÈLE
    # ====================================================
    if TASK == "classification":
        st.subheader("🧪 Évaluation Classification")

        acc = accuracy_score(y_true, y_pred)
        st.metric("Accuracy", round(acc, 3))

        # ---------- ROC / AUC ----------
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            y_proba = pipeline.predict_proba(X_eval)
            n_classes = y_proba.shape[1]

            if n_classes == 2:
                st.subheader("📈 ROC Curve")
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                auc_score = auc(fpr, tpr)

                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
                ax.plot([0, 1], [0, 1], "k--")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
            else:
                auc_score = roc_auc_score(
                    y_true,
                    y_proba,
                    multi_class="ovr",
                    average="macro"
                )
                st.metric("ROC AUC (OvR)", round(auc_score, 3))

        # ---------- Confusion Matrix ----------
        st.subheader("📊 Matrice de confusion")
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        st.pyplot(fig)
        plt.close(fig)

    # ====================================================
    # REGRESSION
    # ====================================================
    else:
        st.subheader("🧪 Évaluation Régression")

        r2 = r2_score(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred)

        col1, col2 = st.columns(2)
        col1.metric("R²", round(r2, 3))
        col2.metric("RMSE", round(rmse, 3))

        st.subheader("📈 Réel vs Prédit")
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred, alpha=0.6)
        ax.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "r--"
        )
        ax.set_xlabel("Valeurs réelles")
        ax.set_ylabel("Valeurs prédites")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("📊 Résidus")
        residuals = y_true - y_pred

        fig, ax = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    # ====================================================
    # 5️⃣ DATASET EXPLORER (OPTIMISÉ) 
    # ====================================================
    st.header("🧠 Dataset Explorer")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    tabs = st.tabs([
        "🟦 Overview",
        "🟩 Distributions",
        "🟨 Relations",
        "🟥 Target",
        "🟪 Qualité"
    ])

    # ---------- OVERVIEW ----------
    with tabs[0]:
        st.metric("Lignes", df.shape[0])
        st.metric("Colonnes", df.shape[1])
        st.write(df.dtypes)
        st.subheader("Valeurs manquantes")
        st.bar_chart(df.isna().sum())

    # ---------- DISTRIBUTIONS ----------
    with tabs[1]:
        if num_cols:
            col = st.selectbox("Variable numérique", num_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)
            plt.close(fig)

        if cat_cols:
            col = st.selectbox("Variable catégorielle", cat_cols)
            st.bar_chart(df[col].value_counts())

    # ---------- RELATIONS ----------
    with tabs[2]:
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[num_cols].corr(), cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            plt.close(fig)

    # ---------- TARGET ----------
    with tabs[3]:
        if TARGET in num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[TARGET], kde=True, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.bar_chart(df[TARGET].value_counts())

    # ---------- QUALITÉ ----------
    with tabs[4]:
        if num_cols:
            col = st.selectbox("Variable pour outliers", num_cols)
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = df[
                (df[col] < q1 - 1.5 * iqr) |
                (df[col] > q3 + 1.5 * iqr)
            ]

            st.write(f"Outliers détectés : {len(outliers)}")

            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            st.pyplot(fig)
            plt.close(fig)

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # ========================================================
    # 🧠 EXPLICABILITÉ DU MODÈLE — SHAP (PRO)
    # ========================================================
    st.header("🧠 Explicabilité du modèle (SHAP)")

    # ----------------------------------------------------
    # Sécurité taille dataset
    # ----------------------------------------------------
    MAX_SHAP_SAMPLES = 500

    if len(X_eval) > MAX_SHAP_SAMPLES:
        st.warning(
            f"Dataset trop grand pour SHAP ({len(X_eval)} lignes). "
            f"Échantillonnage à {MAX_SHAP_SAMPLES} lignes."
        )
        X_shap = X_eval.sample(MAX_SHAP_SAMPLES, random_state=42)
    else:
        X_shap = X_eval.copy()

    # Transformation via le preprocess
    X_shap_transformed = pipeline.named_steps["preprocess"].transform(X_shap)
    model = pipeline.named_steps["model"]

    # ----------------------------------------------------
    # Choix Explainer
    # ----------------------------------------------------
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap_transformed)
    except Exception:
        explainer = shap.Explainer(model, X_shap_transformed)
        shap_values = explainer(X_shap_transformed)

    # ----------------------------------------------------
    # 1️⃣ IMPORTANCE GLOBALE DES FEATURES
    # ----------------------------------------------------
    st.subheader("📊 Importance globale des variables")

    fig, ax = plt.subplots()
    shap.summary_plot(
        shap_values,
        X_shap,
        plot_type="bar",
        show=False
    )
    st.pyplot(fig)
    plt.close(fig)

    # ----------------------------------------------------
    # 2️⃣ SHAP SUMMARY (distribution)
    # ----------------------------------------------------
    st.subheader("🌈 Impact des variables sur les prédictions")

    fig, ax = plt.subplots()
    shap.summary_plot(
        shap_values,
        X_shap,
        show=False
    )
    
    st.pyplot(fig)
    plt.close(fig)

    # ----------------------------------------------------
    # 3️⃣ SHAP FORCE PLOT — INDIVIDU
    # ----------------------------------------------------
    st.subheader("🔍 Explication individuelle")

    index = st.slider(
        "Choisir une observation",
        min_value=0,
        max_value=len(X_shap) - 1,
        value=0
    )

    st.write("📌 Valeurs de l'observation")
    st.dataframe(X_shap.iloc[[index]])

    shap.initjs()

    force_plot = shap.force_plot(
        explainer.expected_value[1] if isinstance(shap_values, list) else explainer.expected_value,
        shap_values[1][index] if isinstance(shap_values, list) else shap_values[index],
        X_shap.iloc[index],
        matplotlib=True
    )

    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

    # ----------------------------------------------------
    # 4️⃣ SHAP DEPENDENCE PLOT
    # ----------------------------------------------------
    st.subheader("📈 Relation feature ↔ prédiction")

    feature_dep = st.selectbox(
        "Choisir une variable",
        FEATURES
    )

    fig, ax = plt.subplots()
    shap.dependence_plot(
        feature_dep,
        shap_values[1] if isinstance(shap_values, list) else shap_values,
        X_shap,
        show=False
    )
    st.pyplot(fig)
    plt.close(fig)
    
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

elif page == "🏭 KPI & Rapport Métier":
    import numpy as np
    st.header("🏭 Tableau de bord métier & aide à la décision")

    # ----------------------------------------------------
    # 1️⃣ Chargement modèle & données
    # ----------------------------------------------------
    if not MODEL_PATH.exists():
        st.error("❌ Aucun modèle entraîné disponible.")
        st.stop()

    pipeline = joblib.load(MODEL_PATH)
    FEATURES = json.load(open(FEATURES_PATH))
    META = json.load(open(TASK_PATH))

    TASK = META["task"]
    TARGET = META.get("target")

    uploaded = st.file_uploader(
        "📂 Charger un dataset opérationnel",
        type=["csv"],
        key="kpi_data"
    )

    if not uploaded:
        st.info("⬆️ Charge un dataset pour générer les KPI métier")
        st.stop()

    df = pd.read_csv(uploaded)

    missing = set(FEATURES) - set(df.columns)
    if missing:
        st.error(f"❌ Features manquantes : {missing}")
        st.stop()

    X = df[FEATURES]
    y_pred = pipeline.predict(X)

    # ----------------------------------------------------
    # 2️⃣ KPI MÉTIER — CLASSIFICATION
    # ----------------------------------------------------
    if TASK == "classification":

        st.subheader("🚨 Indicateurs de risque (Classification)")

        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            proba = pipeline.predict_proba(X)[:, 1]
        else:
            st.warning("Le modèle ne fournit pas de probabilités.")
            st.stop()

        seuil = st.slider(
            "🎚️ Seuil d’alerte",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05
        )

        alerts = proba >= seuil

        col1, col2, col3 = st.columns(3)
        col1.metric("Équipements analysés", len(X))
        col2.metric("Alertes critiques", alerts.sum())
        col3.metric(
            "Taux de risque",
            f"{round(alerts.mean() * 100, 1)} %"
        )

        st.subheader("📊 Distribution des probabilités de panne")
        fig, ax = plt.subplots()
        sns.histplot(proba, kde=True, ax=ax)
        ax.axvline(seuil, color="red", linestyle="--")
        st.pyplot(fig)
        plt.close(fig)

    # ----------------------------------------------------
    # 3️⃣ KPI MÉTIER — RÉGRESSION
    # ----------------------------------------------------
    else:

        st.subheader("🛠️ Indicateurs de maintenance conditionnelle")

        seuil_critique = st.slider(
            "🎯 Seuil critique (ex : RUL minimal)",
            min_value=float(np.min(y_pred)),
            max_value=float(np.max(y_pred)),
            value=float(np.percentile(y_pred, 25))
        )

        critiques = y_pred <= seuil_critique

        col1, col2, col3 = st.columns(3)
        col1.metric("Équipements analysés", len(y_pred))
        col2.metric("Cas critiques", critiques.sum())
        col3.metric("RUL moyenne", round(np.mean(y_pred), 2))

        st.subheader("📉 Répartition des prédictions")
        fig, ax = plt.subplots()
        sns.histplot(y_pred, kde=True, ax=ax)
        ax.axvline(seuil_critique, color="red", linestyle="--")
        st.pyplot(fig)
        plt.close(fig)

    # ----------------------------------------------------
    # 4️⃣ TABLE OPÉRATIONNELLE
    # ----------------------------------------------------
    st.subheader("📋 Table décisionnelle")

    df_result = df.copy()
    df_result["Prediction"] = y_pred

    if TASK == "classification":
        df_result["Probabilité_risque"] = proba
        df_result["Alerte"] = alerts

    st.dataframe(df_result.head(50))

    # ----------------------------------------------------
    # 5️⃣ EXPORT CSV MÉTIER
    # ----------------------------------------------------
    st.download_button(
        "📥 Télécharger les résultats (CSV)",
        data=df_result.to_csv(index=False),
        file_name="resultats_kpi_metier.csv",
        mime="text/csv"
    )


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# ========================================================
# PAGE 4 — VISUALISATION MODELE
# ========================================================
# elif page == "📊 Visualisation Modèle":

#     # ====================================================
#     # 1️⃣ Chargement modèle & métadonnées
#     # ====================================================
#     if not MODEL_PATH.exists():
#         st.error("❌ Aucun modèle trouvé. Entraînez un modèle d'abord.")
#         st.stop()

#     pipeline = joblib.load(MODEL_PATH)
#     FEATURES = json.load(open(FEATURES_PATH))
#     META = json.load(open(TASK_PATH))

#     TASK = META["task"]
#     TARGET = META.get("target")

#     st.header("🧠 Modèle entraîné")
#     st.write(pipeline.named_steps["model"])

#     st.subheader("📌 Features utilisées")
#     st.write(FEATURES)

#     st.subheader("🎯 Type de tâche")
#     st.success(TASK.upper())

#     # ====================================================
#     # 2️⃣ Upload dataset d’évaluation
#     # ====================================================
#     uploaded = st.file_uploader(
#         "📂 Charger un dataset pour analyse & évaluation",
#         type=["csv"],
#         key="eval"
#     )

#     if not uploaded:
#         st.stop()

#     # Lecture CSV sécurisée
#     try:
#         uploaded.seek(0)
#         df = pd.read_csv(uploaded, sep=None, engine="python")
#         if df.empty or df.columns.size == 0:
#             raise ValueError("Dataset vide")
#     except Exception as e:
#         st.error(f"❌ Erreur lecture CSV : {e}")
#         st.stop()

#     st.subheader("👀 Aperçu du dataset")
#     st.dataframe(df.head())

#     # ====================================================
#     # 3️⃣ Vérification cible
#     # ====================================================
#     if TARGET is None:
#         TARGET = st.text_input("Nom de la colonne cible")

#     if not TARGET or TARGET not in df.columns:
#         st.error(f"❌ La colonne cible '{TARGET}' est absente du dataset")
#         st.stop()

#     X_eval = df[FEATURES]
#     y_true = df[TARGET]
#     y_pred = pipeline.predict(X_eval)

#     # ====================================================
#     # 4️⃣ ÉVALUATION DU MODÈLE
#     # ====================================================
#     if TASK == "classification":
#         st.subheader("🧪 Évaluation Classification")

#         col1, col2 = st.columns(2)
#         with col1:
#             acc = accuracy_score(y_true, y_pred)
#             st.metric("Accuracy", round(acc, 3))

#         # ---------- ROC / AUC ----------
        
#         if hasattr(pipeline.named_steps["model"], "predict_proba"):
#             y_proba = pipeline.predict_proba(X_eval)
#             n_classes = y_proba.shape[1]

#             if n_classes == 2:
#                 st.subheader("📈 ROC Curve")
#                 fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
#                 auc_score = auc(fpr, tpr)

#                 fig, ax = plt.subplots()
#                 ax.plot(fpr, tpr, label=f"AUC={auc_score:.2f}")
#                 ax.plot([0,1], [0,1], "k--")
#                 ax.legend()
#                 st.pyplot(fig)
#             else:
#                 auc_score = roc_auc_score(
#                     y_true,
#                     y_proba,
#                     multi_class="ovr",
#                     average="macro"
#                 )
#                 st.metric("ROC AUC (OvR)", round(auc_score, 3)) 

#         # ---------- Confusion Matrix ----------
#         st.subheader("📊 Matrice de confusion")
#         cm = confusion_matrix(y_true, y_pred)
#         fig, ax = plt.subplots()
#         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
#         st.pyplot(fig)

#     # ====================================================
#     # REGRESSION
#     # ====================================================
#     else:
#         st.subheader("🧪 Évaluation Régression")

#         r2 = r2_score(y_true, y_pred)
#         rmse = mean_squared_error(y_true, y_pred, squared=False)

#         col1, col2 = st.columns(2)
#         col1.metric("R²", round(r2, 3))
#         col2.metric("RMSE", round(rmse, 3))

#         st.subheader("📈 Réel vs Prédit")
#         fig, ax = plt.subplots()
#         ax.scatter(y_true, y_pred, alpha=0.6)
#         ax.plot(
#             [y_true.min(), y_true.max()],
#             [y_true.min(), y_true.max()],
#             "r--"
#         )
#         st.pyplot(fig)

#         st.subheader("📊 Résidus")
#         residuals = y_true - y_pred
#         fig, ax = plt.subplots()
#         sns.histplot(residuals, kde=True, ax=ax)
#         st.pyplot(fig)

#         st.subheader("🧪 Résidus vs Prédictions")
#         fig, ax = plt.subplots()
#         ax.scatter(y_pred, residuals)
#         ax.axhline(0, color="red", linestyle="--")
#         st.pyplot(fig)

#     # ====================================================
#     # 5️⃣ DATASET EXPLORER 
#     # ====================================================
#     st.header("🧠 Dataset Explorer")

#     num_cols = df.select_dtypes(exclude="object").columns.tolist()
#     cat_cols = df.select_dtypes(include="object").columns.tolist()

#     tabs = st.tabs([
#         "🟦 Overview",
#         "🟩 Distributions",
#         "🟨 Relations",
#         "🟥 Target",
#         "🟪 Qualité"
#     ])

#     with tabs[0]:
#         st.metric("Lignes", df.shape[0])
#         st.metric("Colonnes", df.shape[1])
#         st.write(df.dtypes)
#         st.subheader("Valeurs manquantes")
#         st.bar_chart(df.isna().sum())

#     with tabs[1]:
#         for col in num_cols:
#             st.write(f"🔢 {col}")
#             # st.pyplot(sns.histplot(df[col], kde=True).figure)
#             st.pyplot(sns.boxplot(df[col]).figure)
#         for col in cat_cols:
#             st.write(f"🏷️ {col}")
#             st.bar_chart(df[col].value_counts())

#     with tabs[2]:
#         if len(num_cols) > 1:
#             fig, ax = plt.subplots()
#             sns.heatmap(df[num_cols].corr(), cmap="coolwarm", ax=ax)
#             st.pyplot(fig)

#     with tabs[3]:
#         st.subheader("🎯 Analyse de la cible")
#         if TARGET in num_cols:
#             st.pyplot(sns.histplot(df[TARGET], kde=True).figure)
#         else:
#             st.bar_chart(df[TARGET].value_counts())

#     with tabs[4]:
#         for col in num_cols:
#             q1, q3 = df[col].quantile([0.25, 0.75])
#             iqr = q3 - q1
#             outliers = df[
#                 (df[col] < q1 - 1.5 * iqr) |
#                 (df[col] > q3 + 1.5 * iqr)
#             ]
#             st.write(f"{col} → {len(outliers)} outliers")
#             st.pyplot(sns.boxplot(df[col]).figure)

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# elif page == "📊 Visualisation Modèle":

#     # ===============================
#     # 1️⃣ Chargement modèle & meta
#     # ===============================
#     if not MODEL_PATH.exists():
#         st.error("❌ Aucun modèle trouvé. Entraînez un modèle d'abord.")
#         st.stop()

#     pipeline = joblib.load(MODEL_PATH)
#     FEATURES = json.load(open(FEATURES_PATH))
#     META = json.load(open(TASK_PATH))

#     TASK = META["task"]
#     TARGET = META.get("target", None)

#     st.header("🧠 Modèle entraîné")
#     st.write(pipeline.named_steps["model"])

#     st.subheader("📌 Features utilisées")
#     st.write(FEATURES)

#     st.subheader("🎯 Type de tâche")
#     st.success(TASK.upper())
    
#     uploaded = st.file_uploader(
#         "📂 Charger un dataset pour analyse & évaluation",
#         type=["csv"],
#         key="eval"
#     )
    
#     if uploaded:
#         df = pd.read_csv(uploaded)
#         target = st.text_input("Nom de la colonne cible", "")

#         if target and target in df.columns:
#             X_eval = df[FEATURES]
#             y_true = df[target]
#             y_pred = pipeline.predict(X_eval)
            
#         if not uploaded:
#             st.stop()

#         df = pd.read_csv(uploaded)
#         st.dataframe(df.head())

#         if TARGET not in df.columns:
#             st.error(f"❌ La colonne cible '{TARGET}' est absente du dataset")
#             st.stop()

#         X_eval = df[FEATURES]
#         y_true = df[TARGET]
#         y_pred = pipeline.predict(X_eval)

#         # ============================================
#         # CLASSIFICATION
#         # ============================================
#         if TASK == "classification":
#             st.subheader("🧪 Évaluation Classification")

#             col1, col2 = st.columns(2)

#             with col1:
#                 acc = accuracy_score(y_true, y_pred)
#                 st.metric("Accuracy", round(acc, 3))

#             if hasattr(pipeline.named_steps["model"], "predict_proba"):
#                 y_proba = pipeline.predict_proba(X_eval)

#                 if y_proba.shape[1] == 2:
#                     fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
#                     auc_score = auc(fpr, tpr)

#                     fig, ax = plt.subplots()
#                     ax.plot(fpr, tpr, label=f"AUC={auc_score:.2f}")
#                     ax.plot([0,1], [0,1], "k--")
#                     ax.legend()
#                     ax.set_title("ROC Curve")
#                     st.pyplot(fig)
#                 else:
#                     auc_score = roc_auc_score(y_true, y_proba, multi_class="ovr")
#                     st.metric("ROC AUC (OvR)", round(auc_score, 3))

#             st.subheader("📊 Matrice de confusion")
#             cm = confusion_matrix(y_true, y_pred)
#             fig, ax = plt.subplots()
#             sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
#             st.pyplot(fig)
            
#             # ============================================
#             # REGRESSION
#             # ============================================
#         else:
#             st.subheader("🧪 Évaluation Régression")

#             r2 = r2_score(y_true, y_pred)
#             rmse = mean_squared_error(y_true, y_pred, squared=False)

#             col1, col2 = st.columns(2)
#             col1.metric("R²", round(r2, 3))
#             col2.metric("RMSE", round(rmse, 3))

#             st.subheader("📈 Réel vs Prédit")
#             fig, ax = plt.subplots()
#             ax.scatter(y_true, y_pred, alpha=0.6)
#             ax.plot(
#                 [y_true.min(), y_true.max()],
#                 [y_true.min(), y_true.max()],
#                 "r--"
#             )
#             st.pyplot(fig)

#             st.subheader("📊 Résidus")
#             residuals = y_true - y_pred
#             fig, ax = plt.subplots()
#             sns.histplot(residuals, kde=True, ax=ax)
#             st.pyplot(fig)

#         st.subheader("🧪 Résidus vs Prédictions (Overfitting)")
#         fig, ax = plt.subplots()
#         ax.scatter(y_pred, residuals)
#         ax.axhline(0, color="red", linestyle="--")
#         st.pyplot(fig)

#     st.header("🧠 Dataset Explorer")

#     num_cols = df.select_dtypes(exclude="object").columns.tolist()
#     cat_cols = df.select_dtypes(include="object").columns.tolist()

#     tabs = st.tabs([
#         "🟦 Overview",
#         "🟩 Distributions",
#         "🟨 Relations",
#         "🟥 Target",
#         "🟪 Qualité"
#     ])

#     with tabs[0]:
#         st.metric("Lignes", df.shape[0])
#         st.metric("Colonnes", df.shape[1])
#         st.write(df.dtypes)
#         st.subheader("Valeurs manquantes")
#         st.bar_chart(df.isna().sum())
    
#     with tabs[1]:
#         for col in num_cols:
#             st.write(f"🔢 {col}")
#             st.pyplot(sns.histplot(df[col], kde=True).figure)

#         for col in cat_cols:
#             st.write(f"🏷️ {col}")
#             st.bar_chart(df[col].value_counts())
    
#     with tabs[2]:
#         if len(num_cols) > 1:
#             fig, ax = plt.subplots()
#             sns.heatmap(df[num_cols].corr(), cmap="coolwarm", ax=ax)
#             st.pyplot(fig)
    
#     with tabs[3]:
#         st.subheader("🎯 Analyse de la cible")

#         if TARGET in num_cols:
#             st.pyplot(sns.histplot(df[TARGET], kde=True).figure)
#         else:
#             st.bar_chart(df[TARGET].value_counts())

#         for col in num_cols:
#             if col != TARGET:
#                 st.pyplot(sns.boxplot(x=df[TARGET], y=df[col]).figure)
    
#     with tabs[4]:
#         for col in num_cols:
#             q1, q3 = df[col].quantile([0.25, 0.75])
#             iqr = q3 - q1
#             outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]
#             st.write(f"{col} → {len(outliers)} outliers")
#             st.pyplot(sns.boxplot(df[col]).figure)
                
                