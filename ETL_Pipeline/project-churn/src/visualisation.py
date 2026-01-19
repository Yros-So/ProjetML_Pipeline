import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import roc_curve, auc 

import sys
from pathlib import Path

# Ajouter automatiquement la racine du projet au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))


# ============================================
# 1. Utilitaires génériques
# ============================================

def ensure_str(df: pd.DataFrame) -> pd.DataFrame:
    """
        Convertit toutes les colonnes non numériques en string pour éviter les erreurs.
    """
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df


# ============================================
# 2. Visualisations exploratoires
# ============================================

def plot_churn_distribution(df: pd.DataFrame):
    """
        Histogramme du churn.
    """
    df = ensure_str(df)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=df, x="Churn", ax=ax)
    ax.set_title("Distribution du Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Nombre de clients")
    plt.tight_layout()

    return fig


def plot_numerical_distribution(df: pd.DataFrame, column: str):
    """
    Distribution d'une variable numérique.
    """
    if column not in df.columns:
        raise ValueError(f"La colonne {column} n'existe pas.")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f"Distribution de {column}")
    plt.tight_layout()

    return fig


def plot_correlation_matrix(df: pd.DataFrame):
    """
    Matrice de corrélation pour toutes les variables numériques.
    """
    numeric_df = df.select_dtypes(include=["float", "int"])

    if numeric_df.empty:
        raise ValueError("Aucune colonne numérique disponible pour la corrélation.")

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    ax.set_title("Matrice de corrélation")
    plt.tight_layout()

    return fig


# ============================================
# 3. Visualisation des prédictions
# ============================================

def plot_roc_curve(y_true, y_pred_proba):
    """
    Courbe ROC avec AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.tight_layout()

    return fig


def plot_predictions_vs_truth(df: pd.DataFrame, y_true_col="Churn", y_pred_col="Prediction"):
    """
    Comparaison entre valeurs réelles et prédites.
    """
    df = ensure_str(df)

    if y_true_col not in df.columns or y_pred_col not in df.columns:
        raise ValueError("Les colonnes Churn et Prediction doivent exister.")

    fig = px.histogram(
        df,
        x=y_pred_col,
        color=y_true_col,
        barmode="group",
        title="Prédictions vs Valeurs réelles"
    )

    return fig


# ============================================
# 4. Importance des features
# ============================================

def plot_feature_importance(feature_names, importances):
    """
    Importance des features (RandomForest, XGBoost).
    """
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importance"
    )

    return fig


# ============================================
# 5. Visualisation interactive (Plotly)
# ============================================

def plot_interactive_scatter(df, x, y, color="Churn"):
    """
    Scatter interactif Plotly.
    """
    df = ensure_str(df)

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        title=f"{x} vs {y}"
    )

    return fig

# Fin de visualisation.py
