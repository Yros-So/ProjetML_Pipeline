# ** - Système hydraulique industriel **
Voici un **README.md clair, structuré et professionnel** pour ton projet **ProjetML_Pipeline** sur GitHub. Il couvre :

✅ Présentation du projet
✅ Structure du dépôt
✅ Installation
✅ Utilisation / exécution
✅ Fonctionnalités
✅ Conseils

Tu peux copier-coller ce contenu dans ton fichier `README.md` pour que ton dépôt soit facile à comprendre et à utiliser.

---

```markdown
# ProjetML_Pipeline

## 🚀 Présentation

**ProjetML_Pipeline** est une application de **maintenance prédictive pour systèmes hydrauliques industriels**.  
L’objectif est de prédire l’état de fonctionnement d’équipements à partir de données capteurs afin d’anticiper les pannes et d’optimiser les opérations de maintenance.

La solution combine :
- Machine Learning (classification & régression),
- Visualisation interactive via Streamlit,
- Explicabilité des prédictions avec SHAP,
- Génération de KPIs métier et recommandations automatiques.

Ce projet a été développé dans le cadre d’un système hydraulique industriel pour répondre aux enjeux de fiabilité, de coûts et d’exploitation. :contentReference[oaicite:0]{index=0}

---

## 📁 Structure du projet

```

ProjetML_Pipeline/
├── DatasetZuMa/                 # Données du banc d’essai hydraulique
├── ETL_Pipeline/                # Scripts de préparation des données
├── models/                     # Modèles sauvegardés (après entraînement)
├── project/project-churn/src/app_streamlit.py            # Tableau de bord interactif
├── predict.py                  # Fonction de prédiction à partir d’un modèle entrainé
├── requirements.txt            # Dépendances Python
└── README.md                   # Ce fichier

````

---

## 🛠️ Prérequis

Avant d’installer et d’exécuter l’application, assurez-vous d’avoir :

- Python **>= 3.8**
- `pip` installé
- Optionnel : un environnement virtuel (recommandé)

---

## 📦 Installation

1. **Cloner le dépôt**

```sh
git clone https://github.com/Yros-So/ProjetML_Pipeline.git
cd ProjetML_Pipeline
````

2. **Créer un environnement virtuel (optionnel mais recommandé)**

```sh
python -m venv venv
```

3. **Activer l’environnement**

➡ Sur macOS/Linux

```sh
source venv/bin/activate
```

➡ Sur Windows (PowerShell)

```sh
.\venv\Scripts\Activate
```

4. **Installer les dépendances**

```sh
pip install -r requirements.txt
```

---

## 🚀 Exécution

### 📌 Lancer le tableau de bord

L’interface principale est développée avec **Streamlit**.

```sh
streamlit run project-churn/src/app_streamlit.py
```

Une page web s’ouvrira automatiquement (souvent à l’adresse [http://localhost:8501](http://localhost:8501)).

---

## 🧩 Fonctionnalités principales

### 🏠 Explorer le dataset

* Charger un fichier CSV
* Visualisation des premières lignes
* Statistiques descriptives
* Graphiques d’exploration (distributions, corrélations, outliers)

---

### ⚙️ Entraîner un modèle

* Détection automatique de la cible
* Choix du type de modèle

  * Random Forest (Classification / Régression)
  * Régression Logistique
  * Régression Linéaire
* Entraînement et sauvegarde automatique
* Évaluation de performance

---

### 🔮 Prédictions en batch

* Charger un CSV d’équipements
* Générer des prédictions
* Exporter les résultats au format CSV
* Visualiser les probabilités / valeurs prévues

---

### 📊 Visualisation du modèle

* Chargement d’un dataset d’évaluation
* Courbes ROC, matrices de confusion
* Graphiques de régression (réel vs prédit)
* Analyse visuelle des caractéristiques

---

### 🧠 Explicabilité du modèle

* Importance des variables avec **SHAP**
* Résumé global des contributions
* Graphiques de dépendance
* Explication individuelle des prédictions

---

### 🏭 KPI & Recommandations métier

Pour un dataset opérationnel :

* Calcul des KPI (nombre de risques, taux de panne)
* Graphiques de probabilité et distribution
* Recommandations automatiques de maintenance
* Export des résultats

---

## 🧠 Comment ça marche (concept global)

1. **Préparation des données**

   * Nettoyage, transformation, features engineering

2. **Machine Learning**

   * Modèles entraînés avec cross-validation

3. **Visualisation & Aide à la décision**

   * Streamlit offre une interface complète pour l’analyse
   * SHAP rend les résultats explicables

4. **KPI métier + recommandations**

   * Fonctions analytiques pour décisions opérationnelles

---

## 📌 Bonnes pratiques

* Assurez-vous que vos données soient bien formatées (colonnes cohérentes)
* Utilisez des datasets représentatifs pour de meilleures prédictions
* Validez les modèles sur des jeux de données réels avant production

---

## ❓ FAQ rapide

**Pourquoi utiliser Random Forest ?**
Random Forest est robuste aux interactions complexes entre variables et offre une interprétabilité acceptable avec SHAP.

**Puis-je réentraîner avec mes propres données ?**
Oui. Chargez simplement votre CSV dans la section “⚙️ Entraîner un modèle” de l’application.

---

## 📜 Licence

Ce projet est libre et open-source.

---

## 📬 Contact

Tu peux me contacter via mon profil GitHub.

---

```

---

### Notes importantes

- Ce README reflète précisément **le contenu affiché du dépôt GitHub** que tu as partagé. :contentReference[oaicite:1]{index=1}  
- Il est conçu pour être **clair pour un lecteur externe**, même sans connaissance préalable du projet.

---

Si tu veux, je peux aussi générer :

✅ un **Fichier CONTRIBUTING.md** pour guider les contributeurs  
✅ un **Document d’architecture** décrivant le pipeline complet  
✅ un **Guide utilisateur** plus visuel

Dis-moi ce que tu souhaites ensuite ! 🚀
::contentReference[oaicite:2]{index=2}
```

## 1. Contexte

Le système étudié est un banc d’essai hydraulique instrumenté par plusieurs capteurs (pression, température, débit, vibration, puissance).
L’objectif du projet est d’utiliser une approche de **maintenance prédictive par Machine Learning** afin d’anticiper les pannes et d’éviter les arrêts non planifiés .

Les analyses réalisées ont montré que les défaillances ne proviennent pas uniquement de l’usure d’un composant isolé, mais souvent de l’**interaction entre plusieurs paramètres dynamiques du système**.

---

## 2. Problématique constatée

Le système hydraulique présente les risques suivants :

* Pannes imprévues entraînant des arrêts de production.
* Coûts élevés liés aux réparations urgentes.
* Dépendance à une maintenance réactive, peu efficace .

L’analyse des données capteurs a notamment mis en évidence :

* Des **pics anormaux sur le capteur FS1_max**, traduisant des contraintes mécaniques excessives.
* Des interactions complexes entre l’état de la valve et les conditions hydrauliques transitoires.
* Une dégradation parfois indépendante de l’état théorique des composants .

---

## 3. Diagnostic principal

Les modèles interprétables (SHAP) ont révélé que :

* La défaillance observée n’est **pas uniquement due à la valve**.
* Elle résulte principalement de :

  * pics de pression anormaux,
  * surcharges mécaniques transitoires,
  * instabilité du circuit hydraulique.

Ainsi, un système peut présenter un composant encore en bon état apparent mais être malgré tout en situation de risque à cause des conditions de fonctionnement .

---

# 4. Solutions à apporter

Les actions correctives recommandées se divisent en deux volets : **préventif et opérationnel**.

---

### A. Solutions de maintenance prédictive

1. **Surveillance en temps réel**

   * Mettre en place un suivi continu de l’indicateur critique **FS1_max**.
   * Définir des seuils d’alerte automatiques.

2. Détection proactive

   * Déclencher des interventions conditionnelles basées sur :

     * pics anormaux,
     * forte variabilité des mesures,
     * non uniquement sur les moyennes.

3. Intégration du modèle ML

   * Déployer le modèle prédictif dans l’environnement industriel pour anticiper les défaillances avant qu’elles ne surviennent .

---

### B. Solutions terrain immédiates

Afin de réduire directement les risques identifiés :

* Inspection ciblée des composants suivants :

  * clapets anti-retour,
  * amortisseurs hydrauliques,
  * soupapes de sécurité.

* Vérifications mécaniques :

  * serrage des raccords,
  * état des joints,
  * contrôle du débit de la pompe.

* Actions correctives :

  * purge complète du circuit pour éliminer l’air (source d’amplification des pics),
  * réduction des chocs hydrauliques par réglages de pression,
  * contrôle des amortissements mécaniques .

---

# 5. Résultats attendus

La mise en œuvre de ces solutions permettrait :

* Réduction estimée des pannes : **≈ 20%**
* Diminution des coûts de maintenance : **≈ 15%**
* Amélioration de la disponibilité des équipements : **≈ 10%** 

---

## 6. Conclusion

Le système hydraulique présente des faiblesses liées principalement aux **contraintes dynamiques** plutôt qu’à une simple usure de composants.

La combinaison :

* d’une surveillance intelligente basée sur Machine Learning,
* d’interventions ciblées sur le circuit hydraulique,
* d’une maintenance conditionnelle pilotée par données,

constitue une solution robuste pour améliorer la fiabilité globale du système et limiter fortement les arrêts imprévus .

---

Si vous le souhaitez, je peux :

* adapter ce rapport au format Word/PDF,
* le personnaliser pour une présentation professionnelle,
* ou vous aider à rédiger un plan d’action détaillé spécifique à votre installation réelle.
