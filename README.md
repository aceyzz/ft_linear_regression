<img title="42_ft-linear-regression" alt="42_ft-linear-regression" src="./utils/banner.png" width="100%">

<br>

# `ft_linear_regression`

Prédire le prix d’une voiture à partir de son kilométrage avec une **régression linéaire simple** et **descente de gradient** (pas de librairies “magiques”).

<br>

## Sommaire
- [Objectifs du projet](#objectifs-du-projet)
- [Installation & environnement](#installation--environnement)
- [Lancement rapide](#lancement-rapide)
- [Fonctionnalités principales](#fonctionnalités-principales)
- [Bonus implémentés](#bonus-implémentés)
- [Détails d’implémentation](#détails-dimplémentation)
- [Architecture du repo](#architecture-du-repo)
- [Choix techniques & limites](#choix-techniques--limites)
- [Évaluation](#évaluation)
- [Liens utiles](#liens-utiles)
- [Grade](#grade)

<br>

## Objectifs du projet
- Implémenter **l’hypothèse**: `estimatePrice(km) = theta0 + theta1 * km` (hypothèse du sujet).
- Entraîner `theta0` et `theta1` par **descente de gradient** sur un dataset `km,price`.
- Sauvegarder les **paramètres appris** dans `data/theta.json` et les **réutiliser** pour prédire.
- (Optionnel) Visualiser les **données** et la **droite apprise** sur un graphique.

<br>

## Installation & environnement

### 1. Cloner et configurer l'environnement
```bash
git clone https://github.com/aceyzz/ft_linear_regression.git
cd ft_linear_regression/project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Lancer l'entrainement
```bash
python3 train.py
```
- Lit `data/data.csv` (colonnes `km,price`).
- Entraîne `theta0, theta1`.
- Écrit `data/theta.json`.

### 3. Lancer la prédiction
```bash
python3 predict.py
# OR
python3 predict.py 150000 # ou 150000 est le nombre de km
```

### 4. Activer les bonus  

Ajustez la variable `BONUS_ENABLED` à `True` dans [le programme d'entrainement](./project/train.py) et [le programme de prédiction](./project/predict.py) (ligne 6)

5. Nettoyer le cache (facultatif)
```bash
# A la racine du repo
chmod +x clean.sh
./clean.sh
```

<br>

## Fonctionnalités principales
- **Prediction CLI** : saisie d’un kilométrage → **prix estimé**
- **Training CLI** : lecture CSV, **descente de gradient**, sauvegarde des poids
- **Logs** lisibles (couleurs, titres)
- **Robustesse** CSV (entête exigée, valeurs numériques, pas de négatifs)
- **Alerte divergence** / **stop** si nan/inf
- **Normalisation min–max** interne des `km` pour un entraînement stable (cf. “Choix techniques”)

<br>

## Bonus implémentés
- Graphique `matplotlib` :
  - nuage de points du dataset
  - droite du modèle appris
  - zone d’entraînement sur l’axe des `km`
  - **point de prédiction** (en rouge) annoté
  - **style** moderne
- `predict.py` supporte **mode interactif** (si aucun argument)

<br>

## Détails d’implémentation

- **Hypothèse / modèle**
  - `ŷ = theta0 + theta1 * x_norm`  
    Pour la stabilité numérique, on **entraîne sur `km` normalisés** en `[0,1]`.  
    On sauvegarde `x_min` et `x_max` pour normaliser toute **nouvelle** entrée à la prédiction.
  - `theta_handler.py` gère lecture/écriture de `data/theta.json` :
    ```json
    { "theta0": ..., "theta1": ..., "x_min": ..., "x_max": ... }
    ```

- **Descente de gradient**
  - Mise à jour simultanée de `theta0, theta1`
  - `mse` pour monitorer la progression
  - Arrêt si `nan/inf` ou si convergence activée (`ENABLED_TOL`)

- **Graphique (bonus)**
  - `sources/bonus/graph_handler.py` : API `render_graph(...)` appelée depuis `train.py` et `predict.py`
  - Affiche la droite `ŷ(x)` échantillonnée de `min(km)` à `max(km)`

<br>

## Architecture du repo
```
project/
├─ data/
│  ├─ data.csv             # dataset (km,price)
│  └─ theta.json           # paramètres appris
├─ sources/
│  ├─ csv_handler.py       # lecture/validation CSV -> Dataset
│  ├─ theta_handler.py     # I/O des thetas + min/max
│  └─ bonus/
│     └─ graph_handler.py  # rendu matplotlib (dark)
├─ utils/
│  ├─ console.py           # helpers console (couleurs/titres)
│  └─ __init__.py
├─ predict.py              # CLI prédiction
├─ train.py                # CLI entraînement
└─ requirements.txt        # matplotlib
```

<br>

## Choix techniques & limites

- **Normalisation min–max**  
  - Objectif : **stabiliser le gradient** et éviter les `NaN` (km sont très grands vs. prix).  
  - Nous **n’altérons pas l’hypothèse** du sujet (qui est bien linéaire) : la linéarité est **préservée** par un changement d’échelle.  
  - `theta.json` inclut `x_min/x_max` pour que `predict.py` applique la **même** normalisation qu’en training.
- **Hors plage**  
  - Si vous prédisez un `km` **bien au‑delà** du dataset, la droite extrapole (prix possiblement < 0). C’est attendu pour une **régression linéaire**.
- **Librairies**  
  - Interdiction d’utiliser des fonctions qui “font tout”. Ici, **pas de `numpy.polyfit`**. L’algorithme est **codé à la main**.

<br>

## Évaluation

- **2 programmes distincts** : `predict.py` et `train.py` ✔️
- **Forme de l’hypothèse** : `theta0 + theta1 * x` (ici `x` est `km` **normalisé** avec `x_min/x_max` appris) ✔️
- **Formules du training** : descente de gradient, **mise à jour simultanée** de `theta0, theta1` ✔️
- **Lecture CSV** : `km,price` en entête, parsing robuste ✔️
- **Sauvegarde** : `theta0` et `theta1` persistés (plus `x_min/x_max`) ✔️
- **Prédiction avant training** : renvoie 0 au début si `theta.json` vierge (ou valeurs par défaut) ✔️
- **Bonus** (si mandatory parfait) : graphiques de données + droite, précision éventuelle, etc. ✔️

<br>

## Liens utiles

Qu'est ce que le Machine Learning par 'Machine Learnia', voir [playlist YouTube](https://www.youtube.com/watch?v=EUD07IiviJg&list=PLO_fdPEVlfKqUF5BPKjGSh7aV9aBshrpY)

<br>

## Grade

> En cours d'evaluation
