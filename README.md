# Projet Deep Learning — EMSI Casablanca

**Module :** Deep Learning — Année universitaire 2025–2026  
**Auteur :** Abderrahmane Nahhal  
**Établissement :** EMSI Casablanca

---

## Idée générale du projet

Ce projet constitue l’évaluation finale du module Deep Learning. Il compare **trois familles de modèles** sur des données réelles de nature différente, afin de montrer comment le paradigme d’apprentissage supervisé doit s’adapter à la structure des données :

| Partie | Type de données | Modèle | Dataset |
|--------|-----------------|--------|---------|
| **I** | Tabulaire | MLP (PyTorch) | Breast Cancer Wisconsin (`sklearn`) |
| **II** | Images | CNN type LeNet | CIFAR-10 (`torchvision`) |
| **III** | Séquences / texte | RNN, LSTM, GRU, Seq2Seq | Corpus parallèle fra-eng (ManyThings) |

Chaque partie comprend : fondements théoriques, implémentation PyTorch, expérimentations comparatives, analyse critique et question de synthèse.

---

## Structure du repository

```
deeplearningnew/
│
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances Python
├── main.py                            # Script principal (lance les 3 parties)
│
├── partie1_MLP_tabulaire.ipynb        # Notebook Partie I (MLP)
├── partie2_CNN_images.ipynb           # Notebook Partie II (CNN)
├── partie3_RNN_Seq2Seq.ipynb          # Notebook Partie III (RNN / Seq2Seq)
│
├── src/                               # Code source commenté
│   ├── partie1_mlp.py               # Pipeline MLP
│   ├── partie2_cnn.py               # Pipeline CNN
│   ├── partie3_seq2seq.py           # Pipeline Seq2Seq
│   ├── corpus.py                    # Chargement corpus fra-eng
│   ├── vocab_utils.py               # Construction vocabulaire
│   └── experiment_io.py             # Export figures / tableaux → annexe/
│
├── rapport/                           # Synthèse et résultats
│   ├── rapport_scientifique.pdf       # Rapport final (à ajouter)
│   └── resultats_experimentaux.json # Métriques JSON des 3 parties
│
├── annexe/                            # Annexe expérimentale (livrable 4)
│   ├── partie1/
│   │   ├── figures/                 # Courbes, matrices de confusion
│   │   └── tableaux/                # CSV, métriques, rapports
│   ├── partie2/
│   │   ├── figures/                 # MLP vs CNN, feature maps
│   │   └── tableaux/
│   ├── partie3/
│   │   ├── figures/                 # Perplexité, perte Seq2Seq
│   │   └── tableaux/
│   └── summary_global.json          # Résumé global des expériences
│
├── checkpoints/                       # Modèles sauvegardés (Partie I)
│   └── mlp_breast_cancer.pth
│
└── data/                              # Données texte (Partie III)
    └── fra-eng/
        └── fra.txt                    # Corpus EN-FR (téléchargé automatiquement)
```

> **Note :** le dossier `data_cifar/` (CIFAR-10, ~170 Mo) n’est pas versionné : il se télécharge automatiquement au premier lancement.

---

## Installation et exécution

### Prérequis

- Python 3.10 ou supérieur (testé sur Python 3.14)
- `pip`

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Lancer tout le projet (script principal)

```bash
python main.py
```

Options utiles :

```bash
python main.py --partie 1              # MLP uniquement
python main.py --partie 2              # CNN uniquement
python main.py --partie 3              # Seq2Seq uniquement
python main.py --epochs-mlp 50 --epochs-cnn 8
```

### Lancer les notebooks

Ouvrir les fichiers `.ipynb` dans Jupyter ou VS Code / Cursor, puis **Run All** (noyau Python).

```bash
jupyter notebook
```

Les résultats sont enregistrés dans les notebooks et exportés vers `annexe/`.

---

## Livrables du projet

Conformément au sujet EMSI (§9) :

| # | Livrable | Emplacement |
|---|----------|-------------|
| 1 | Rapport scientifique (PDF) | `rapport/rapport_scientifique.pdf` |
| 2 | Code source commenté | `src/` |
| 3 | Notebooks / script exécutable | `partie*.ipynb` + `main.py` |
| 4 | Annexe expérimentale | `annexe/` (figures, CSV, métriques) |

---

## Résultats principaux (indicatifs)

| Partie | Métrique clé | Résultat |
|--------|--------------|----------|
| I — MLP | Accuracy test (Custom) | ~93 % |
| II — CNN | Accuracy test vs MLP | ~49 % vs ~44 % |
| III — Seq2Seq | BLEU (glouton) | ~26 |

Détails complets : `rapport/resultats_experimentaux.json` et `annexe/summary_global.json`.

---

## Fichiers volumineux et Git LFS

Les datasets **CIFAR-10** et l’archive **fra-eng.zip** ne sont pas inclus dans le repository (trop lourds). Ils sont téléchargés automatiquement à l’exécution. Les figures de l’annexe (~50–65 Ko chacune) sont versionnées normalement sans Git LFS.

Si vous ajoutez des données > 100 Mo, installez [Git LFS](https://git-lfs.github.com/) :

```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

---

## Auteur

Abderrahmane Nahhal — EMSI Casablanca — Deep Learning 2025–2026


