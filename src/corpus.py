"""
Chargement d'un corpus parallèle EN-FR (ManyThings / Tatoeba style).

Compatible Python 3.14 sans torchtext récent : télécharge fra-eng.zip une fois
et met en cache sous data/fra-eng/.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from .experiment_io import PROJECT_ROOT

FRA_ENG_URL = "https://www.manythings.org/anki/fra-eng.zip"
DATA_DIR = PROJECT_ROOT / "data" / "fra-eng"

# Corpus minimal intégré si le téléchargement échoue (hors ligne)
FALLBACK_PAIRS = [
    ("I am a student", "je suis etudiant"),
    ("We are learning deep learning", "nous apprenons le deep learning"),
    ("The cat is on the table", "le chat est sur la table"),
    ("It is raining today", "il pleut aujourd hui"),
    ("She likes machine learning", "elle aime le machine learning"),
    ("They work at the university", "ils travaillent a l universite"),
    ("Open the door please", "ouvre la porte s il te plait"),
    ("I do not understand", "je ne comprends pas"),
    ("This model works well", "ce modele fonctionne bien"),
    ("Training takes a long time", "l entrainement prend beaucoup de temps"),
    ("Python is a popular language", "python est un langage populaire"),
    ("We use pytorch for neural networks", "nous utilisons pytorch pour les reseaux"),
    ("The data must be normalized", "les donnees doivent etre normalisees"),
    ("Accuracy improved after ten epochs", "la precision s est amelioree apres dix epoques"),
    ("Gradient clipping stabilizes training", "le gradient clipping stabilise l entrainement"),
] * 600  # répété pour avoir assez d'exemples


def _download_fra_eng() -> Path | None:
    """Télécharge et extrait fra-eng.zip si nécessaire. Retourne None si échec."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = DATA_DIR / "fra.txt"
    if txt_path.exists():
        return txt_path

    zip_path = DATA_DIR / "fra-eng.zip"
    print(f"Telechargement du corpus fra-eng depuis {FRA_ENG_URL} ...")
    try:
        req = Request(FRA_ENG_URL, headers={"User-Agent": "Mozilla/5.0 (EMSI-DeepLearning-Project)"})
        with urlopen(req, timeout=60) as resp:
            zip_path.write_bytes(resp.read())
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        if txt_path.exists():
            return txt_path
    except Exception as exc:
        print(f"Telechargement impossible ({exc}) -> corpus de secours integre.")
    return None


def load_parallel_pairs(max_train: int = 8000, max_val: int = 800) -> tuple[list, list]:
    """
    Retourne (train_pairs, val_pairs) où chaque élément est (phrase_en, phrase_fr).
    Format ManyThings : ligne 'anglais\\tfrançais' (ignorer les lignes #).
    """
    txt_path = _download_fra_eng()
    pairs: list[tuple[str, str]] = []
    if txt_path is not None:
        with txt_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    pairs.append((parts[0].strip(), parts[1].strip()))
    else:
        pairs = list(FALLBACK_PAIRS)
        print(f"Utilisation du corpus integre : {len(pairs)} paires EN-FR")

    n_val = min(max_val, max(1, len(pairs) // 10))
    n_train = min(max_train, len(pairs) - n_val)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train : n_train + n_val]
    print(f"Corpus fra-eng : {len(train_pairs)} train, {len(val_pairs)} val")
    return train_pairs, val_pairs
