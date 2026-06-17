"""
Utilitaires d'export pour l'annexe expérimentale (livrable 4).

Toutes les figures, tableaux CSV et métriques JSON sont écrits sous annexe/
de manière reproductible depuis les notebooks ou main.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Racine du projet = parent de src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNEXE_ROOT = PROJECT_ROOT / "annexe"
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
RAPPORT_DIR = PROJECT_ROOT / "rapport"


def ensure_dirs(*parts: str) -> Path:
    """Crée annexe/partieX/... et renvoie le chemin."""
    path = ANNEXE_ROOT.joinpath(*parts) if parts else ANNEXE_ROOT
    path.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    RAPPORT_DIR.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, partie: str, filename: str, dpi: int = 150) -> Path:
    """Enregistre une figure matplotlib dans annexe/partieX/figures/."""
    out_dir = ensure_dirs(partie, "figures")
    path = out_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def save_metrics(partie: str, metrics: dict[str, Any], filename: str = "metrics.json") -> Path:
    """Enregistre un dictionnaire de métriques en JSON."""
    out_dir = ensure_dirs(partie, "tableaux")
    path = out_dir / filename
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_table(partie: str, data: dict[str, Any] | pd.DataFrame, filename: str) -> Path:
    """Enregistre un tableau comparatif en CSV."""
    out_dir = ensure_dirs(partie, "tableaux")
    path = out_dir / filename
    if isinstance(data, pd.DataFrame):
        data.to_csv(path, index=True)
    else:
        pd.DataFrame([data]).to_csv(path, index=False)
    return path


def append_global_summary(entry: dict[str, Any]) -> Path:
    """Ajoute une entrée au résumé global annexe/summary_global.json."""
    ensure_dirs()
    summary_path = ANNEXE_ROOT / "summary_global.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        data = []
    data.append(entry)
    summary_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path
