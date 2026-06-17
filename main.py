#!/usr/bin/env python
"""
Script principal exécutable — Projet Deep Learning EMSI (livrable 3).

Lance les trois parties du projet, exporte métriques/figures vers annexe/
et affiche un résumé console. Les notebooks restent la version interactive détaillée.

Usage:
    py main.py
    py main.py --partie 1
    py main.py --epochs-mlp 40 --epochs-cnn 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Permettre l'import de src/ depuis la racine du projet
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import partie1_mlp, partie2_cnn, partie3_seq2seq
from src.experiment_io import ANNEXE_ROOT, RAPPORT_DIR


def parse_args():
    p = argparse.ArgumentParser(description="Projet Deep Learning EMSI — exécution globale")
    p.add_argument("--partie", type=int, choices=[1, 2, 3], default=None, help="Exécuter une seule partie")
    p.add_argument("--epochs-mlp", type=int, default=50, help="Époques Partie I")
    p.add_argument("--epochs-cnn", type=int, default=8, help="Époques Partie II")
    p.add_argument("--lm-epochs", type=int, default=3, help="Époques LM Partie III")
    p.add_argument("--s2s-epochs", type=int, default=5, help="Époques Seq2Seq Partie III")
    p.add_argument("--no-export", action="store_true", help="Ne pas écrire dans annexe/")
    return p.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = parse_args()
    export = not args.no_export
    results = {}

    print("=" * 60)
    print("PROJET DEEP LEARNING EMSI - execution des experiences")
    print(f"Export annexe : {export} -> {ANNEXE_ROOT}")
    print("=" * 60)

    if args.partie in (None, 1):
        print("\n>>> PARTIE I - MLP (Breast Cancer, sklearn)")
        results["partie1"] = partie1_mlp.run(export=export, epochs=args.epochs_mlp)

    if args.partie in (None, 2):
        print("\n>>> PARTIE II - CNN (CIFAR-10, torchvision)")
        results["partie2"] = partie2_cnn.run(export=export, epochs=args.epochs_cnn)

    if args.partie in (None, 3):
        print("\n>>> PARTIE III - RNN / Seq2Seq (corpus fra-eng ManyThings)")
        results["partie3"] = partie3_seq2seq.run(
            export=export, lm_epochs=args.lm_epochs, s2s_epochs=args.s2s_epochs
        )

    # Génère un fichier de résultats pour compléter le rapport
    resultats_path = RAPPORT_DIR / "resultats_experimentaux.json"
    resultats_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("TERMINE")
    print(f"  - Resultats JSON : {resultats_path}")
    print(f"  - Annexe (figures/tableaux) : {ANNEXE_ROOT}")
    print(f"  - Rapport a completer : {RAPPORT_DIR / 'rapport_scientifique.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
