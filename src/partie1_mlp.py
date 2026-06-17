"""
Partie I — MLP sur données tabulaires (Breast Cancer Wisconsin, sklearn).

Couvre : Sequential vs classe custom, initialisations, state_dict, GPU, métriques.
Exporte courbes et tableaux vers annexe/partie1/.
"""
from __future__ import annotations

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .experiment_io import CHECKPOINTS, append_global_summary, save_figure, save_metrics, save_table

SEED = 42


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPCustom(nn.Module):
    """MLP avec classe nn.Module personnalisée (flexible pour forward non trivial)."""

    def __init__(self, in_dim: int, hidden=(128, 64), n_out: int = 2, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.out = nn.Linear(hidden[1], n_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.out(x)


def build_sequential(in_dim: int, n_out: int) -> nn.Sequential:
    """Équivalent fonctionnel via nn.Sequential."""
    return nn.Sequential(
        nn.Linear(in_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_out),
    )


def _init_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.zeros_(m.bias)


def _init_constant(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 1.0)
        nn.init.zeros_(m.bias)


def _init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


INIT_MAP = {"gaussienne": _init_normal, "constante": _init_constant, "xavier": _init_xavier}


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_sum, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return loss_sum / total, correct / total


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    init_name="xavier",
    epochs=60,
    label="",
):
    model = copy.deepcopy(model).to(device)
    model.apply(INIT_MAP[init_name])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_state, best_acc = None, 0.0
    prefix = f"  [{label}] " if label else "  "
    for ep in range(1, epochs + 1):
        tr_loss, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())
        if ep == 1 or ep == epochs or ep % max(1, epochs // 5) == 0:
            print(f"{prefix}epoch {ep}/{epochs} | val_acc={va_acc:.4f} | val_loss={va_loss:.4f}")
    model.load_state_dict(best_state)
    print(f"{prefix}meilleure val_acc={best_acc:.4f} (init={init_name})")
    return model, history, best_acc


def run(export: bool = True, epochs: int = 60) -> dict:
    """Pipeline complet Partie I. Retourne un dict de métriques pour le rapport."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()

    data = load_breast_cancer()
    X, y = data.data.astype(np.float32), data.target.astype(np.int64)
    target_names = list(data.target_names)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    n_features, n_classes = X_train.shape[1], len(np.unique(y))
    print(f"  Dataset Breast Cancer : {len(X)} echantillons, {n_features} features, device={device}")
    print(f"  Split : train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=32,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=32
    )

    # Comparaison des initialisations
    print("  Comparaison des initialisations (gaussienne, constante, xavier)...")
    init_results = {}
    histories = {}
    for name in INIT_MAP:
        model, hist, acc = train_model(
            MLPCustom(n_features, n_out=n_classes),
            train_loader,
            val_loader,
            device,
            init_name=name,
            epochs=epochs,
            label=f"init-{name}",
        )
        init_results[name] = acc
        histories[name] = hist

    if export:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, h in histories.items():
            ax.plot(h["val_acc"], label=name)
        ax.set_xlabel("Époque")
        ax.set_ylabel("Accuracy validation")
        ax.set_title("Partie I — Influence de l'initialisation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_figure(fig, "partie1", "courbe_initialisations.png")
        save_table("partie1", init_results, "comparaison_initialisations.csv")

    # Meilleur modèle custom + sauvegarde state_dict
    print("  Entrainement MLP (classe personnalisee, init xavier)...")
    best_custom, _, _ = train_model(
        MLPCustom(n_features, n_out=n_classes),
        train_loader,
        val_loader,
        device,
        init_name="xavier",
        epochs=epochs,
        label="MLP-Custom",
    )
    ckpt = CHECKPOINTS / "mlp_breast_cancer.pth"
    torch.save(
        {
            "state_dict": best_custom.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "n_features": n_features,
            "n_classes": n_classes,
        },
        ckpt,
    )

    # Sequential vs Custom
    print("  Entrainement MLP (nn.Sequential, init xavier)...")
    best_seq, _, _ = train_model(
        build_sequential(n_features, n_classes),
        train_loader,
        val_loader,
        device,
        init_name="xavier",
        epochs=epochs,
        label="MLP-Sequential",
    )

    def predict(model, X_arr):
        model.eval()
        with torch.no_grad():
            x = torch.tensor(X_arr, dtype=torch.float32, device=device)
            return model(x).argmax(1).cpu().numpy()

    metrics_final = {}
    for label, model in [("Sequential", best_seq), ("Custom", best_custom)]:
        y_pred = predict(model, X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))
        metrics_final[label] = {
            "accuracy": acc,
            "precision": float(precision_score(y_test, y_pred, average="weighted")),
            "recall": float(recall_score(y_test, y_pred, average="weighted")),
            "f1": f1,
        }
        print(f"  Test {label} : accuracy={acc:.4f}, F1={f1:.4f}")
        if export:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                xticklabels=target_names,
                yticklabels=target_names,
                ax=ax,
            )
            ax.set_title(f"Matrice de confusion — {label}")
            ax.set_ylabel("Vrai")
            ax.set_xlabel("Prédit")
            save_figure(fig, "partie1", f"confusion_matrix_{label.lower()}.png")
            report = classification_report(y_test, y_pred, target_names=target_names)
            (Path(__file__).parent.parent / "annexe" / "partie1" / "tableaux").mkdir(
                parents=True, exist_ok=True
            )
            (
                Path(__file__).parent.parent
                / "annexe"
                / "partie1"
                / "tableaux"
                / f"classification_report_{label.lower()}.txt"
            ).write_text(report, encoding="utf-8")

    out = {
        "device": str(device),
        "n_train": len(X_train),
        "initialisations_val_acc": init_results,
        "test_metrics": metrics_final,
        "checkpoint": str(ckpt),
    }
    if export:
        save_metrics("partie1", out)
        append_global_summary({"partie": "I_MLP", **out})
    return out
