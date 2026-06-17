"""
Partie II — CNN sur CIFAR-10 (torchvision).

Implémentations manuelles conv/pooling, LeNet, comparaison MLP vs CNN, feature maps.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from .experiment_io import append_global_summary, save_figure, save_metrics, save_table

SEED = 42
DATA_DIR = "data_cifar"
SUBSET_SIZE = 8000  # None pour dataset complet


def cross_corr2d(X, K, padding=0, stride=1):
    """Corrélation croisée 2D manuelle (1 canal)."""
    X = np.asarray(X, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    if padding > 0:
        X = np.pad(X, padding, mode="constant")
    h_k, w_k = K.shape
    out_h = (X.shape[0] - h_k) // stride + 1
    out_w = (X.shape[1] - w_k) // stride + 1
    Y = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            patch = X[i * stride : i * stride + h_k, j * stride : j * stride + w_k]
            Y[i, j] = (patch * K).sum()
    return Y


def max_pool2d(X, pool_size=2, stride=2):
    X = np.asarray(X, dtype=np.float32)
    out_h = (X.shape[0] - pool_size) // stride + 1
    out_w = (X.shape[1] - pool_size) // stride + 1
    Y = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            patch = X[i * stride : i * stride + pool_size, j * stride : j * stride + pool_size]
            Y[i, j] = patch.max()
    return Y


class LeNetCIFAR(nn.Module):
    """CNN inspiré de LeNet adapté à CIFAR-10 (32×32, 3 canaux)."""

    def __init__(self, n_classes=10, channels=(6, 16), use_1x1=False):
        super().__init__()
        c1, c2 = channels
        layers = [
            nn.Conv2d(3, c1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(c1, c2, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        ]
        if use_1x1:
            layers += [nn.Conv2d(c2, c2, kernel_size=1), nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 6 * 6, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MLPImage(nn.Module):
    """Baseline MLP sur images aplaties (peu adapté aux images)."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_classifier(model, train_loader, test_loader, device, epochs=10, name="model"):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    hist = {"loss": [], "acc": []}
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
        model.eval()
        tc, tt = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                tc += (model(xb).argmax(1) == yb).sum().item()
                tt += yb.size(0)
        hist["loss"].append(loss_sum / total)
        hist["acc"].append(tc / tt)
        print(f"  [{name}] epoch {ep}/{epochs} test_acc={hist['acc'][-1]:.4f}")
    return model, hist


def run(export: bool = True, epochs: int = 10) -> dict:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_full = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    if SUBSET_SIZE:
        idx = torch.randperm(len(train_full))[:SUBSET_SIZE]
        train_set = Subset(train_full, idx.tolist())
    else:
        train_set = train_full

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    # Vérification manuel vs PyTorch
    X_demo = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    K_demo = np.array([[0, 1], [2, 3]], dtype=np.float32)
    manual = cross_corr2d(X_demo, K_demo)
    x_t = torch.tensor(X_demo).unsqueeze(0).unsqueeze(0)
    k_t = torch.tensor(K_demo).unsqueeze(0).unsqueeze(0)
    pytorch_out = F.conv2d(x_t, k_t).squeeze().numpy()
    ecart_conv = float(np.abs(manual - pytorch_out).max())

    print("Entraînement CNN LeNet...")
    cnn_model, hist_cnn = train_classifier(
        LeNetCIFAR(), train_loader, test_loader, device, epochs=epochs, name="CNN"
    )
    print("Entraînement MLP baseline...")
    mlp_model, hist_mlp = train_classifier(
        MLPImage(), train_loader, test_loader, device, epochs=epochs, name="MLP"
    )

    # Expériences architecturales (epochs réduits)
    exp_scores = {}
    for label, kwargs in [
        ("baseline", {}),
        ("plus_filtres", {"channels": (16, 32)}),
        ("conv_1x1", {"use_1x1": True}),
    ]:
        _, h = train_classifier(
            LeNetCIFAR(**kwargs),
            train_loader,
            test_loader,
            device,
            epochs=max(5, epochs // 2),
            name=label,
        )
        exp_scores[label] = h["acc"][-1]

    out = {
        "device": str(device),
        "ecart_max_conv_manuel_pytorch": ecart_conv,
        "test_acc_cnn": hist_cnn["acc"][-1],
        "test_acc_mlp": hist_mlp["acc"][-1],
        "experiences_architecturales": exp_scores,
    }

    if export:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(hist_cnn["loss"], label="CNN")
        axes[0].plot(hist_mlp["loss"], label="MLP")
        axes[0].set_title("Loss entraînement")
        axes[0].legend()
        axes[1].plot(hist_cnn["acc"], label="CNN")
        axes[1].plot(hist_mlp["acc"], label="MLP")
        axes[1].set_title("Accuracy test")
        axes[1].legend()
        save_figure(fig, "partie2", "courbe_mlp_vs_cnn.png")
        save_table("partie2", exp_scores, "experiences_architecturales.csv")

        # Feature maps
        cnn_model.eval()
        images, _ = next(iter(test_loader))
        images = images.to(device)
        with torch.no_grad():
            feat = cnn_model.features(images[:1])
        fm = feat[0].cpu()
        n_show = min(6, fm.shape[0])
        fig, axes = plt.subplots(1, n_show, figsize=(12, 2))
        for i in range(n_show):
            axes[i].imshow(fm[i].numpy(), cmap="viridis")
            axes[i].set_title(f"F{i}")
            axes[i].axis("off")
        fig.suptitle("Feature maps — couche conv 1")
        save_figure(fig, "partie2", "feature_maps.png")

        save_metrics("partie2", out)
        append_global_summary({"partie": "II_CNN", **out})

    return out
