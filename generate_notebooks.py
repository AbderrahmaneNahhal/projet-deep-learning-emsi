"""Génère les 3 notebooks du projet Deep Learning EMSI."""
import json
from pathlib import Path

ROOT = Path(__file__).parent


def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


def save(name, cells):
    path = ROOT / name
    path.write_text(json.dumps(nb(cells), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Créé : {path}")


LIVRABLES_MD = """## Correspondance avec les livrables (§9 du sujet)

| Livrable | Fichier / dossier |
|----------|-------------------|
| 1. Rapport scientifique | `rapport/rapport_scientifique.md` → PDF |
| 2. Code source commenté | `src/partie*.py` |
| 3. Notebook exécutable | ce fichier + `main.py` |
| 4. Annexe expérimentale | `annexe/` (généré par `py main.py`) |

Voir `LIVRABLES.md` à la racine du projet."""


# =============================================================================
# PARTIE I – MLP
# =============================================================================
partie1 = [
    md(LIVRABLES_MD),
    md("""# Partie I – MLP et ingénierie PyTorch
**Projet Deep Learning — EMSI Casablanca 2025–2026**

**Dataset :** Breast Cancer Wisconsin (`sklearn.datasets`, intégré à scikit-learn)  
**Tâche :** classification binaire (tumeur maligne / bénigne)

Ce notebook couvre : théorie, préparation des données, deux implémentations MLP, initialisations, sauvegarde/chargement, GPU et métriques."""),
    md("## 0. Dépendances"),
    code("%pip install -q torch scikit-learn matplotlib seaborn"),
    md("""## 1. Théorie – `nn.Module`, gradients et device

- **`nn.Module`** : abstraction PyTorch ; `forward()` définit la propagation avant.
- **Paramètres** : tenseurs apprenables (`weight`, `bias`) enregistrés automatiquement.
- **`state_dict()`** : dictionnaire nom → tenseur pour sauvegarder les poids.
- **Rétropropagation** : `loss.backward()` calcule les gradients ; l'optimiseur met à jour θ.
- **Device** : modèle et données doivent être sur le **même** device (CPU ou CUDA)."""),
    code("""import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()
print("Device :", device)
torch.manual_seed(42)
np.random.seed(42)"""),
    md("## 2. Préparation des données"),
    code("""data = load_breast_cancer()
X = data.data.astype(np.float32)
y = data.target.astype(np.int64)
feature_names = data.feature_names
target_names = data.target_names

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

n_features = X_train.shape[1]
n_classes = len(np.unique(y))
print(f"Échantillons : train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
print(f"Features : {n_features}, classes : {list(target_names)}")"""),
    code("""def make_loaders(X_tr, y_tr, X_va, y_va, batch_size=32):
    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    val_ds = TensorDataset(torch.tensor(X_va), torch.tensor(y_va))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
    )

train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val)"""),
    md("## 3. MLP avec `nn.Sequential`"),
    code("""mlp_seq = nn.Sequential(
    nn.Linear(n_features, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, n_classes),
).to(device)

print("Architecture Sequential :")
print(mlp_seq)
for name, p in mlp_seq.named_parameters():
    print(f"  {name:30s} {tuple(p.shape)}")"""),
    md("## 4. MLP avec classe personnalisée"),
    code("""class MLPCustom(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64), n_out=2, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.out = nn.Linear(hidden[1], n_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.out(x)

mlp_custom = MLPCustom(n_features, n_out=n_classes).to(device)
print(mlp_custom)
print("\\nstate_dict (extrait) :", list(mlp_custom.state_dict().keys())[:4], "...")"""),
    md("## 5. Initialisation – gaussienne, constante, Xavier"),
    code("""def init_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.zeros_(m.bias)

def init_constant(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 1.0)
        nn.init.zeros_(m.bias)

def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

INIT_FNS = {
    "gaussienne": init_normal,
    "constante": init_constant,
    "xavier": init_xavier,
}"""),
    md("## 6. Boucle d'entraînement et évaluation"),
    code("""def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


def predict_all(model, X):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32, device=device)
        return model(x).argmax(1).cpu().numpy()


def train_model(model, init_name="xavier", epochs=80, lr=1e-3):
    model = copy.deepcopy(model).to(device)
    model.apply(INIT_FNS[init_name])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_state, best_acc = None, 0.0
    for ep in range(1, epochs + 1):
        tr_loss, _ = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history, best_acc"""),
    code("""results_init = {}
histories = {}
for name in INIT_FNS:
    model, hist, val_acc = train_model(mlp_custom, init_name=name, epochs=60)
    results_init[name] = val_acc
    histories[name] = hist
    print(f"Init {name:12s} → meilleure val accuracy : {val_acc:.4f}")

plt.figure(figsize=(8, 4))
for name, h in histories.items():
    plt.plot(h["val_acc"], label=name)
plt.xlabel("Époque")
plt.ylabel("Accuracy validation")
plt.title("Influence de l'initialisation")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""),
    md("## 7. Sauvegarde et rechargement du meilleur modèle"),
    code("""best_model, best_hist, _ = train_model(mlp_custom, init_name="xavier", epochs=80)
save_path = Path("checkpoints")
save_path.mkdir(exist_ok=True)
ckpt_file = save_path / "mlp_breast_cancer.pth"

torch.save(
    {
        "state_dict": best_model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "n_features": n_features,
        "n_classes": n_classes,
    },
    ckpt_file,
)
print("Modèle sauvegardé :", ckpt_file)

# Rechargement
loaded = MLPCustom(n_features, n_out=n_classes).to(device)
checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
loaded.load_state_dict(checkpoint["state_dict"])
loaded.eval()

y_pred = predict_all(loaded, X_test)
print("Test accuracy (modèle rechargé) :", accuracy_score(y_test, y_pred))"""),
    md("## 8. Comparaison Sequential vs Custom + métriques"),
    code("""mlp_seq_trained, _, _ = train_model(mlp_seq, init_name="xavier", epochs=80)

for label, model in [("Sequential", mlp_seq_trained), ("Custom", best_model)]:
    y_pred = predict_all(model, X_test)
    print(f"\\n=== {label} ===")
    print(classification_report(y_test, y_pred, target_names=target_names))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Matrice de confusion – {label}")
    plt.ylabel("Vrai")
    plt.xlabel("Prédit")
    plt.show()

    print(
        "F1 :", f1_score(y_test, y_pred, average="weighted"),
        "| Precision :", precision_score(y_test, y_pred, average="weighted"),
        "| Recall :", recall_score(y_test, y_pred, average="weighted"),
    )"""),
    md("""## 9. Question de synthèse – Partie I

**Question :** Dans quelle mesure un MLP bien paramétré constitue-t-il une solution pertinente pour la classification tabulaire sur un dataset réel, et quelles sont ses principales limites ?

**Éléments à développer dans votre rapport** (à personnaliser avec vos chiffres) :

1. **Pertinence** : sur Breast Cancer, les features sont déjà structurées et de faible dimension (~30). Un MLP capture des interactions non linéaires entre biomarqueurs sans ingénierie de noyaux manuelle.
2. **Méthodologie** : normalisation indispensable ; split stratifié ; initialisation Xavier souvent plus stable que gaussienne/constante (symétrie des neurones).
3. **Résultats** : comparer accuracy/F1 entre initialisations et entre Sequential/Custom (performances proches si même capacité).
4. **Limites** : pas d'inductif bias pour la structure spatiale ou temporelle ; risque de surapprentissage si réseau trop large ; interprétabilité limitée vs modèles linéaires ou arbres ; besoin de beaucoup de données pour des tables haute dimension.

> Rédigez 1–2 pages en style académique en vous appuyant sur les courbes et tableaux ci-dessus."""),
    md("## 10. Export vers l'annexe expérimentale"),
    code("""# Exécute depuis la racine du projet : py main.py --partie 1
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src import partie1_mlp
metrics = partie1_mlp.run(export=True, epochs=60)
print("→ Figures et tableaux dans annexe/partie1/")"""),
]

# =============================================================================
# PARTIE II – CNN
# =============================================================================
partie2 = [
    md(LIVRABLES_MD),
    md("""# Partie II – CNN et vision par ordinateur
**Projet Deep Learning — EMSI Casablanca 2025–2026**

**Dataset :** CIFAR-10 (`torchvision.datasets`, téléchargement automatique)  
**Tâche :** classification 10 classes, images 32×32 RGB

Contenu : théorie CNN, implémentations manuelles, LeNet, expériences architecturales, visualisation des feature maps, comparaison MLP vs CNN."""),
    md("## 0. Dépendances"),
    code("%pip install -q torch torchvision matplotlib scikit-learn"),
    md("""## 1. Théorie – Pourquoi un CNN plutôt qu'un MLP ?

- **Localité** : un filtre convolutif ne voit qu'un voisinage → pertinent pour les textures et contours.
- **Partage des poids** : même filtre sur toute l'image → moins de paramètres, invariance translationnelle approximative.
- **Hiérarchie** : couches basses = contours ; couches hautes = concepts.

**Taille de sortie convolution** (sans dilation) :
$$H_{out} = \\left\\lfloor \\frac{H_{in} + 2P - K}{S} \\right\\rfloor + 1$$

**Pooling** : réduction spatiale (max ou moyenne)."""),
    code("""import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)
torch.manual_seed(42)

DATA_DIR = "./data_cifar"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_full = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
classes = train_full.classes
print("Classes CIFAR-10 :", classes)"""),
    code("""# Sous-ensemble pour expériences rapides (modifier SUBSET_SIZE=None pour tout le dataset)
SUBSET_SIZE = 8000
if SUBSET_SIZE:
    idx = torch.randperm(len(train_full))[:SUBSET_SIZE]
    train_set = Subset(train_full, idx.tolist())
else:
    train_set = train_full

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
len(train_set), len(test_set)"""),
    md("## 2. Calculs manuels – corrélation croisée et pooling"),
    code("""def cross_corr2d(X, K, padding=0, stride=1):
    \"\"\"Corrélation croisée 2D (canal unique).\"\"\"
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


def avg_pool2d(X, pool_size=2, stride=2):
    X = np.asarray(X, dtype=np.float32)
    out_h = (X.shape[0] - pool_size) // stride + 1
    out_w = (X.shape[1] - pool_size) // stride + 1
    Y = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            patch = X[i * stride : i * stride + pool_size, j * stride : j * stride + pool_size]
            Y[i, j] = patch.mean()
    return Y

# Exemple pédagogique
X_demo = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
K_demo = np.array([[0, 1], [2, 3]], dtype=np.float32)
manual = cross_corr2d(X_demo, K_demo, padding=0, stride=1)
print("Entrée 4x4 :\\n", X_demo)
print("Noyau :\\n", K_demo)
print("Sortie manuelle :\\n", manual)"""),
    code("""# Comparaison avec PyTorch
x_t = torch.tensor(X_demo).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
k_t = torch.tensor(K_demo).unsqueeze(0).unsqueeze(0)
pytorch_out = F.conv2d(x_t, k_t, padding=0, stride=1).squeeze().numpy()
print("PyTorch conv2d :\\n", pytorch_out)
print("Écart max :", np.abs(manual - pytorch_out).max())

x_pool = torch.tensor(X_demo).unsqueeze(0).unsqueeze(0)
print("MaxPool manuel :\\n", max_pool2d(X_demo))
print("MaxPool PyTorch :\\n", F.max_pool2d(x_pool, 2, 2).squeeze().numpy())"""),
    md("## 3. CNN type LeNet pour CIFAR-10"),
    code("""class LeNetCIFAR(nn.Module):
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
        x = self.features(x)
        return self.classifier(x)


class MLPImage(nn.Module):
    \"\"\"MLP aplati sur CIFAR-10 (baseline peu adapté).\"\"\"
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
        return self.net(x)"""),
    md("## 4. Entraînement et comparaison MLP vs CNN"),
    code("""def train_classifier(model, epochs=12, lr=1e-3, name="model"):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
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
        tr_loss = loss_sum / total
        acc = tc / tt
        hist["loss"].append(tr_loss)
        hist["acc"].append(acc)
        print(f"[{name}] Epoch {ep:02d} | loss={tr_loss:.4f} | test_acc={acc:.4f}")
    return model, hist


cnn_model, hist_cnn = train_classifier(LeNetCIFAR(), epochs=12, name="CNN-LeNet")
mlp_model, hist_mlp = train_classifier(MLPImage(), epochs=12, name="MLP")"""),
    code("""plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(hist_cnn["loss"], label="CNN")
plt.plot(hist_mlp["loss"], label="MLP")
plt.title("Loss entraînement")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(hist_cnn["acc"], label="CNN")
plt.plot(hist_mlp["acc"], label="MLP")
plt.title("Accuracy test")
plt.legend()
plt.tight_layout()
plt.show()"""),
    md("## 5. Expériences – padding, stride, pooling, filtres, conv 1×1"),
    code("""def eval_config(**kwargs):
    model, hist = train_classifier(LeNetCIFAR(**kwargs), epochs=8, name=str(kwargs))
    return hist["acc"][-1]

experiments = {
    "baseline": {},
    "plus_de_filtres": {"channels": (16, 32)},
    "conv_1x1": {"channels": (6, 16), "use_1x1": True},
}
scores = {k: eval_config(**v) for k, v in experiments.items()}
for k, v in scores.items():
    print(f"{k:20s} → test accuracy = {v:.4f}")"""),
    code("""# Effet du stride / padding sur une conv isolée
x = torch.randn(1, 3, 32, 32)
for desc, kw in [
    ("padding=0, stride=1", dict(kernel_size=5, padding=0, stride=1)),
    ("padding=2, stride=1", dict(kernel_size=5, padding=2, stride=1)),
    ("stride=2", dict(kernel_size=5, padding=0, stride=2)),
]:
    y = F.conv2d(x, torch.randn(16, 3, 5, 5), padding=kw["padding"], stride=kw["stride"])
    print(desc, "-> sortie", tuple(y.shape))"""),
    md("## 6. Visualisation des feature maps"),
    code("""cnn_model.eval()
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)
with torch.no_grad():
    feat = cnn_model.features(images[:1])

# Première couche : 6 filtres
fm = feat[0].cpu()
n_show = min(6, fm.shape[0])
fig, axes = plt.subplots(1, n_show, figsize=(12, 2))
for i in range(n_show):
    axes[i].imshow(fm[i].numpy(), cmap="viridis")
    axes[i].set_title(f"Filtre {i}")
    axes[i].axis("off")
plt.suptitle("Feature maps – 1ère couche convolutive")
plt.show()

# Image originale
img = images[0].cpu().permute(1, 2, 0).numpy()
mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2470, 0.2435, 0.2616])
img = np.clip(img * std + mean, 0, 1)
plt.imshow(img)
plt.title(f"Image test – vraie classe : {classes[labels[0]]}")
plt.axis("off")
plt.show()"""),
    md("""## 7. Question de synthèse – Partie II

**Question :** Pourquoi un CNN est-il plus pertinent qu'un MLP pour la classification d'images sur un dataset réel ?

**Pistes de réponse** (à rédiger dans le rapport avec vos courbes) :

- Le MLP traite 3072 pixels indépendamment → explosion paramétrique, pas d'exploitation de la structure 2D.
- Le CNN exploite localité et partage des poids → meilleure généralisation sur CIFAR-10.
- Padding préserve les bords ; stride réduit la résolution ; pooling apporte invariance locale ; conv 1×1 mélange canaux sans changer la résolution spatiale.
- Interpréter les feature maps : filtres bas niveau ≈ contours / textures."""),
    md("## 8. Export vers l'annexe expérimentale"),
    code("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src import partie2_cnn
metrics = partie2_cnn.run(export=True, epochs=10)
print("→ annexe/partie2/")"""),
]

# =============================================================================
# PARTIE III – RNN / Seq2Seq
# =============================================================================
partie3 = [
    md(LIVRABLES_MD),
    md("""# Partie III – RNN, LSTM, GRU et Seq2Seq
**Projet Deep Learning — EMSI Casablanca 2025–2026**

**Dataset :** corpus parallèle **fra-eng** (ManyThings, téléchargement auto) — équivalent fra-eng du sujet EMSI  
**Tâches :** modélisation de séquences, comparaison RNN/LSTM/GRU, traduction Seq2Seq, décodage glouton vs beam search, métrique BLEU."""),
    md("## 0. Dépendances"),
    code("%pip install -q torch matplotlib sacrebleu"),
    md("""## 1. Théorie – modèle de langage et perplexité

**Factorisation :** $P(x_1,\\ldots,x_T) = \\prod_t P(x_t \\mid x_{<t})$

**Perplexité :** $\\mathrm{PPL} = \\exp\\big(-\\frac{1}{T}\\sum_t \\log P(x_t \\mid x_{<t})\\big)$ — plus bas = mieux.

**BPTT** : rétropropagation à travers le temps ; risque de gradients explosifs → **gradient clipping**."""),
    code("""import math
import random
import time
from collections import Counter

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from torchtext.data.utils import get_tokenizer
from src.corpus import load_parallel_pairs
from src.vocab_utils import build_vocab, SimpleVocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = SPECIALS
MAX_LEN = 25
BATCH_SIZE = 64
EMBED_DIM = 128
HIDDEN_DIM = 256"""),
    md("## 2. Chargement et préparation du corpus fra-eng"),
    code("""tokenizer = get_tokenizer("basic_english")

MAX_TRAIN = 8000
MAX_VAL = 800
train_pairs, val_pairs = load_parallel_pairs(max_train=MAX_TRAIN, max_val=MAX_VAL)
print("Exemple :", train_pairs[0])"""),
    code("""src_vocab = build_vocab(tokenizer, train_pairs, lang_idx=0, specials=SPECIALS)
tgt_vocab = build_vocab(tokenizer, train_pairs, lang_idx=1, specials=SPECIALS)

def encode(tokens, vocab, add_bos_eos=False):
    ids = [vocab[t] for t in tokens]
    if add_bos_eos:
        return [vocab["<bos>"]] + ids + [vocab["<eos>"]]
    return ids

def decode(ids, vocab):
    itos = vocab.get_itos()
    words = []
    for i in ids:
        w = itos[i]
        if w == EOS:
            break
        if w not in (PAD, BOS):
            words.append(w)
    return " ".join(words)"""),
    code("""class TranslationDataset(Dataset):
    def __init__(self, pairs, max_len=MAX_LEN):
        self.samples = []
        for src, tgt in pairs:
            src_ids = encode(tokenizer(src), src_vocab)[: max_len - 2]
            tgt_ids = encode(tokenizer(tgt), tgt_vocab)[: max_len - 2]
            self.samples.append(
                (
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_batch(batch):
    src_list, tgt_list = zip(*batch)
    src_pad = pad_sequence(src_list, batch_first=True, padding_value=src_vocab["<pad>"])
    tgt_in = [torch.tensor([tgt_vocab["<bos>"]] + t.tolist(), dtype=torch.long) for t in tgt_list]
    tgt_out = [torch.tensor(t.tolist() + [tgt_vocab["<eos>"]], dtype=torch.long) for t in tgt_list]
    tgt_in_pad = pad_sequence(tgt_in, batch_first=True, padding_value=tgt_vocab["<pad>"])
    tgt_out_pad = pad_sequence(tgt_out, batch_first=True, padding_value=tgt_vocab["<pad>"])
    return src_pad, tgt_in_pad, tgt_out_pad

train_ds = TranslationDataset(train_pairs)
val_ds = TranslationDataset(val_pairs)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_batch)
len(src_vocab), len(tgt_vocab)"""),
    md("## 3. Comparaison RNN / LSTM / GRU (encodeur seul – perplexité)"),
    code("""class RecurrentLM(nn.Module):
    def __init__(self, vocab_size, cell="gru", hidden=HIDDEN_DIM, embed=EMBED_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed, padding_idx=src_vocab["<pad>"])
        if cell == "rnn":
            self.rnn = nn.RNN(embed, hidden, batch_first=True)
        elif cell == "lstm":
            self.rnn = nn.LSTM(embed, hidden, batch_first=True)
        else:
            self.rnn = nn.GRU(embed, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)
        self.cell = cell

    def forward(self, x):
        emb = self.embed(x)
        if self.cell == "lstm":
            out, _ = self.rnn(emb)
        else:
            out, _ = self.rnn(emb)
        return self.fc(out)


def masked_nll(logits, targets, pad_idx):
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_idx,
        reduction="sum",
    )
    mask = targets != pad_idx
    n_tokens = mask.sum().item()
    return loss / max(n_tokens, 1), n_tokens


import torch.nn.functional as F

def train_lm(cell, epochs=5):
    model = RecurrentLM(len(src_vocab), cell=cell).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for ep in range(1, epochs + 1):
        model.train()
        total_loss, n_tok = 0.0, 0
        for src, _, _ in train_dl:
            src = src.to(device)
            inp, tgt = src[:, :-1], src[:, 1:]
            logits = model(inp)
            loss, nt = masked_nll(logits, tgt, src_vocab["<pad>"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * nt
            n_tok += nt
        ppl = math.exp(total_loss / n_tok)
        losses.append(ppl)
        print(f"{cell.upper()} epoch {ep} | perplexité train ≈ {ppl:.2f}")
    return model, losses"""),
    code("""lm_results = {}
for cell in ["rnn", "lstm", "gru"]:
    _, losses = train_lm(cell, epochs=4)
    lm_results[cell] = losses

plt.figure(figsize=(7, 4))
for cell, losses in lm_results.items():
    plt.plot(losses, marker="o", label=cell.upper())
plt.ylabel("Perplexité")
plt.xlabel("Époque")
plt.title("RNN vs LSTM vs GRU – langage source EN")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""),
    md("## 4. Seq2Seq encodeur–décodeur (GRU) + teacher forcing"),
    code("""class Encoder(nn.Module):
    def __init__(self, vocab_size, embed=EMBED_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed, padding_idx=src_vocab["<pad>"])
        self.rnn = nn.GRU(embed, hidden, batch_first=True)

    def forward(self, src):
        emb = self.embed(src)
        outputs, hidden = self.rnn(emb)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed=EMBED_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed, padding_idx=tgt_vocab["<pad>"])
        self.rnn = nn.GRU(embed, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, tgt_in, hidden):
        emb = self.embed(tgt_in)
        out, hidden = self.rnn(emb, hidden)
        return self.fc(out), hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt_in):
        _, hidden = self.encoder(src)
        logits, _ = self.decoder(tgt_in, hidden)
        return logits"""),
    code("""def train_seq2seq(epochs=8):
    model = Seq2Seq(Encoder(len(src_vocab)), Decoder(len(tgt_vocab))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses = []
    for ep in range(1, epochs + 1):
        model.train()
        total, n_tok = 0.0, 0
        t0 = time.time()
        for src, tgt_in, tgt_out in train_dl:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            logits = model(src, tgt_in)
            loss, nt = masked_nll(logits, tgt_out, tgt_vocab["<pad>"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # BPTT + clipping
            opt.step()
            total += loss.item() * nt
            n_tok += nt
        train_losses.append(total / n_tok)
        print(f"Seq2Seq epoch {ep} | loss/token={train_losses[-1]:.4f} | {time.time()-t0:.1f}s")
    return model, train_losses

seq2seq_model, s2s_losses = train_seq2seq(epochs=8)
plt.plot(s2s_losses)
plt.title("Perte Seq2Seq (token moyen)")
plt.xlabel("Époque")
plt.show()"""),
    md("## 5. Décodage glouton vs beam search"),
    code("""def translate_sentence(model, sentence, max_len=MAX_LEN, method="greedy", beam_width=3):
    model.eval()
    tokens = tokenizer(sentence)
    src = torch.tensor([encode(tokens, src_vocab)], device=device)
    with torch.no_grad():
        enc_out, hidden = model.encoder(src)

    generated = [tgt_vocab["<bos>"]]

    if method == "greedy":
        dec_in = torch.tensor([[tgt_vocab["<bos>"]]], device=device)
        for _ in range(max_len):
            logits, hidden = model.decoder(dec_in, hidden)
            next_id = logits[:, -1, :].argmax(-1).item()
            generated.append(next_id)
            if next_id == tgt_vocab["<eos>"]:
                break
            dec_in = torch.tensor([[next_id]], device=device)
        return decode(generated, tgt_vocab)

    # Beam search simplifié
    beams = [([tgt_vocab["<bos>"]], hidden, 0.0)]
    completed = []
    for _ in range(max_len):
        new_beams = []
        for seq, h, score in beams:
            if seq[-1] == tgt_vocab["<eos>"]:
                completed.append((seq, score))
                continue
            dec_in = torch.tensor([[seq[-1]]], device=device)
            logits, new_h = model.decoder(dec_in, h)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
            topk = log_probs.topk(beam_width)
            for i in range(topk.values.size(0)):
                nid = topk.indices[i].item()
                new_beams.append((seq + [nid], new_h, score + topk.values[i].item()))
        beams = sorted(new_beams, key=lambda x: x[2] / len(x[0]), reverse=True)[:beam_width]
        if all(s[-1] == tgt_vocab["<eos>"] for s, _, _ in beams):
            break
    best = beams[0][0] if beams else generated
    return decode(best, tgt_vocab)"""),
    code("""examples = [
    "a group of people are sitting on a bench",
    "two young boys playing with a ball",
    "the man is wearing a red shirt",
]
for sent in examples:
    g = translate_sentence(seq2seq_model, sent, method="greedy")
    b = translate_sentence(seq2seq_model, sent, method="beam", beam_width=5)
    print("EN :", sent)
    print("Greedy DE :", g)
    print("Beam DE   :", b)
    print("-" * 60)"""),
    md("## 6. Évaluation BLEU (validation)"),
    code("""try:
    import sacrebleu
except ImportError:
    sacrebleu = None

def compute_bleu(model, pairs, n=200, method="greedy"):
    refs, hyps = [], []
    for i, (src, tgt) in enumerate(pairs[:n]):
        pred = translate_sentence(model, src, method=method, beam_width=5)
        ref = " ".join(tokenizer(tgt))
        hyps.append(pred)
        refs.append([ref])
    if sacrebleu:
        return sacrebleu.corpus_bleu(hyps, refs).score
    # fallback simpliste : exact match ratio
    exact = sum(h == r[0] for h, r in zip(hyps, refs)) / len(refs)
    return exact

bleu_greedy = compute_bleu(seq2seq_model, val_pairs, n=150, method="greedy")
bleu_beam = compute_bleu(seq2seq_model, val_pairs, n=150, method="beam")
print(f"BLEU (greedy, n=150) : {bleu_greedy:.2f}")
print(f"BLEU (beam, n=150)   : {bleu_beam:.2f}")"""),
    md("""## 7. Question de synthèse – Partie III

**Question :** Dans quelle mesure les architectures récurrentes modélisent-elles efficacement une séquence réelle ? Justifier le passage RNN → LSTM/GRU → Seq2Seq.

**Éléments pour votre rapport :**

1. **Modèle de langage** : factorisation par la règle de chaîne ; perplexité pour comparer RNN/LSTM/GRU.
2. **Stabilité** : LSTM/GRU gèrent mieux les dépendances longues ; clipping indispensable sur RNN simple.
3. **Seq2Seq** : encodeur résume la source ; décodeur génère la cible mot à mot ; teacher forcing stabilise l'entraînement.
4. **Décodage** : glouton rapide mais sous-optimal ; beam search explore plusieurs hypothèses → BLEU souvent meilleur.
5. **Limites** : compression du contexte dans un seul vecteur ; pas d'attention → performances plafonnent vs Transformers.

---

## 8. Question transversale (rapport final)

Relier MLP (géométrie tabulaire), CNN (localité spatiale) et RNN/Seq2Seq (dépendances temporelles) : le paradigme supervisé reste identique (perte + gradient), mais l'**inductive bias** architectural doit suivre la structure des données."""),
    md("## 9. Export vers l'annexe expérimentale"),
    code("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src import partie3_seq2seq
metrics = partie3_seq2seq.run(export=True, lm_epochs=3, s2s_epochs=5)
print("→ annexe/partie3/")"""),
]

save("partie1_MLP_tabulaire.ipynb", partie1)
save("partie2_CNN_images.ipynb", partie2)
save("partie3_RNN_Seq2Seq.ipynb", partie3)
