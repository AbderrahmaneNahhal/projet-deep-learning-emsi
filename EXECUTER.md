# Comment exécuter le projet

## 1. Terminal (script principal)

Ouvre PowerShell dans le dossier du projet :

```powershell
cd c:\Users\HP\Desktop\deeplearningnew
py -m pip install torch torchvision scikit-learn matplotlib seaborn pandas numpy sacrebleu jupyter ipykernel
py main.py
```

### Options utiles

```powershell
py main.py --partie 1              # seulement MLP
py main.py --partie 2              # seulement CNN
py main.py --partie 3              # seulement Seq2Seq
py main.py --epochs-mlp 40 --epochs-cnn 6
```

Résultats : dossier `annexe\` + `rapport\resultats_experimentaux.json`

---

## 2. Notebooks (.ipynb)

### Dans VS Code

1. Ouvre `partie1_MLP_tabulaire.ipynb` (puis partie 2 et 3).
2. En haut à droite : choisis le noyau **Python (deeplearning EMSI)** ou **Python 3.14**.
3. Exécute cellule par cellule : **Shift + Entrée**.
4. Commence toujours par la cellule `%pip install` puis les imports.

### Dans le navigateur (Jupyter)

```powershell
cd c:\Users\HP\Desktop\deeplearningnew
py -m notebook
```

Clique sur chaque `.ipynb` dans la page qui s’ouvre.

---

## 3. Ordre recommandé

1. `partie1_MLP_tabulaire.ipynb`
2. `partie2_CNN_images.ipynb`
3. `partie3_RNN_Seq2Seq.ipynb`

Ou une seule fois : `py main.py` (équivalent aux 3 parties).
