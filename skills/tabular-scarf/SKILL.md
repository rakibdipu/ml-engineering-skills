---
name: tabular-scarf
description: >-
  Self-supervised contrastive representation learning for tabular/CSV data using
  SCARF (Self-Supervised Contrastive Learning using Random Feature Corruption)
  with the `pytorch-scarf` library. Use this skill when training models on
  unlabeled or semi-supervised tabular datasets, extracting rich embeddings from
  tabular data where labels are scarce, improving downstream classification or
  regression performance with XGBoost/LightGBM/MLPs, or handling mixed
  numerical-categorical tabular features.
---

# Tabular SCARF — Self-Supervised Contrastive Learning for Tables

> **Source:** [clabrugere/pytorch-scarf](https://github.com/clabrugere/pytorch-scarf)  
> **Paper:** [SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption (Bahri et al., ICLR 2022)](https://arxiv.org/abs/2106.15147)

**SCARF** adapts SimCLR-style contrastive learning to tabular data. Instead of image augmentations (crops, flips), SCARF builds contrastive views by **randomly corrupting a random subset of features** using values sampled from each feature's empirical marginal distribution. The encoder is forced to produce representations invariant to this corruption — learning the true structure of the data.

---

## 1. Installation

```bash
pip install pytorch-scarf torch scikit-learn pandas numpy
```

> **Important:** The correct import path is `from pytorch_scarf.model import SCARF` (underscore, not hyphen). The PyPI package installs as `pytorch_scarf`.

### Prerequisite: Preprocessing

SCARF only operates on **numerical features**. Categorical columns must be encoded before use:

```python
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

df = pd.read_csv("data.csv")

num_cols  = ["age", "income", "tenure"]
cat_cols  = ["country", "plan_type"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(),   num_cols),
    ("cat", OrdinalEncoder(),   cat_cols),
])

X_train_proc = preprocessor.fit_transform(df)
```

---

## 2. Core Architecture

```
Unlabeled Tabular Row  x  (n_features)
          │
          ├──► Original     x ───────────► Encoder f(·) ──► Projection Head g(·) ──► z
          │                                                                             │
          └──► Corrupted    x̃ (SCARF aug) ► Encoder f(·) ──► Projection Head g(·) ──► z̃
                                                                                        │
                                                               NT-Xent Loss: maximize sim(z, z̃)
                                                               (same row = positive pair)
```

**Two-phase training strategy:**
1. **Pre-training (Unsupervised):** Train encoder + projection head on all data (labeled + unlabeled) using NT-Xent loss.
2. **Downstream:** Discard projection head. Use frozen encoder as feature extractor → feed into XGBoost/LightGBM/LogReg.

---

## 3. Pre-Training the SCARF Encoder

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_scarf.model import SCARF
from pytorch_scarf.loss import NTXent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare dataset — X_train_proc is already StandardScaler-normalized NumPy array
X_tensor = torch.tensor(X_train_proc, dtype=torch.float32)
dataset   = TensorDataset(X_tensor)
loader    = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

input_dim = X_train_proc.shape[1]

# Instantiate SCARF
model = SCARF(
    input_dim=input_dim,
    emb_dim=128,            # Output embedding size (tune: 64–256)
    encoder_depth=4,        # Encoder MLP depth (tune: 3–6)
    head_depth=2,           # Projection head depth
    corruption_rate=0.6,    # Fraction of features corrupted per sample (tune: 0.3–0.7)
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = NTXent(temperature=0.07)   # Lower temp = harder negatives, better separation

# Training loop
model.train()
for epoch in range(100):
    epoch_loss = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Model internally generates the corrupted view from the marginal distribution
        z_anchor, z_corrupted = model(batch)
        loss = criterion(z_anchor, z_corrupted)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        avg = epoch_loss / len(loader)
        print(f"Epoch [{epoch+1:3d}/100]  NT-Xent Loss: {avg:.4f}")

# Save pre-trained encoder
torch.save(model.state_dict(), "scarf_pretrained.pt")
print("Pre-training complete. Encoder saved.")
```

---

## 4. Downstream: Extract Embeddings → Train Classifier

```python
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Load pre-trained weights
model.load_state_dict(torch.load("scarf_pretrained.pt", map_location=device))
model.eval()

def get_embeddings(X_np: np.ndarray) -> np.ndarray:
    """Extract encoder representations (NOT projection head) for downstream use."""
    with torch.no_grad():
        t = torch.tensor(X_np, dtype=torch.float32).to(device)
        # encoder(x) gives the raw representation before the projection head
        emb = model.encoder(t).cpu().numpy()
    return emb

# Get embeddings for train and test sets
X_train_emb = get_embeddings(X_train_proc)
X_test_emb  = get_embeddings(X_test_proc)

# Train a downstream classifier on the SCARF embeddings
clf = XGBClassifier(n_estimators=300, learning_rate=0.05, use_label_encoder=False)
clf.fit(X_train_emb, y_train)

y_pred = clf.predict(X_test_emb)
print(classification_report(y_test, y_pred))
```

---

## 5. Fine-Tuning with a Supervised Head (Option B)

When some labeled data is available, attach a head and fine-tune end-to-end:

```python
import torch.nn as nn

class SCARFWithHead(nn.Module):
    def __init__(self, scarf_model, emb_dim, n_classes):
        super().__init__()
        self.encoder  = scarf_model.encoder  # Reuse pretrained encoder
        self.head     = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.head(self.encoder(x))

# Build supervised model from pretrained encoder
supervised_model = SCARFWithHead(model, emb_dim=128, n_classes=2).to(device)

# Fine-tune with a small learning rate to preserve pretrained representations
optimizer_ft = torch.optim.AdamW(supervised_model.parameters(), lr=5e-5)
criterion_ft = nn.CrossEntropyLoss()

# Standard PyTorch supervised training loop here...
```

---

## 6. Hyperparameter Recommendations

| Parameter | Recommended | Notes |
| :--- | :--- | :--- |
| `corruption_rate` | `0.5`–`0.6` | 0.6 is the paper's default. Too low → trivial task; too high → destroys structure |
| `temperature` | `0.07`–`0.1` | Lower = harder negatives and sharper representations |
| `emb_dim` | `64`–`256` | Match to dataset size; larger datasets benefit from higher dims |
| `encoder_depth` | `3`–`6` | Shallower for small datasets (<10k rows); deeper for large ones |
| Batch size | ≥256 | NT-Xent needs diverse in-batch negatives — use as large a batch as VRAM allows |
| `learning_rate` | `1e-3` (pretrain), `5e-5` (finetune) | Use AdamW with weight decay `1e-4` |
| Epochs | 50–200 | Monitor NT-Xent loss plateau; early stop after no improvement for 20 epochs |

### Common Pitfalls
1. **Categorical features not encoded:** SCARF expects all-numeric input. Encode categoricals with `OrdinalEncoder` or `TargetEncoder` first.
2. **Using the projection head for downstream tasks:** Always extract from `model.encoder(x)`, not the full `model(x)` output.
3. **Small batch sizes:** NT-Xent loss quality degrades badly below batch size 64 due to lack of in-batch negatives.
4. **Not normalizing features:** Always apply `StandardScaler` or `QuantileTransformer` before SCARF — the marginal corruption distribution is computed per-feature.
