# examples/example.py

import pandas as pd
import numpy as np
import torch

from smartimpute import train_smartimpute, Generator, D1_WGAN_GP

# 1) Load & normalize

raw = pd.read_csv("data/sample_data.csv", index_col=0)
# per-cell (row) total‐count normalize to 10⁴, then log1p
counts = raw.values
normed = counts / counts.sum(axis=1, keepdims=True) * 1e4
log1p  = np.log1p(normed)

# 2) Subset to target genes (if present)

target_genes = ["GeneA", "GeneB", "GeneC", "GeneX"]
present      = [g for g in target_genes if g in raw.columns]
if not present:
    raise ValueError("None of your target_genes are in the data!")
log1p_df = pd.DataFrame(log1p, index=raw.index, columns=raw.columns)[present]

# 3) Build PyTorch tensors & masks

X_np = log1p_df.values.astype(np.float32)
# M = 1 if observed (>0), 0 if true zero
M_np = (X_np > 0).astype(np.float32)
# F_mask all zeros here (real data → no “forced zeros” known)
F_np = np.zeros_like(M_np, dtype=np.float32)

X_t = torch.tensor(X_np, dtype=torch.float32)
M_t = torch.tensor(M_np, dtype=torch.float32)
F_t = torch.tensor(F_np, dtype=torch.float32)

# 4) Train

# you can also import hyperparameters from your config
G_model, history, imputed = train_smartimpute(
    X_t, M_t, F_mask=F_t,
    recon_loss="mse",     # or "poisson"
    epochs=100,
    batch_size=64,
)


# 5) Save imputed DataFrame

imputed_df = pd.DataFrame(
    imputed, 
    index=log1p_df.index, 
    columns=log1p_df.columns
)
imputed_df.to_csv("examples/imputed_sample_data.csv")
print("Wrote examples/imputed_sample_data.csv")

