"""
One-hot CNN baseline (Koo-style "does pretraining help at all" control): a small supervised
convolutional net trained from scratch on one-hot sequence windows, under the same
leave-one-chromosome-out folds as every other method. If a from-scratch one-hot CNN matches or
beats a pretrained gLM, pretraining is not what carries the zero-shot signal.

Window: +/-100 bp around the variant, reference sequence with the alt substituted at the
centre, one-hot 4xL. Small CNN (2 conv layers + global max-pool + linear). CPU/MPS.

Output: results/onehot_cnn.csv (dataset, auroc, auprc)
Usage: ~/Downloads/venv/bin/python code/17_onehot_cnn.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pysam
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_utils as A  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REF = os.path.join(ROOT, "data", "reference", "hg38.fa")
W = 100
L = 2 * W + 1
torch.manual_seed(0)
np.random.seed(0)
MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def onehot_windows(df, fa):
    X = np.zeros((len(df), 4, L), dtype=np.float32)
    ok = np.ones(len(df), bool)
    ch = df["chrom"].astype(str).values
    pos = df["pos"].astype(int).values
    alt = df["alt"].astype(str).values
    for i in range(len(df)):
        c = ch[i] if ch[i].startswith("chr") else "chr" + ch[i]
        try:
            seq = fa.fetch(c, pos[i] - 1 - W, pos[i] + W).upper()
        except Exception:
            ok[i] = False; continue
        if len(seq) != L:
            ok[i] = False; continue
        seq = seq[:W] + alt[i] + seq[W + 1:]
        for j, b in enumerate(seq):
            k = MAP.get(b)
            if k is not None:
                X[i, k, j] = 1.0
    return X, ok


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 32, 8, padding=4), nn.ReLU(),
            nn.Conv1d(32, 32, 8, padding=4), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(),
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_predict(Xtr, ytr, Xte, epochs=12, bs=128):
    dev = "cpu"
    m = CNN().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1)))
    Xtr = torch.tensor(Xtr); ytr_t = torch.tensor(ytr.astype(np.float32))
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        m.train()
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            opt.zero_grad()
            loss = lossf(m(Xtr[idx]), ytr_t[idx])
            loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(Xte))).numpy()


def run(name, path, labkind):
    fa = pysam.FastaFile(REF)
    d = pd.read_parquet(path)
    y = ((d["label"] == "Pathogenic") if labkind == "path" else d["label"]).astype(int).values
    ch = d["chrom"].astype(str).str.replace("chr", "", regex=False).values
    # subsample large sets for CPU tractability (keeps class balance)
    if len(d) > 45000:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(d), 45000, replace=False)
        d = d.iloc[keep].reset_index(drop=True); y = y[keep]
        ch = d["chrom"].astype(str).str.replace("chr", "", regex=False).values
    X, ok = onehot_windows(d, fa)
    d, y, ch, X = d[ok], y[ok], ch[ok], X[ok]
    # 5-fold grouped-by-chromosome CV (chrom assigned to a fold) — fewer trains than full LOCO
    uch = pd.unique(ch)
    fold_of = {c: i % 5 for i, c in enumerate(uch)}
    folds = np.array([fold_of[c] for c in ch])
    oof = np.full(len(d), np.nan)
    for f in range(5):
        te = folds == f; tr = ~te
        if y[tr].sum() < 5 or y[tr].sum() == tr.sum():
            continue
        oof[te] = train_predict(X[tr], y[tr], X[te], epochs=6)
    m = ~np.isnan(oof)
    au = roc_auc_score(y[m], oof[m]); ap = average_precision_score(y[m], oof[m])
    print(f"[{name}] one-hot CNN LOCO AUROC={au:.3f} AUPRC={ap:.3f} (n={m.sum()})", flush=True)
    return dict(dataset=name, auroc=au, auprc=ap, n=int(m.sum()))


def main():
    jobs = [("TraitGym-mendelian", "data/traitgym/mendelian_scored.parquet", "bool"),
            ("TraitGym-complex", "data/traitgym/complex_scored.parquet", "bool"),
            ("ClinVar", "data/clinvar/clinvar_scored.parquet", "path")]
    out = []
    for name, path, lab in jobs:
        try:
            out.append(run(name, os.path.join(ROOT, path), lab))
        except Exception as e:
            print(f"[{name}] failed: {e}", flush=True)
    pd.DataFrame(out).to_csv(os.path.join(ROOT, "results", "onehot_cnn.csv"), index=False)
    print("saved results/onehot_cnn.csv")


if __name__ == "__main__":
    main()
