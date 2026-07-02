"""
Shared statistics + evaluation utilities for the composition-confound VEP audit (E1-E6).

Design goals (mirrors outline section 9):
  - AUROC and AUPRC, oriented so higher score = more positive (pathogenic / causal / effect).
  - Per-chromosome cross-validation with sample-size-weighted averaging (TraitGym protocol);
    bootstrap standard errors.
  - Composition-only classifier (gradient-boosted trees on C1/C2 features) trained under the
    SAME CV so recovery ratios are apples-to-apples.
  - Partial Spearman (rank-residualize on covariates) for confound correlations that control
    for region/consequence.
  - DeLong test for paired AUROC differences.

Pure CPU; depends only on numpy/scipy/pandas/sklearn (all in the project venv).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------- orientation

# Canonical composition/mutation-expectedness feature set (C1 + C2), no sequence context.
COMP_FEATURES = [
    "mu_5mer", "cpg_ref", "cpg_alt", "is_transition", "flank_gc_w51",
    "dGC_w11", "dGC_w21", "dGC_w51",
    "dkmer1_w11", "dkmer2_w11", "dkmer3_w11",
    "dkmer1_w21", "dkmer2_w21", "dkmer3_w21",
    "dkmer1_w51", "dkmer2_w51", "dkmer3_w51",
]


def orient(score, y):
    """Return score flipped so that higher => positive label (AUROC >= 0.5).
    Returns (oriented_score, sign) where sign in {+1,-1}. NaNs preserved."""
    s = np.asarray(score, dtype="float64")
    y = np.asarray(y)
    m = ~np.isnan(s)
    if m.sum() < 3 or len(np.unique(y[m])) < 2:
        return s, 1
    a = roc_auc_score(y[m], s[m])
    sign = 1 if a >= 0.5 else -1
    return s * sign, sign


# ---------------------------------------------------------------- AUC + CIs

def auc_pair(score, y, orient_score=True):
    """AUROC and AUPRC on non-NaN entries, optionally auto-oriented."""
    s = np.asarray(score, dtype="float64")
    y = np.asarray(y).astype(int)
    m = ~np.isnan(s)
    s, y = s[m], y[m]
    if orient_score:
        s, _ = orient(s, y)
    return roc_auc_score(y, s), average_precision_score(y, s), int(m.sum())


def bootstrap_auc_ci(score, y, n_boot=1000, seed=0, orient_score=True):
    """Percentile bootstrap CI for AUROC and AUPRC. Returns dict."""
    s = np.asarray(score, dtype="float64"); y = np.asarray(y).astype(int)
    m = ~np.isnan(s); s, y = s[m], y[m]
    if orient_score:
        s, _ = orient(s, y)
    rng = np.random.default_rng(seed)
    n = len(y); idx = np.arange(n)
    aurocs = np.empty(n_boot); auprcs = np.empty(n_boot)
    for b in range(n_boot):
        bi = rng.choice(idx, n, replace=True)
        if len(np.unique(y[bi])) < 2:
            aurocs[b] = np.nan; auprcs[b] = np.nan; continue
        aurocs[b] = roc_auc_score(y[bi], s[bi])
        auprcs[b] = average_precision_score(y[bi], s[bi])
    return {
        "auroc": roc_auc_score(y, s), "auprc": average_precision_score(y, s),
        "auroc_lo": np.nanpercentile(aurocs, 2.5), "auroc_hi": np.nanpercentile(aurocs, 97.5),
        "auprc_lo": np.nanpercentile(auprcs, 2.5), "auprc_hi": np.nanpercentile(auprcs, 97.5),
        "n": int(m.sum()),
    }


def per_chrom_auc(df, score_col, label_col="label_bin", chrom_col="chrom", orient_score=True):
    """Per-chromosome AUROC/AUPRC, sample-size-weighted average across chroms (TraitGym).
    Orientation is fit GLOBALLY (once) to avoid per-fold sign leakage."""
    y = df[label_col].astype(int).values
    s = df[score_col].astype("float64").values
    if orient_score:
        s, _ = orient(s, y)
    rows = []
    for chrom, g in df.assign(_s=s, _y=y).groupby(chrom_col):
        gg = g[~np.isnan(g["_s"])]
        if len(gg) < 10 or gg["_y"].nunique() < 2:
            continue
        rows.append((chrom, len(gg),
                     roc_auc_score(gg["_y"], gg["_s"]),
                     average_precision_score(gg["_y"], gg["_s"])))
    if not rows:
        return dict(auroc=np.nan, auprc=np.nan, n=0, n_chrom=0)
    r = pd.DataFrame(rows, columns=["chrom", "n", "auroc", "auprc"])
    w = r["n"] / r["n"].sum()
    return dict(auroc=float((w * r["auroc"]).sum()),
                auprc=float((w * r["auprc"]).sum()),
                n=int(r["n"].sum()), n_chrom=len(r), per_chrom=r)


# ---------------------------------------------------------------- composition classifier

def composition_cv(df, label_col="label_bin", chrom_col="chrom", feat_cols=None,
                   seed=0, return_oof=False):
    """Leave-one-chromosome-out CV of a HistGBM composition-only classifier.
    Returns per-chrom-weighted AUROC/AUPRC and (optionally) out-of-fold predictions."""
    feat_cols = feat_cols or [c for c in COMP_FEATURES if c in df.columns]
    y = df[label_col].astype(int).values
    X = df[feat_cols].astype("float64").values
    chroms = df[chrom_col].values
    oof = np.full(len(df), np.nan)
    for chrom in pd.unique(chroms):
        te = chroms == chrom
        tr = ~te
        if y[tr].sum() < 5 or y[te].sum() < 1 or len(np.unique(y[tr])) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05,
                                             max_depth=None, random_state=seed)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    tmp = df[[chrom_col]].copy()
    tmp["label_bin"] = y
    tmp["_oof"] = oof
    res = per_chrom_auc(tmp, "_oof", "label_bin", chrom_col, orient_score=False)
    res["feat_cols"] = feat_cols
    if return_oof:
        res["oof"] = oof
    return res


def recovery_ratio(comp_auroc, model_auroc):
    """(comp - 0.5) / (model - 0.5). Undefined/inf-guarded when model ~ chance."""
    denom = model_auroc - 0.5
    if denom <= 1e-6:
        return np.nan
    return (comp_auroc - 0.5) / denom


# ---------------------------------------------------------------- partial Spearman

def partial_spearman(x, y, covars=None):
    """Spearman correlation of x,y after rank-residualizing both on covariate matrix covars.
    covars: 2D array-like or None. Rank-transform everything, OLS-residualize, correlate."""
    x = np.asarray(x, "float64"); y = np.asarray(y, "float64")
    m = ~(np.isnan(x) | np.isnan(y))
    if covars is not None:
        covars = np.asarray(covars, "float64")
        if covars.ndim == 1:
            covars = covars[:, None]
        m &= ~np.isnan(covars).any(axis=1)
    x, y = x[m], y[m]
    if m.sum() < 10:
        return np.nan, np.nan, int(m.sum())
    rx = stats.rankdata(x); ry = stats.rankdata(y)
    if covars is None:
        r, p = stats.spearmanr(x, y)
        return float(r), float(p), int(m.sum())
    C = covars[m]
    Cr = np.column_stack([stats.rankdata(C[:, j]) for j in range(C.shape[1])])
    Cr = np.column_stack([np.ones(len(Cr)), Cr])
    bx, *_ = np.linalg.lstsq(Cr, rx, rcond=None)
    by, *_ = np.linalg.lstsq(Cr, ry, rcond=None)
    ex = rx - Cr @ bx; ey = ry - Cr @ by
    r, p = stats.pearsonr(ex, ey)  # Pearson on rank-residuals = partial Spearman
    return float(r), float(p), int(m.sum())


# ---------------------------------------------------------------- DeLong paired AUROC test

def _compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N, dtype=float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float); T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]; neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _compute_midrank(pos[r]); ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(preds_sorted[r])
    aucs = (tz[:, :m].sum(axis=1) / m / n) - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(score_a, score_b, y):
    """Two-sided DeLong p-value for H0: AUROC(a)==AUROC(b) on the SAME samples.
    NaNs (in either score) dropped pairwise. Returns (auroc_a, auroc_b, p)."""
    sa = np.asarray(score_a, "float64"); sb = np.asarray(score_b, "float64")
    yy = np.asarray(y).astype(int)
    m = ~(np.isnan(sa) | np.isnan(sb))
    sa, sb, yy = sa[m], sb[m], yy[m]
    if yy.sum() < 1 or yy.sum() == len(yy):
        return np.nan, np.nan, np.nan
    m_pos = int(yy.sum())
    pos_idx = np.where(yy == 1)[0]; neg_idx = np.where(yy == 0)[0]
    preds = np.vstack([sa, sb])
    preds = np.concatenate([preds[:, pos_idx], preds[:, neg_idx]], axis=1)  # positives first
    aucs, cov = _fast_delong(preds, m_pos)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(p)


def benjamini_hochberg(pvals):
    """BH-FDR adjusted p-values."""
    p = np.asarray(pvals, "float64"); n = len(p)
    order = np.argsort(p); ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(ranked, 0, 1)
    return out
