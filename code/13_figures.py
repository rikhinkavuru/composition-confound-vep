"""
Publication figures for the composition-confound VEP audit. Four composite figures (fits an
8pp PMLR paper; extra panels -> appendix). Reads results/e{1..6}_*.csv.

Fig1  Headline: composition-only recovery ratio per model, grouped unmatched (ClinVar) vs
      matched (TraitGym) + absolute AUROC (composition vs models).
Fig2  Confound structure: |partial rho|(|LLR|, mu_5mer) per model x dataset (E1) +
      collapse curve model-minus-composition edge by |dGC| bin (E3a).
Fig3  Calibration (E5): mutation-rate coupling before/after + AUROC cost.
Fig4  Within-element clean test (E4 raw vs partial per element) + surviving-signal ORs (E6).

Outputs: results/figures/fig{1,2,3,4}.pdf (+ .png)
Usage: ~/Downloads/venv/bin/python code/13_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 220,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "axes.titleweight": "bold",
    "axes.linewidth": 0.8, "axes.titlepad": 8, "figure.facecolor": "white",
})
C = {"ClinVar": "#d1495b", "TraitGym-mendelian": "#2e86ab", "TraitGym-complex": "#61a5c2",
     "GLRB-eQTL": "#8d99ae"}
DSHORT = {"ClinVar": "ClinVar\n(unmatched)", "TraitGym-mendelian": "TraitGym\nMendelian",
          "TraitGym-complex": "TraitGym\ncomplex", "GLRB-eQTL": "GLRB\neQTL"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)
    print("  wrote", name)


def fig1():
    e2 = pd.read_csv(os.path.join(RES, "e2_recovery.csv"))
    e2 = e2[e2.model_auroc > 0.55]  # models with real signal to recover
    order = ["ClinVar", "TraitGym-mendelian", "TraitGym-complex"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    # left: recovery ratio bars per model, grouped by dataset
    models = ["GPN-MSA", "Evo2-7B", "Evo2-40B", "CADD", "phyloP-100v", "phyloP", "phyloP-100w"]
    dsl = [d for d in order if d in e2.dataset.unique()]
    x = np.arange(len(dsl)); w = 0.8
    for i, ds in enumerate(dsl):
        sub = e2[e2.dataset == ds].sort_values("recovery_ratio", ascending=False)
        sub = sub[(sub.recovery_ratio > 0) & (sub.recovery_ratio < 1.2)]
        vals = sub["recovery_ratio"].values
        xs = x[i] + np.linspace(-w/2, w/2, len(vals))
        ax1.bar(xs, vals, width=w/max(len(vals), 1)*0.9, color=C.get(ds, "#888"),
                edgecolor="white", linewidth=0.3)
        ax1.text(x[i], max(vals) + 0.02, f"n={len(vals)}", ha="center", fontsize=7, color="#555")
    ax1.axhline(0.5, ls="--", c="#333", lw=0.8)
    ax1.text(len(dsl)-0.5, 0.51, "50% recovery", fontsize=7, color="#333", ha="right")
    ax1.set_xticks(x); ax1.set_xticklabels([DSHORT[d] for d in dsl])
    ax1.set_ylabel("recovery ratio\n(comp-0.5)/(model-0.5)")
    ax1.set_title("(a) Composition-only recovers ~2× more\non unmatched than matched")
    ax1.set_ylim(0, 0.75)
    # right: absolute composition AUROC vs best model AUROC per dataset
    comp = e2.groupby("dataset")["comp_auroc"].first()
    bestm = e2.groupby("dataset")["model_auroc"].max()
    dd = [d for d in order if d in comp.index]
    xx = np.arange(len(dd))
    ax2.bar(xx - 0.2, [comp[d] for d in dd], 0.4, label="composition-only",
            color="#e0a458", edgecolor="white")
    ax2.bar(xx + 0.2, [bestm[d] for d in dd], 0.4, label="best model",
            color="#386641", edgecolor="white")
    ax2.axhline(0.5, ls=":", c="#999", lw=0.8)
    ax2.set_xticks(xx); ax2.set_xticklabels([DSHORT[d] for d in dd])
    ax2.set_ylabel("AUROC"); ax2.set_ylim(0.5, 1.0)
    ax2.set_title("(b) Absolute AUROC:\ncomposition vs best model")
    ax2.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "fig1_recovery")


def fig2():
    e1 = pd.read_csv(os.path.join(RES, "e1_confound.csv"))
    e1 = e1[e1.confounder == "mu_5mer"]
    e3 = pd.read_csv(os.path.join(RES, "e3_collapse.csv"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    # left: |partial rho absLLR| vs mu per model, grouped by dataset
    piv = e1.pivot_table(index="model", columns="dataset", values="partial_rho_absLLR")
    dss = [d for d in ["ClinVar", "TraitGym-mendelian", "TraitGym-complex"] if d in piv.columns]
    piv = piv.reindex(columns=dss)
    piv = piv.reindex(piv.abs().max(axis=1).sort_values(ascending=False).index)
    yy = np.arange(len(piv))
    for j, ds in enumerate(dss):
        ax1.barh(yy + (j-1)*0.25, piv[ds].abs().values, height=0.24,
                 color=C.get(ds), label=DSHORT[ds].replace("\n", " "))
    ax1.set_yticks(yy); ax1.set_yticklabels(piv.index)
    ax1.invert_yaxis()
    ax1.set_xlabel("|partial ρ|  (|LLR| vs 5-mer mutation rate)")
    ax1.set_title("(a) Mutation-rate coupling by model")
    ax1.legend(frameon=False, fontsize=7)
    # right: collapse curve — model-minus-comp edge by |dGC| bin, TraitGym-mendelian
    sub = e3[(e3.dataset == "TraitGym-mendelian") & (e3.binvar == "abs_dGC")].copy()
    sub["edge"] = sub.model_auroc - sub.comp_auroc
    zero_edge = ("NT-2.5B", "Caduceus", "HyenaDNA")  # smaller masked single-seq gLMs
    for model, g in sub.groupby("model"):
        g = g.sort_values("bin")
        ls = "--" if model in zero_edge else "-"
        ax2.plot(g["bin"], g["edge"], marker="o", ms=4, label=model, lw=1.4, ls=ls)
    ax2.axhline(0, c="#333", lw=0.8)
    ax2.set_xticks([0, 1, 2, 3]); ax2.set_xticklabels(["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"])
    ax2.set_xlabel("|ΔGC| quartile")
    ax2.set_ylabel("model AUROC − composition AUROC")
    ax2.set_title("(b) Smaller single-seq gLMs (dashed):\nzero edge over composition")
    ax2.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout()
    save(fig, "fig2_confound")


def fig3():
    e5 = pd.read_csv(os.path.join(RES, "e5_calibration.csv"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    # left: |corr mu| before vs after per model, ClinVar
    for ds, ax, ttl in [("ClinVar", ax1, "(a) ClinVar: mutation-rate coupling\nbefore→after calibration")]:
        sub = e5[e5.dataset == ds]
        yy = np.arange(len(sub))
        for k, (_, r) in enumerate(sub.iterrows()):
            ax.plot([abs(r.corr_mu_raw), abs(r.corr_mu_cal)], [k, k], c="#bbb", lw=1, zorder=1)
            ax.scatter(abs(r.corr_mu_raw), k, c="#d1495b", s=28, zorder=2,
                       label="raw" if k == 0 else "")
            ax.scatter(abs(r.corr_mu_cal), k, c="#2e86ab", s=28, zorder=2,
                       label="calibrated" if k == 0 else "")
        ax.set_yticks(yy); ax.set_yticklabels(sub.model.values)
        ax.invert_yaxis(); ax.set_xlabel("|ρ|  (|LLR| vs mutation rate)")
        ax.set_title(ttl); ax.legend(frameon=False)
    # right: AUROC raw vs calibrated (cost) per model across datasets
    ax2.plot([0.5, 1.0], [0.5, 1.0], ls=":", c="#999", lw=0.8)
    for ds in e5.dataset.unique():
        sub = e5[e5.dataset == ds]
        ax2.scatter(sub.auroc_raw, sub.auroc_cal, c=C.get(ds, "#888"), s=26,
                    label=DSHORT[ds].replace("\n", " "), edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("AUROC (raw)"); ax2.set_ylabel("AUROC (calibrated)")
    ax2.set_title("(b) Calibration AUROC cost is small\n(points near diagonal)")
    ax2.legend(frameon=False, fontsize=7)
    ax2.set_xlim(0.48, 1.0); ax2.set_ylim(0.48, 1.0)
    fig.tight_layout()
    save(fig, "fig3_calibration")


def fig4():
    e4 = pd.read_csv(os.path.join(RES, "e4_kircher.csv"))
    e6 = pd.read_csv(os.path.join(RES, "e6_surviving.csv"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    # left: Kircher raw vs partial per element
    e4 = e4.sort_values("rho_raw")
    yy = np.arange(len(e4))
    for k, (_, r) in enumerate(e4.iterrows()):
        ax1.plot([r.rho_raw, r.rho_partial], [k, k], c="#ccc", lw=0.8, zorder=1)
    ax1.scatter(e4.rho_raw, yy, c="#d1495b", s=16, label="raw ρ", zorder=2)
    ax1.scatter(e4.rho_partial, yy, c="#2e86ab", s=16, label="partial ρ", zorder=2)
    ax1.axvline(0, c="#333", lw=0.6)
    ax1.set_yticks(yy); ax1.set_yticklabels(e4.element.values, fontsize=6)
    ax1.set_xlabel("Spearman(GPN-MSA LLR, log2 effect)")
    ax1.set_title("(a) Kircher within-element:\npartial ≈ raw (no leakage)")
    ax1.legend(frameon=False, fontsize=7)
    # right: E6 odds ratios
    piv = e6.pivot_table(index="feature", columns="dataset", values="odds_ratio")
    feats = list(piv.index); xx = np.arange(len(feats))
    dss = list(piv.columns)
    for j, ds in enumerate(dss):
        ax2.bar(xx + (j - (len(dss)-1)/2)*0.35, piv[ds].values, 0.33,
                color=C.get(ds, "#888"), label=DSHORT.get(ds, ds).replace("\n", " "))
    ax2.axhline(1, ls="--", c="#333", lw=0.8)
    ax2.set_yscale("log")
    ax2.set_xticks(xx); ax2.set_xticklabels(feats, rotation=15, fontsize=7)
    ax2.set_ylabel("odds ratio (gLM-wins vs comp-wins)")
    ax2.set_title("(b) Surviving signal:\ngLM-wins enriched for TF motifs")
    ax2.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig4_within_surviving")


def main():
    print("building figures ->", FIG)
    fig1(); fig2(); fig3(); fig4()


if __name__ == "__main__":
    main()
