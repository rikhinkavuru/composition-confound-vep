"""
E5 (post-hoc pentanucleotide+composition calibration -- constructive core) +
E6 (surviving-signal characterization).

calibrate_llr(): model-agnostic post-hoc transform. For each variant with 5-mer mutation
context ctx = (pentamer_ref, alt), define the neutral expected score
    LLR_neutral(ctx) = mean model score over CONTROL variants (label==0) sharing ctx,
estimated LEAVE-ONE-CHROMOSOME-OUT (context means from training chroms only -> applied to the
held-out chrom), so calibration never sees test-fold labels. Calibrated score =
    raw - LLR_neutral(ctx).
Unseen contexts on a test fold fall back to the transition/transversion x CpG-stratified
global control mean. This generalizes GPN-Star's pentanucleotide neutral-score calibration
into a benchmark-wide, cross-model, post-hoc evaluation transform.

E5 metrics per (dataset, model): raw vs calibrated |partial rho|(score, mu_5mer);
raw vs calibrated per-chrom AUROC + composition recovery ratio; Kendall-tau reorder of the
model leaderboard pre/post calibration.

E6: on TraitGym (carries SpliceAI + motif + RemapTF annotations), among positives, compare
variants the calibrated gLM ranks well but the composition baseline does not ("gLM-wins")
against composition-wins: enrichment (odds ratios) for splice signal, TF-motif/footprint
overlap, and composition-neutral status.

Outputs: results/e5_calibration.csv, results/neutral_5mer_<model>.csv, results/e6_surviving.csv
Usage: ~/Downloads/venv/bin/python code/12_e5_e6.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pysam
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_utils as A  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
REF = os.path.join(ROOT, "data", "reference", "hg38.fa")
os.makedirs(RES, exist_ok=True)


def add_context5(df):
    """Add `context5` = 5-mer reference pentamer (centered on variant) + '>' + alt, using
    indexed hg38. chrom normalized to 'chrN'. Rows where the pentamer center != ref get NaN."""
    fa = pysam.FastaFile(REF)
    contigs = set(fa.references)
    ctx = np.empty(len(df), dtype=object)
    chrom = df["chrom"].astype(str).values
    pos = df["pos"].astype(int).values
    ref = df["ref"].astype(str).values
    alt = df["alt"].astype(str).values
    for i in range(len(df)):
        c = chrom[i]
        if not c.startswith("chr"):
            c = "chr" + c
        if c not in contigs:
            ctx[i] = None; continue
        try:
            pent = fa.fetch(c, pos[i] - 3, pos[i] + 2).upper()  # 0-based half-open, 5bp
        except Exception:
            ctx[i] = None; continue
        if len(pent) != 5 or pent[2] != ref[i]:
            ctx[i] = None; continue
        ctx[i] = f"{pent}>{alt[i]}"
    df = df.copy()
    df["context5"] = ctx
    return df


def calibrate_llr(df, score_col, label_col="label_bin", chrom_col="chrom_norm",
                  ctx_col="context5"):
    """LOCO cross-fit pentanucleotide neutral-score calibration. Returns calibrated Series."""
    s = df[score_col].astype("float64").values
    y = df[label_col].astype(int).values
    ctx = df[ctx_col].values
    chroms = df[chrom_col].values
    cal = np.full(len(df), np.nan)
    is_ts = df["is_transition"].astype(int).values if "is_transition" in df.columns else np.zeros(len(df), int)
    cpg = df["cpg_ref"].astype(int).values if "cpg_ref" in df.columns else np.zeros(len(df), int)
    for test_chrom in pd.unique(chroms):
        te = chroms == test_chrom
        tr = (~te) & (y == 0) & ~np.isnan(s)  # controls only, train chroms
        if tr.sum() < 50:
            tr = (~te) & ~np.isnan(s)  # fallback: all train variants
        # per-context neutral mean
        dtr = pd.DataFrame({"ctx": ctx[tr], "s": s[tr], "ts": is_ts[tr], "cpg": cpg[tr]})
        ctx_mean = dtr.groupby("ctx")["s"].mean()
        strat_mean = dtr.groupby(["ts", "cpg"])["s"].mean()
        glob = dtr["s"].mean()
        te_idx = np.where(te)[0]
        for j in te_idx:
            if np.isnan(s[j]):
                continue
            base = ctx_mean.get(ctx[j], np.nan)
            if np.isnan(base):
                base = strat_mean.get((is_ts[j], cpg[j]), glob)
            cal[j] = s[j] - base
    return pd.Series(cal, index=df.index)


def load_binary_with_ctx():
    ds = []
    specs = [
        ("ClinVar", "data/clinvar/clinvar_scored.parquet", "label_pathogenic", "ccre_class",
         False, {"gpn_msa_full": "GPN-MSA", "evo2_7b_llr": "Evo2-7B", "evo2_40b_llr": "Evo2-40B",
                 "cadd_csv": "CADD", "phylop_100way": "phyloP"}),
        ("TraitGym-mendelian", "data/traitgym/mendelian_scored.parquet", "label_bool", "consequence",
         True, {"gpn_msa_llr": "GPN-MSA", "evo2_7b_llr": "Evo2-7B", "evo2_40b_llr": "Evo2-40B",
                "nt_llr": "NT-2.5B", "caduceus_llr": "Caduceus", "cadd_rawscore": "CADD",
                "phylop_100v": "phyloP"}),
        ("TraitGym-complex", "data/traitgym/complex_scored.parquet", "label_bool", "consequence",
         True, {"gpn_msa_llr": "GPN-MSA", "evo2_7b_llr": "Evo2-7B", "evo2_40b_llr": "Evo2-40B",
                "nt_llr": "NT-2.5B", "caduceus_llr": "Caduceus", "cadd_rawscore": "CADD",
                "phylop_100v": "phyloP"}),
    ]
    for name, path, labkind, cons, matched, models in specs:
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            continue
        cache = p.replace(".parquet", "_ctx.parquet")
        if os.path.exists(cache):
            d = pd.read_parquet(cache)
        else:
            d = pd.read_parquet(p)
            d = add_context5(d)
            d.to_parquet(cache, index=False)
        d["label_bin"] = ((d["label"] == "Pathogenic").astype(int) if labkind == "label_pathogenic"
                          else d["label"].astype(int))
        d["chrom_norm"] = d["chrom"].astype(str).str.replace("chr", "", regex=False)
        models = {c: n for c, n in models.items() if c in d.columns and d[c].notna().sum() > 100}
        ds.append(dict(name=name, df=d, models=models, cons=cons, matched=matched))
    return ds


def run_e5(ds):
    rows = []
    for D in ds:
        d = D["df"]
        comp = A.composition_cv(d, "label_bin", "chrom_norm", return_oof=True)
        raw_auc, cal_auc = {}, {}
        for col, pretty in D["models"].items():
            cal = calibrate_llr(d, col).values
            # E1-style confound corr vs mu_5mer (|score|), raw vs calibrated
            r_raw, _, _ = A.partial_spearman(np.abs(d[col].values), d["mu_5mer"].values)
            r_cal, _, _ = A.partial_spearman(np.abs(cal), d["mu_5mer"].values)
            # AUROC raw vs calibrated (per-chrom)
            draw = d.assign(_s=d[col].values)
            a_raw = A.per_chrom_auc(draw, "_s", "label_bin", "chrom_norm")["auroc"]
            dcal = d.assign(_s=cal)
            a_cal = A.per_chrom_auc(dcal, "_s", "label_bin", "chrom_norm")["auroc"]
            rr_raw = A.recovery_ratio(comp["auroc"], a_raw)
            rr_cal = A.recovery_ratio(comp["auroc"], a_cal)
            raw_auc[pretty] = a_raw; cal_auc[pretty] = a_cal
            rows.append(dict(dataset=D["name"], model=pretty, matched=D["matched"],
                             corr_mu_raw=r_raw, corr_mu_cal=r_cal,
                             auroc_raw=a_raw, auroc_cal=a_cal, auroc_delta=a_cal - a_raw,
                             comp_auroc=comp["auroc"], recovery_raw=rr_raw, recovery_cal=rr_cal))
        common = list(raw_auc.keys())
        if len(common) >= 3:
            tau, _ = kendalltau([raw_auc[k] for k in common], [cal_auc[k] for k in common])
            print(f"[{D['name']}] leaderboard Kendall tau raw vs calibrated = {tau:.3f} "
                  f"(comp AUROC {comp['auroc']:.3f})")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "e5_calibration.csv"), index=False)
    print(f"E5 -> results/e5_calibration.csv ({len(out)} rows)")
    print(out[["dataset", "model", "corr_mu_raw", "corr_mu_cal", "auroc_raw", "auroc_cal",
               "recovery_raw", "recovery_cal"]].round(3).to_string(index=False))
    return out


def run_e6(ds):
    rows = []
    for D in ds:
        if not D["name"].startswith("TraitGym"):
            continue
        d = D["df"]
        if "cadd_spliceai_acc_gain" not in d.columns:
            continue
        pos = d[d["label_bin"] == 1].copy()
        if len(pos) < 30:
            continue
        # calibrated GPN-MSA vs composition oof, restricted to positives: who ranks them high?
        comp = A.composition_cv(d, "label_bin", "chrom_norm", return_oof=True)
        gcol = "gpn_msa_llr" if "gpn_msa_llr" in d.columns else list(D["models"])[0]
        gcal = calibrate_llr(d, gcol)
        g_o, _ = A.orient(gcal.values, d["label_bin"].values)
        c_o = comp["oof"]
        # percentile rank within all variants
        gr = pd.Series(g_o).rank(pct=True).values
        cr = pd.Series(c_o).rank(pct=True).values
        posmask = (d["label_bin"] == 1).values
        gwin = posmask & (gr > 0.7) & (cr < 0.5)     # gLM ranks high, composition misses
        cwin = posmask & (cr > 0.7) & (gr < 0.5)     # composition ranks high, gLM misses
        spliceai = d[["cadd_spliceai_acc_gain", "cadd_spliceai_acc_loss",
                      "cadd_spliceai_don_gain", "cadd_spliceai_don_loss"]].max(axis=1).values
        motif = (d["cadd_motifecount"].fillna(0).values > 0) | (d["cadd_remapoverlaptf"].fillna(0).values > 0)
        comp_neutral = (np.abs(d["dGC_w51"].values) < np.nanmedian(np.abs(d["dGC_w51"].values)))
        for feat_name, feat in [("splice(>0.2)", spliceai > 0.2), ("TFmotif", motif),
                                ("comp_neutral", comp_neutral)]:
            a = (gwin & feat).sum(); b = (gwin & ~feat).sum()
            c = (cwin & feat).sum(); dd = (cwin & ~feat).sum()
            orr = ((a + 0.5) * (dd + 0.5)) / ((b + 0.5) * (c + 0.5))
            rows.append(dict(dataset=D["name"], feature=feat_name, n_gwin=int(gwin.sum()),
                             n_cwin=int(cwin.sum()), gwin_frac=a / max(gwin.sum(), 1),
                             cwin_frac=c / max(cwin.sum(), 1), odds_ratio=orr))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "e6_surviving.csv"), index=False)
    print(f"\nE6 -> results/e6_surviving.csv ({len(out)} rows)")
    if len(out):
        print(out.round(3).to_string(index=False))
    return out


def main():
    ds = load_binary_with_ctx()
    print("ctx coverage:", [(D["name"], int(D["df"]["context5"].notna().sum()), len(D["df"]))
                            for D in ds])
    run_e5(ds)
    run_e6(ds)


if __name__ == "__main__":
    main()
