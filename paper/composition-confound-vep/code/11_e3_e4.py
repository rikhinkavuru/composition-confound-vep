"""
E3 (stratified + composition-matched evaluation) + E4 (satMutMPRA within-element clean test).

E3a  Collapse curve: bin variants by composition-change magnitude (|ΔGC_w51| and by
     mutation-rate quantile); within each bin compute model AUROC and composition-only AUROC.
     Shows whether the gLM edge concentrates where composition moves.
E3b  Composition-matched re-eval: on each binary benchmark, nearest-neighbor match positives
     to controls on standardized composition/mutation features (mu_5mer, ΔGC_w51, cpg_ref,
     is_transition, flank_gc_w51) WITHIN consequence/region class (on top of TraitGym's own
     matching). Re-evaluate every model on the matched subset; report matched AUROC and the
     Kendall-tau reordering of model rankings vs unmatched.
E4   Kircher satMutMPRA within-element: per element (fixed background), Spearman(GPN-MSA LLR,
     log2 effect) RAW vs PARTIAL controlling for substitution type (6 classes) + Δcomposition
     (ΔGC_w51) + CpG. Partial < raw ⇒ composition leakage; residual = functional signal.

Outputs: results/e3_collapse.csv, results/e3_matched.csv, results/e4_kircher.csv
Usage: ~/Downloads/venv/bin/python code/11_e3_e4.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_utils as A  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

MATCH_FEATS = ["mu_5mer", "dGC_w51", "cpg_ref", "is_transition", "flank_gc_w51"]


def load_binary():
    from importlib import import_module
    m = import_module("10_e1_e2") if False else None  # avoid import name issues
    # inline the same loader logic
    ds = []
    p = os.path.join(ROOT, "data", "clinvar", "clinvar_scored.parquet")
    if os.path.exists(p):
        d = pd.read_parquet(p)
        d["label_bin"] = (d["label"] == "Pathogenic").astype(int)
        d["chrom_norm"] = d["chrom"].astype(str).str.replace("chr", "", regex=False)
        models = {c: n for c, n in [("gpn_msa_full", "GPN-MSA"), ("evo2_7b_llr", "Evo2-7B"),
                  ("evo2_40b_llr", "Evo2-40B"), ("cadd_csv", "CADD"),
                  ("phylop_100way", "phyloP")] if c in d.columns and d[c].notna().sum() > 100}
        ds.append(dict(name="ClinVar", df=d, models=models, cons="ccre_class", matched=False))
    for split in ["mendelian", "complex"]:
        p = os.path.join(ROOT, "data", "traitgym", f"{split}_scored.parquet")
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        d["label_bin"] = d["label"].astype(int)
        d["chrom_norm"] = d["chrom"].astype(str).str.replace("chr", "", regex=False)
        models = {c: n for c, n in [("gpn_msa_llr", "GPN-MSA"), ("evo2_7b_llr", "Evo2-7B"),
                  ("evo2_40b_llr", "Evo2-40B"), ("nt_llr", "NT-2.5B"),
                  ("caduceus_llr", "Caduceus"), ("cadd_rawscore", "CADD"),
                  ("phylop_100v", "phyloP")] if c in d.columns and d[c].notna().sum() > 100}
        ds.append(dict(name=f"TraitGym-{split}", df=d, models=models,
                       cons="consequence", matched=True))
    return ds


def run_e3a(ds):
    rows = []
    for D in ds:
        d = D["df"]; y = d["label_bin"].values
        comp = A.composition_cv(d, "label_bin", "chrom_norm", return_oof=True)
        for binvar, label in [("dGC_w51", "abs_dGC"), ("mu_5mer", "mu_5mer")]:
            v = np.abs(d[binvar].values) if binvar == "dGC_w51" else d[binvar].values
            q = pd.qcut(pd.Series(v).rank(method="first"), 4, labels=False)
            for b in range(4):
                mask = (q == b).values
                if y[mask].sum() < 5 or (y[mask] == 0).sum() < 5:
                    continue
                for col, pretty in D["models"].items():
                    a, ap, n = A.auc_pair(d[col].values[mask], y[mask])
                    ca, _, _ = A.auc_pair(comp["oof"][mask], y[mask], orient_score=False)
                    rows.append(dict(dataset=D["name"], binvar=label, bin=b, model=pretty,
                                     model_auroc=a, comp_auroc=ca, n=int(mask.sum()),
                                     n_pos=int(y[mask].sum())))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "e3_collapse.csv"), index=False)
    print(f"E3a -> results/e3_collapse.csv ({len(out)} rows)")
    return out


def composition_match(d, feats):
    """NN-match each positive to a distinct negative on standardized feats within the same
    consequence class. Returns boolean mask of the matched subset (pos + matched negs)."""
    d = d[np.isfinite(d[feats].astype("float64").values).all(axis=1)]
    pos = d[d.label_bin == 1]; neg = d[d.label_bin == 0]
    if len(pos) < 10 or len(neg) < 10:
        return None
    keep_idx = []
    used = set()
    for cons, gp in pos.groupby(d["cons_tmp"]):
        gn = neg[neg["cons_tmp"] == cons]
        if len(gn) == 0:
            continue
        sc = StandardScaler().fit(pd.concat([gp[feats], gn[feats]]).values)
        Xp = sc.transform(gp[feats].values); Xn = sc.transform(gn[feats].values)
        k = min(5, len(gn))
        nn = NearestNeighbors(n_neighbors=k).fit(Xn)
        _, ind = nn.kneighbors(Xp)
        gn_idx = gn.index.values
        for pi, cand in zip(gp.index.values, ind):
            keep_idx.append(pi)
            for c in cand:
                ni = gn_idx[c]
                if ni not in used:
                    used.add(ni); keep_idx.append(ni); break
    return keep_idx  # original index labels of the matched subset


def run_e3b(ds):
    rows = []
    for D in ds:
        d = D["df"].copy()
        d["cons_tmp"] = d[D["cons"]].astype(str) if D["cons"] in d.columns else "all"
        if any(f not in d.columns for f in MATCH_FEATS):
            continue
        keep_idx = composition_match(d, MATCH_FEATS)
        if keep_idx is None or len(keep_idx) < 20:
            continue
        dm = d.loc[keep_idx]
        print(f"[{D['name']}] composition-matched subset: {len(dm)} "
              f"(pos {int(dm.label_bin.sum())}) from {len(d)}")
        unm, mat = {}, {}
        for col, pretty in D["models"].items():
            a0, _, _ = A.auc_pair(d[col].values, d["label_bin"].values)
            a1, _, n1 = A.auc_pair(dm[col].values, dm["label_bin"].values)
            unm[pretty] = a0; mat[pretty] = a1
            rows.append(dict(dataset=D["name"], model=pretty, auroc_unmatched=a0,
                             auroc_comp_matched=a1, delta=a1 - a0, n_matched=n1))
        common = list(unm.keys())
        if len(common) >= 3:
            tau, _ = kendalltau([unm[k] for k in common], [mat[k] for k in common])
            print(f"    Kendall tau (ranking pre/post comp-matching) = {tau:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "e3_matched.csv"), index=False)
    print(f"E3b -> results/e3_matched.csv ({len(out)} rows)")
    return out


def run_e4():
    p = os.path.join(ROOT, "data", "kircher_mpra", "satmut_scored.parquet")
    if not os.path.exists(p):
        print("E4 skipped: no kircher scored table")
        return None
    d = pd.read_parquet(p)
    d = d[d["gpn_msa_llr"].notna() & d["log2_effect"].notna()].copy()
    # substitution type (6 classes), strand-collapsed
    comp_pair = {"A": "T", "T": "A", "C": "G", "G": "C"}

    def subtype(r):
        a, b = r["ref"], r["alt"]
        if a in "GT":  # collapse to pyrimidine-ref convention
            a, b = comp_pair[a], comp_pair[b]
        return f"{a}>{b}"
    d["subtype"] = d.apply(subtype, axis=1)
    rows = []
    for elem, g in d.groupby("locus"):
        if len(g) < 30:
            continue
        # raw Spearman
        r_raw, p_raw, n = A.partial_spearman(g["gpn_msa_llr"].values, g["log2_effect"].values)
        # partial: control substitution type dummies + dGC_w51 + cpg_ref
        dummies = pd.get_dummies(g["subtype"]).values.astype(float)
        cov = np.column_stack([dummies, g["dGC_w51"].values, g["cpg_ref"].values])
        r_par, p_par, _ = A.partial_spearman(g["gpn_msa_llr"].values,
                                             g["log2_effect"].values, covars=cov)
        rows.append(dict(element=elem, n=n, rho_raw=r_raw, rho_partial=r_par,
                         drop=r_raw - r_par))
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    out.to_csv(os.path.join(RES, "e4_kircher.csv"), index=False)
    print(f"\nE4 -> results/e4_kircher.csv ({len(out)} elements)")
    print(out.to_string(index=False))
    print(f"\n  pooled: mean rho_raw={out['rho_raw'].mean():.3f} "
          f"mean rho_partial={out['rho_partial'].mean():.3f} "
          f"mean drop={out['drop'].mean():.3f}")
    return out


def main():
    ds = load_binary()
    run_e3a(ds)
    run_e3b(ds)
    run_e4()


if __name__ == "__main__":
    main()
