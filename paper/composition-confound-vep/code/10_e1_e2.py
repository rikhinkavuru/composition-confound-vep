"""
E1 (confound characterization) + E2 (trivial-baseline recovery) across benchmarks.

E1: for each model, Spearman + PARTIAL Spearman (controlling for consequence/region) of the
    model's zero-shot score against context-free confounders (mu_5mer mutation rate, CpG,
    ΔGC, local k-mer shift). Signed LLR and |LLR| both reported (sign matters for the
    mechanism; |LLR| is the effect-magnitude used for VEP ranking).
E2: composition-only classifier (C1/C2 features, HistGBM, leave-one-chromosome-out CV) vs
    each model's per-chromosome AUROC/AUPRC; recovery ratio = (comp-0.5)/(model-0.5);
    paired DeLong comp-vs-model. Bootstrap CIs on the composition baseline.

Outputs: results/e1_confound.csv, results/e2_recovery.csv
Binary benchmarks only (ClinVar, TraitGym mendelian/complex, GLRB). Kircher (continuous) -> E4.

Usage: ~/Downloads/venv/bin/python code/10_e1_e2.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_utils as A  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

CONFOUNDERS = ["mu_5mer", "cpg_ref", "dGC_w51", "dkmer3_w51", "flank_gc_w51"]


def load_datasets():
    """Return list of dicts: name, df (with label_bin, chrom_norm), models{col->pretty},
    consequence_col (for partial control) or None."""
    ds = []

    # ---- ClinVar noncoding (unmatched, region-heterogeneous) ----
    p = os.path.join(ROOT, "data", "clinvar", "clinvar_scored.parquet")
    if os.path.exists(p):
        d = pd.read_parquet(p)
        d["label_bin"] = (d["label"] == "Pathogenic").astype(int)
        d["chrom_norm"] = d["chrom"].astype(str).str.replace("chr", "", regex=False)
        models = {}
        for c, name in [("gpn_msa_full", "GPN-MSA"), ("gpn_msa_llr_csv", "GPN-MSA(csv)"),
                        ("evo2_7b_llr", "Evo2-7B"), ("evo2_40b_llr", "Evo2-40B"),
                        ("cadd_csv", "CADD"), ("phylop_100way", "phyloP-100w"),
                        ("phylop_447way", "phyloP-447w"), ("spliceai", "SpliceAI")]:
            if c in d.columns and d[c].notna().sum() > 100:
                models[c] = name
        ds.append(dict(name="ClinVar", df=d, models=models,
                       consequence_col="ccre_class", matched=False))

    # ---- TraitGym mendelian + complex (matched) ----
    for split in ["mendelian", "complex"]:
        p = os.path.join(ROOT, "data", "traitgym", f"{split}_scored.parquet")
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        d["label_bin"] = d["label"].astype(int)
        d["chrom_norm"] = d["chrom"].astype(str).str.replace("chr", "", regex=False)
        models = {}
        for c, name in [("gpn_msa_llr", "GPN-MSA"), ("evo2_7b_llr", "Evo2-7B"),
                        ("evo2_40b_llr", "Evo2-40B"), ("nt_llr", "NT-2.5B"),
                        ("caduceus_llr", "Caduceus"), ("hyenadna_llr", "HyenaDNA"),
                        ("specieslm_llr", "SpeciesLM"), ("gpn_final_llr", "GPN"),
                        ("cadd_rawscore", "CADD"), ("phylop_100v", "phyloP-100v"),
                        ("phastcons_43p", "phastCons")]:
            if c in d.columns and d[c].notna().sum() > 100:
                models[c] = name
        ds.append(dict(name=f"TraitGym-{split}", df=d, models=models,
                       consequence_col="consequence", matched=True))

    # ---- GLRB eQTL test (low-signal stress) ----
    p = os.path.join(ROOT, "data", "glrb_eqtl", "test_scored.parquet")
    if os.path.exists(p):
        d = pd.read_parquet(p)
        d["label_bin"] = d["label"].astype(int)
        d["chrom_norm"] = d["chrom"].astype(str).str.replace("chr", "", regex=False)
        models = {c: n for c, n in [("gpn_msa_llr", "GPN-MSA")]
                  if c in d.columns and d[c].notna().sum() > 100}
        ds.append(dict(name="GLRB-eQTL", df=d, models=models,
                       consequence_col=None, matched=True))
    return ds


def consequence_codes(df, col):
    if col is None or col not in df.columns:
        return None
    return pd.Categorical(df[col].astype(str)).codes.astype(float)


def run_e1(ds):
    rows = []
    for D in ds:
        d = D["df"]; y = d["label_bin"].values
        cc = consequence_codes(d, D["consequence_col"])
        for col, pretty in D["models"].items():
            s = d[col].astype("float64").values
            absll = np.abs(s)
            for conf in CONFOUNDERS:
                if conf not in d.columns:
                    continue
                cv = d[conf].astype("float64").values
                r_s, p_s, n = A.partial_spearman(s, cv, covars=None)
                r_a, p_a, _ = A.partial_spearman(absll, cv, covars=None)
                rp_a, pp_a, _ = (A.partial_spearman(absll, cv, covars=cc)
                                 if cc is not None else (np.nan, np.nan, n))
                rows.append(dict(dataset=D["name"], model=pretty, confounder=conf, n=n,
                                 rho_signedLLR=r_s, rho_absLLR=r_a,
                                 partial_rho_absLLR=rp_a, matched=D["matched"]))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "e1_confound.csv"), index=False)
    print(f"E1 -> results/e1_confound.csv ({len(out)} rows)")
    # headline: |partial rho| vs mu_5mer per model/dataset
    hl = out[out.confounder == "mu_5mer"][["dataset", "model", "rho_absLLR", "partial_rho_absLLR"]]
    print(hl.to_string(index=False))
    return out


def run_e2(ds):
    rows = []
    for D in ds:
        d = D["df"]
        if d["label_bin"].nunique() < 2 or d["label_bin"].sum() < 10:
            continue
        comp = A.composition_cv(d, label_col="label_bin", chrom_col="chrom_norm",
                                return_oof=True)
        comp_boot = A.bootstrap_auc_ci(comp["oof"], d["label_bin"].values,
                                       n_boot=500, orient_score=False)
        print(f"\n[{D['name']}] composition-only LOCO-CV: "
              f"AUROC={comp['auroc']:.3f} AUPRC={comp['auprc']:.3f} "
              f"(CI {comp_boot['auroc_lo']:.3f}-{comp_boot['auroc_hi']:.3f}), "
              f"matched={D['matched']}")
        for col, pretty in D["models"].items():
            m = A.per_chrom_auc(d, col, "label_bin", "chrom_norm")
            rr = A.recovery_ratio(comp["auroc"], m["auroc"])
            # DeLong comp vs model on shared covered rows
            mask = d[col].notna().values & ~np.isnan(comp["oof"])
            s_model, _ = A.orient(d[col].values[mask], d["label_bin"].values[mask])
            aucd_c, aucd_m, pdl = A.delong_test(comp["oof"][mask], s_model,
                                                d["label_bin"].values[mask])
            rows.append(dict(dataset=D["name"], model=pretty, matched=D["matched"],
                             comp_auroc=comp["auroc"], comp_auprc=comp["auprc"],
                             comp_auroc_lo=comp_boot["auroc_lo"], comp_auroc_hi=comp_boot["auroc_hi"],
                             model_auroc=m["auroc"], model_auprc=m["auprc"],
                             model_n=m["n"], recovery_ratio=rr, delong_p=pdl))
            print(f"    {pretty:14s} AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f} "
                  f"recovery={rr:.2f} delong_p={pdl:.1e}")
    out = pd.DataFrame(rows)
    if len(out):
        out["delong_p_bh"] = A.benjamini_hochberg(out["delong_p"].fillna(1).values)
    out.to_csv(os.path.join(RES, "e2_recovery.csv"), index=False)
    print(f"\nE2 -> results/e2_recovery.csv ({len(out)} rows)")
    return out


def main():
    ds = load_datasets()
    print("datasets:", [(D["name"], len(D["df"]), list(D["models"].values())) for D in ds])
    run_e1(ds)
    run_e2(ds)


if __name__ == "__main__":
    main()
