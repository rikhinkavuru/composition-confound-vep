"""
gnomAD-common control arm: re-run the ClinVar composition-recovery test with the SAME
pathogenic positives but frequency-based gnomAD-common controls instead of ClinVar-benign.
Tests whether the composition confound is real or an artifact of ClinVar-benign ascertainment.

Controls fetched via Colab (notebooks/gnomad_controls.ipynb) -> data/clinvar/gnomad_common_controls.csv.
Here we score them (C1/C2 + GPN-MSA local lookup) and compare, on the SAME chromosomes:
  Arm A: pathogenic vs ClinVar-benign   (original)
  Arm B: pathogenic vs gnomAD-common    (frequency controls)

Output: results/e2_gnomad_arm.csv
Usage: ~/Downloads/venv/bin/python code/19_gnomad_arm.py
"""
import os
import sys

import pandas as pd
import pysam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_utils as A  # noqa: E402
from features import compute_features  # noqa: E402
import importlib.util
spec = importlib.util.spec_from_file_location("gpnmod",
        os.path.join(os.path.dirname(__file__), "08_gpn_msa_lookup.py"))
gpnmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(gpnmod)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REF = os.path.join(ROOT, "data", "reference", "hg38.fa")
BGZ = os.path.join(ROOT, "data", "gpn_msa", "scores.tsv.bgz")


def main():
    ctl = pd.read_csv(os.path.join(ROOT, "data", "clinvar", "gnomad_common_controls.csv"),
                      dtype={"chrom": str})
    print(f"gnomAD-common controls: {len(ctl)}")
    # C1/C2 features
    ctl = compute_features(ctl, ref_fasta_path=REF, n_jobs=1)
    print(f"  C1/C2: ref_match {int(ctl['ref_match'].sum())}/{len(ctl)}")
    # GPN-MSA local lookup
    tbx = pysam.TabixFile(BGZ)
    ctl["gpn_msa_full"] = gpnmod.lookup_table(ctl, tbx, "gpn_msa_full")

    cv = pd.read_parquet(os.path.join(ROOT, "data", "clinvar", "clinvar_scored.parquet"))
    cv["chrom_norm"] = cv["chrom"].astype(str).str.replace("chr", "", regex=False)
    ctl["chrom_norm"] = ctl["chrom"].astype(str).str.replace("chr", "", regex=False)
    path = cv[cv["label"] == "Pathogenic"].copy()
    benign = cv[cv["label"] == "Benign"].copy()
    chroms = set(ctl["chrom_norm"]) & set(path["chrom_norm"])
    path = path[path["chrom_norm"].isin(chroms)]
    benign = benign[benign["chrom_norm"].isin(chroms)]
    ctl = ctl[ctl["chrom_norm"].isin(chroms)]

    feats = A.COMP_FEATURES
    rows = []
    for arm, neg in [("ClinVar-benign", benign), ("gnomAD-common", ctl)]:
        d = pd.concat([path.assign(label_bin=1), neg.assign(label_bin=0)], ignore_index=True)
        d = d.dropna(subset=[c for c in feats if c in d.columns])
        comp = A.composition_cv(d, "label_bin", "chrom_norm", feat_cols=feats, return_oof=True)
        gpn = A.per_chrom_auc(d, "gpn_msa_full", "label_bin", "chrom_norm")
        rr = A.recovery_ratio(comp["auroc"], gpn["auroc"])
        print(f"\n[{arm}]  n_pos={int((d.label_bin==1).sum())} n_neg={int((d.label_bin==0).sum())}")
        print(f"  composition-only AUROC={comp['auroc']:.3f}  GPN-MSA AUROC={gpn['auroc']:.3f}  "
              f"recovery={rr:.2f}")
        rows.append(dict(arm=arm, n_pos=int((d.label_bin == 1).sum()),
                         n_neg=int((d.label_bin == 0).sum()),
                         comp_auroc=comp["auroc"], gpn_auroc=gpn["auroc"], recovery=rr))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, "results", "e2_gnomad_arm.csv"), index=False)
    print("\nsaved results/e2_gnomad_arm.csv")


if __name__ == "__main__":
    main()
