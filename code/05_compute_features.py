"""
Annotate every benchmark variant table with the C1/C2 composition & mutation-expectedness
features (code/features.py, already unit-tested 27/27). This is the confound axis used by
every downstream experiment (E1-E6) and has zero external dependency beyond the indexed
hg38 reference.

Datasets annotated:
  - ClinVar noncoding-regulatory subset (in_ccre==True)  -> data/clinvar/clinvar_noncoding_feat.parquet
  - TraitGym Mendelian                                    -> data/traitgym/mendelian_traits_feat.parquet
  - TraitGym complex                                      -> data/traitgym/complex_traits_feat.parquet
  - Kircher satMutMPRA (SNVs only; 1bp dels dropped)      -> data/kircher_mpra/satmut_mpra_feat.parquet
  - GLRB eQTL test split                                  -> data/glrb_eqtl/test_feat.parquet

Kircher deletions (alt=="-") are excluded here: the C1/C2 pipeline is defined for single-base
SNV substitutions. Deletions are retained in the raw table for any separate indel analysis.

Usage: ~/Downloads/venv/bin/python code/05_compute_features.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import compute_features  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REF = os.path.join(ROOT, "data", "reference", "hg38.fa")

# (label, input parquet, row-filter fn or None, output parquet)
JOBS = [
    ("ClinVar noncoding",
     "data/clinvar/clinvar_noncoding_labeled.parquet",
     lambda d: d[d["in_ccre"] == True].copy(),
     "data/clinvar/clinvar_noncoding_feat.parquet"),
    ("TraitGym Mendelian",
     "data/traitgym/mendelian_traits_test.parquet",
     None,
     "data/traitgym/mendelian_traits_feat.parquet"),
    ("TraitGym complex",
     "data/traitgym/complex_traits_test.parquet",
     None,
     "data/traitgym/complex_traits_feat.parquet"),
    ("Kircher satMutMPRA (SNVs)",
     "data/kircher_mpra/satmut_mpra_combined.parquet",
     lambda d: d[(~d["is_deletion"]) & (d["ref"].str.len() == 1) & (d["alt"].str.len() == 1)].copy(),
     "data/kircher_mpra/satmut_mpra_feat.parquet"),
    ("GLRB eQTL test",
     "data/glrb_eqtl/test.parquet",
     None,
     "data/glrb_eqtl/test_feat.parquet"),
]

FEATURE_COLS = None  # discovered from first run for reporting


def main():
    for label, inp, filt, outp in JOBS:
        inp_abs = os.path.join(ROOT, inp)
        outp_abs = os.path.join(ROOT, outp)
        df = pd.read_parquet(inp_abs)
        n0 = len(df)
        if filt is not None:
            df = filt(df)
        print(f"\n=== {label}: {inp}  ({n0} -> {len(df)} rows after filter) ===")

        feat = compute_features(df, ref_fasta_path=REF, n_jobs=1)

        # QC
        ref_ok = int(feat["ref_match"].sum())
        feat_ok = int(feat["feat_ok"].sum())
        n = len(feat)
        print(f"  ref_match: {ref_ok}/{n} ({100*ref_ok/n:.2f}%)")
        print(f"  feat_ok:   {feat_ok}/{n} ({100*feat_ok/n:.2f}%)")
        added = [c for c in feat.columns if c not in df.columns]
        print(f"  added {len(added)} feature cols: {added}")
        nan_mu = int(feat["mu_5mer"].isna().sum())
        print(f"  mu_5mer NaN: {nan_mu}  | mu_5mer range: "
              f"[{feat['mu_5mer'].min():.3g}, {feat['mu_5mer'].max():.3g}]")

        feat.to_parquet(outp_abs, index=False)
        print(f"  saved -> {outp}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
