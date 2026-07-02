"""
Build the ClinVar noncoding-regulatory score+confound matrix (the region-heterogeneous,
*unmatched* benchmark where the composition confound is expected LARGEST; contrast against
the matched TraitGym benchmark).

Sources of per-variant model scores:
  1. goodarzilab/evo2-clinvar  (processed_clivnar_with_scores.csv) — precomputed
     Evo2-7B, Evo2-40B, GPN-MSA, CADD, NT-2.5B, phyloP (100/241/447/470-way), SpliceAI,
     splice_proximity, AlphaMissense, ESM2 for a subset of ClinVar SNVs. Joined on
     (chrom, pos=start[1-based], ref, alt). Covers ~73k of our 153k noncoding variants.
  2. GPN-MSA genome-wide lookup (data/gpn_msa/scores.tsv.bgz) — added by a separate step
     (code/08) to give GPN-MSA coverage on ALL 153k variants, incl. those absent from (1).

Starting point: data/clinvar/clinvar_noncoding_feat.parquet (153,508 rows, already carries
C1/C2 confound features + label + ccre_class).

Output: data/clinvar/clinvar_scored.parquet
  = our noncoding variants + C1/C2 + joined precomputed model scores (NaN where a model did
    not score that variant). Analysis uses each model's covered subset.

Sign conventions: Evo2/GPN-MSA raw LLR-like scores (more negative = more deleterious for
GPN-MSA); CADD higher = more deleterious; phyloP higher = more conserved. Handled downstream.

Usage: ~/Downloads/venv/bin/python code/07_build_clinvar_scored.py
"""
import glob
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SCORE_COLS = {
    "evo2_7b": "evo2_7b_llr",
    "evo2_40b": "evo2_40b_llr",
    "gpnmsa": "gpn_msa_llr_csv",   # CSV's GPN-MSA (partial); full-coverage added in code/08
    "cadd": "cadd_csv",
    "nt_2.5b_ms": "nt_llr",
    "nt_2.5b_1000g": "nt_1000g_llr",
    "phyloP100way": "phylop_100way",
    "phyloP241way": "phylop_241way",
    "phyloP447way": "phylop_447way",
    "phyloP470way": "phylop_470way",
    "spliceai": "spliceai",
    "splice_proximity": "splice_proximity",
    "alphamissense": "alphamissense",
    "esm2_650m": "esm2_650m",
    "gtf_feature": "gtf_feature",
}


def main():
    cv = pd.read_parquet(os.path.join(ROOT, "data", "clinvar", "clinvar_noncoding_feat.parquet"))
    cv["chrom_j"] = cv["chrom"].astype(str).str.replace("chr", "", regex=False)
    n0 = len(cv)
    print(f"ClinVar noncoding (C1/C2 annotated): {n0} rows")

    csv = glob.glob("/Users/rikhinkavuru/.cache/huggingface/**/processed_clivnar_with_scores.csv",
                    recursive=True)
    if not csv:
        from huggingface_hub import hf_hub_download
        csv = [hf_hub_download("goodarzilab/evo2-clinvar", repo_type="dataset",
                               filename="processed_clivnar_with_scores.csv")]
    usecols = ["chrom", "start", "ref_allele", "alt_allele", "variant_type"] + list(SCORE_COLS.keys())
    e = pd.read_csv(csv[0], dtype={"ref_allele": str, "alt_allele": str}, low_memory=False,
                    usecols=usecols)
    e = e[e["variant_type"] == "SNV"].copy()
    e["chrom_j"] = e["chrom"].astype(str).str.replace("chr", "", regex=False)
    e["pos"] = e["start"].astype(int)  # 1-based match verified (73,408 overlap vs 714 at +1)
    e = e.rename(columns={"ref_allele": "ref", "alt_allele": "alt"})
    # dedupe on physical variant (multiple transcript rows share identical DNA-model scores)
    e = e.drop_duplicates(subset=["chrom_j", "pos", "ref", "alt"], keep="first")
    keep = ["chrom_j", "pos", "ref", "alt"] + list(SCORE_COLS.keys())
    e = e[keep].rename(columns=SCORE_COLS)

    out = cv.merge(e, on=["chrom_j", "pos", "ref", "alt"], how="left")
    assert len(out) == n0, f"row count changed on merge: {len(out)} != {n0}"

    covered = out["evo2_7b_llr"].notna().sum()
    print(f"  joined evo2-clinvar scores: {covered}/{n0} variants covered "
          f"({100*covered/n0:.1f}%)")
    for src, col in SCORE_COLS.items():
        if col in out.columns and col != "gtf_feature":
            nn = out[col].notna().sum()
            print(f"    {col:20s} nonnull {nn}")

    out = out.drop(columns=["chrom_j"])
    outp = os.path.join(ROOT, "data", "clinvar", "clinvar_scored.parquet")
    out.to_parquet(outp, index=False)
    print(f"  label balance (all 153k): {out['label'].value_counts().to_dict()}")
    print(f"  saved -> data/clinvar/clinvar_scored.parquet  {out.shape}")


if __name__ == "__main__":
    main()
