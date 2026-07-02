"""
GPN-MSA genome-wide score lookup via local tabix (pysam) against the downloaded
data/gpn_msa/scores.tsv.bgz (+ .tbi). GPN-MSA is the alignment-based ANCHOR gLM present on
EVERY dataset (the only model we can put on ClinVar + Kircher + GLRB uniformly, since those
are not in TraitGym's precomputed roster).

File format (README-verified): columns = chrom  pos  ref  alt  score ; chrom has NO 'chr'
prefix (e.g. "17"); score more-negative = more constrained/deleterious. Scores exist for all
~9B possible SNVs, so coverage should be ~100%.

Augments, adding column `gpn_msa_llr` (and for ClinVar, `gpn_msa_full` to distinguish from the
partial CSV column):
  - data/clinvar/clinvar_scored.parquet        (full 153k coverage)
  - data/kircher_mpra/satmut_mpra_feat.parquet -> data/kircher_mpra/satmut_scored.parquet
  - data/glrb_eqtl/test_feat.parquet           -> data/glrb_eqtl/test_scored.parquet

Usage: ~/Downloads/venv/bin/python code/08_gpn_msa_lookup.py
"""
import os

import pandas as pd
import pysam

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BGZ = os.path.join(ROOT, "data", "gpn_msa", "scores.tsv.bgz")


def bare(c):
    c = str(c)
    return c[3:] if c.startswith("chr") else c


def lookup_table(df, tbx, out_col="gpn_msa_llr"):
    """Query one GPN-MSA score per (chrom,pos,ref,alt). Groups by chrom, one fetch per
    position, matches ref/alt. Returns a float Series aligned to df.index."""
    scores = pd.Series(index=df.index, dtype="float64")
    df = df.copy()
    df["_bare"] = df["chrom"].map(bare)
    hit = miss = 0
    for chrom, g in df.groupby("_bare"):
        try:
            contig_ok = chrom in tbx.contigs
        except Exception:
            contig_ok = True
        if not contig_ok:
            miss += len(g)
            continue
        for idx, row in g.iterrows():
            pos = int(row["pos"]); ref = str(row["ref"]); alt = str(row["alt"])
            try:
                recs = tbx.fetch(chrom, pos - 1, pos)
            except Exception:
                recs = []
            val = None
            for r in recs:
                f = r.split("\t")
                if len(f) >= 5 and int(f[1]) == pos and f[2] == ref and f[3] == alt:
                    val = float(f[4]); break
            if val is None:
                miss += 1
            else:
                scores.at[idx] = val; hit += 1
    print(f"    matched {hit}/{len(df)} ({100*hit/len(df):.1f}%), missed {miss}")
    return scores


def main():
    assert os.path.exists(BGZ), f"missing {BGZ}"
    assert os.path.exists(BGZ + ".tbi"), f"missing {BGZ}.tbi"
    tbx = pysam.TabixFile(BGZ)

    # sanity: BRCA1 example from README (chr17:43044295 T>A ~ -1.60)
    for r in tbx.fetch("17", 43044294, 43044295):
        print("  sanity:", r); break

    jobs = [
        ("ClinVar (full)", "data/clinvar/clinvar_scored.parquet",
         "data/clinvar/clinvar_scored.parquet", "gpn_msa_full"),
        ("Kircher satMutMPRA", "data/kircher_mpra/satmut_mpra_feat.parquet",
         "data/kircher_mpra/satmut_scored.parquet", "gpn_msa_llr"),
        ("GLRB eQTL test", "data/glrb_eqtl/test_feat.parquet",
         "data/glrb_eqtl/test_scored.parquet", "gpn_msa_llr"),
    ]
    for label, inp, outp, col in jobs:
        df = pd.read_parquet(os.path.join(ROOT, inp))
        print(f"\n=== {label}: {len(df)} variants ===")
        df[col] = lookup_table(df, tbx, col)
        df.to_parquet(os.path.join(ROOT, outp), index=False)
        print(f"  saved -> {outp}  (col '{col}', nonnull {df[col].notna().sum()})")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
