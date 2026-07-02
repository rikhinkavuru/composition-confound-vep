"""
Build the unified per-variant score+confound matrix for the TraitGym matched benchmark
(the central "already-matched" benchmark of the audit; H2-H5).

TraitGym ships raw per-variant model scores under
  {mendelian,complex}_traits_matched_9/features/<MODEL>.parquet
each a single-column 'score' parquet, ROW-ALIGNED (by author design) to the split's
canonical variant table  {split}_matched_9/test.parquet.

To eliminate any alignment risk we:
  1. Fetch the canonical matched_9/test.parquet (the alignment anchor).
  2. Attach every model's 'score' column to it by row index.
  3. Recompute our C1/C2 confound features on the SAME table (so scores + features share
     one row order). Cross-check that this canonical table equals our earlier local copy.

Output: data/traitgym/{mendelian,complex}_scored.parquet
  = variant cols + C1/C2 features + one column per model score (raw LLR / conservation /
    CADD RawScore) + selected CADD confound annotations (GC, CpG, Roulette-MR, priPhyloP,
    SpliceAI-*), for downstream E1-E6.

Sign conventions (documented, handled in analysis not here):
  - gLM *_LLR 'score' = log p(alt) - log p(ref); more NEGATIVE = alt less likely =
    more deleterious/conserved. TraitGym ranks with -LLR or |LLR|.
  - phyloP/phastCons: higher = more conserved.
  - CADD RawScore: higher = more deleterious.

Usage: ~/Downloads/venv/bin/python code/06_build_traitgym_scored.py
"""
import os
import sys

import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import compute_features  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REF = os.path.join(ROOT, "data", "reference", "hg38.fa")
REPO = "songlab/TraitGym"

# single-'score'-column feature parquets -> output column name
LLR_MODELS = {
    "GPN-MSA_LLR": "gpn_msa_llr",
    "GPN_final_LLR": "gpn_final_llr",
    "evo2_1b_base_LLR": "evo2_1b_llr",
    "evo2_7b_LLR": "evo2_7b_llr",
    "evo2_40b_LLR": "evo2_40b_llr",
    "NucleotideTransformer_LLR": "nt_llr",
    "Caduceus_LLR": "caduceus_llr",
    "HyenaDNA_LLR": "hyenadna_llr",
    "SpeciesLM_LLR": "specieslm_llr",
    "AIDO.DNA_LLR": "aido_llr",
}
ABSLLR_MODELS = {
    "GPN-MSA_absLLR": "gpn_msa_absllr",
    "evo2_7b_absLLR": "evo2_7b_absllr",
    "evo2_40b_absLLR": "evo2_40b_absllr",
    "NucleotideTransformer_absLLR": "nt_absllr",
    "Caduceus_absLLR": "caduceus_absllr",
}
CONSERVATION = {
    "phyloP-100v": "phylop_100v",
    "phyloP-241m": "phylop_241m",
    "phastCons-43p": "phastcons_43p",
    "s_het": "s_het",
}
SUPERVISED = {  # distance-style scores (higher = more different ref/alt)
    "Borzoi_L2_L2": "borzoi_l2",
    "Enformer_L2_L2": "enformer_l2",
    "Sei": "sei",
}
# feature_name -> source column inside its parquet (default 'score')
SRC_COL = {
    "s_het": "s_het",
    "Borzoi_L2_L2": "all",
    "Enformer_L2_L2": "all",
    "Sei": "seqclass_max_absdiff",
}
# CADD confound-relevant annotation columns to carry along (from the 114-col CADD parquet)
CADD_KEEP = ["RawScore", "GC", "CpG", "priPhyloP", "mamPhyloP", "verPhyloP",
             "Roulette-MR", "Roulette-AR",
             "SpliceAI-acc-gain", "SpliceAI-acc-loss", "SpliceAI-don-gain", "SpliceAI-don-loss",
             "minDistTSS", "Dst2Splice", "motifECount", "RemapOverlapTF"]

SPLITS = {"mendelian": "mendelian_traits", "complex": "complex_traits"}


def fetch_score(split_dir, feat_name):
    p = hf_hub_download(REPO, repo_type="dataset",
                        filename=f"{split_dir}_matched_9/features/{feat_name}.parquet")
    return pd.read_parquet(p)


def build(split_key, split_dir):
    print(f"\n=== {split_key} ({split_dir}_matched_9) ===")
    anchor_path = hf_hub_download(REPO, repo_type="dataset",
                                  filename=f"{split_dir}_matched_9/test.parquet")
    var = pd.read_parquet(anchor_path).reset_index(drop=True)
    n = len(var)
    print(f"  anchor variant table: {n} rows, cols={list(var.columns)}")

    # sanity: our earlier local copy should be identical in variant identity
    local = os.path.join(ROOT, "data", "traitgym", f"{split_dir}_test.parquet")
    if os.path.exists(local):
        loc = pd.read_parquet(local).reset_index(drop=True)
        same = (len(loc) == n and (loc[["chrom", "pos", "ref", "alt"]].values
                                   == var[["chrom", "pos", "ref", "alt"]].values).all())
        print(f"  local copy identical variant order: {same}")

    out = var.copy()

    # attach all single-column score parquets by row index
    for group in (LLR_MODELS, ABSLLR_MODELS, CONSERVATION, SUPERVISED):
        for feat_name, col in group.items():
            try:
                s = fetch_score(split_dir, feat_name)
            except Exception as e:
                print(f"    [skip] {feat_name}: {e}")
                continue
            assert len(s) == n, f"{feat_name} len {len(s)} != {n}"
            src = SRC_COL.get(feat_name, "score")
            out[col] = s[src].values

    # CADD annotation columns
    try:
        cadd = fetch_score(split_dir, "CADD")
        assert len(cadd) == n
        for c in CADD_KEEP:
            if c in cadd.columns:
                safe = "cadd_" + c.lower().replace("-", "_")
                out[safe] = cadd[c].values
            else:
                print(f"    [warn] CADD missing col {c}")
    except Exception as e:
        print(f"    [skip] CADD: {e}")

    # recompute C1/C2 on the SAME anchor table (shared row order)
    feat = compute_features(var[["chrom", "pos", "ref", "alt"]].copy(),
                            ref_fasta_path=REF, n_jobs=1)
    added = [c for c in feat.columns if c not in var.columns]
    print(f"  C1/C2 recomputed: ref_match {int(feat['ref_match'].sum())}/{n}, "
          f"added {len(added)} cols")
    for c in added:
        out[c] = feat[c].values

    score_cols = [c for c in out.columns
                  if c not in var.columns and c not in added]
    print(f"  model/baseline score cols ({len(score_cols)}): {score_cols}")

    outp = os.path.join(ROOT, "data", "traitgym", f"{split_key}_scored.parquet")
    out.to_parquet(outp, index=False)
    print(f"  label balance: {out['label'].value_counts().to_dict()}")
    print(f"  saved -> data/traitgym/{split_key}_scored.parquet  ({out.shape})")


def main():
    for split_key, split_dir in SPLITS.items():
        build(split_key, split_dir)
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
