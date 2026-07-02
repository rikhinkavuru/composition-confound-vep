"""
Download TraitGym (songlab/TraitGym) test splits for the mendelian_traits and
complex_traits configs, and save them locally as parquet under data/traitgym/.

Usage: ~/Downloads/venv/bin/python code/01_get_traitgym.py

Notes on config choice (verified against the HF dataset card, 2026-07-01):
  The repo `songlab/TraitGym` defines four `load_dataset` config names in its
  README YAML front matter:
    - "mendelian_traits"      -> mendelian_traits_matched_9/test.parquet  (used here)
    - "complex_traits"        -> complex_traits_matched_9/test.parquet   (used here)
    - "mendelian_traits_full" -> mendelian_traits_all/test.parquet       (unmatched, all variants)
    - "complex_traits_full"   -> complex_traits_all/test.parquet        (unmatched, all variants)
  CLAUDE.md's dataset-feasibility note says to use "mendelian_traits"/"complex_traits",
  which resolve to the `_matched_9` (9 controls : 1 positive) variant-matched files.
  This script uses exactly those two configs, split="test" (the only split TraitGym ships).
"""
import os
from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "traitgym")
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

CONFIGS = {
    "mendelian_traits": "mendelian_traits_test.parquet",
    "complex_traits": "complex_traits_test.parquet",
}

EXPECTED_TRAIT_COUNTS = {
    "mendelian_traits": ("OMIM", 113),
    "complex_traits": ("trait", 83),  # atomic traits after splitting comma-joined groups
}


def atomic_trait_count(df, col):
    atoms = set()
    for val in df[col].dropna().unique():
        for a in str(val).split(","):
            if a:
                atoms.add(a)
    return len(atoms)


def main():
    summary = {}
    for cfg, fname in CONFIGS.items():
        print(f"=== Loading songlab/TraitGym config={cfg!r} split=test ===")
        ds = load_dataset("songlab/TraitGym", cfg, split="test")
        df = ds.to_pandas()
        out_path = os.path.join(OUT_DIR, fname)
        df.to_parquet(out_path, index=False)

        n_rows = len(df)
        n_pos = int(df["label"].sum())
        n_neg = int((~df["label"]).sum())
        cols = list(df.columns)

        trait_col, expected_n = EXPECTED_TRAIT_COUNTS[cfg]
        if cfg == "mendelian_traits":
            n_traits = df[trait_col].nunique()
        else:
            n_traits = atomic_trait_count(df, trait_col)

        print(f"  rows={n_rows}  positives={n_pos}  negatives={n_neg}")
        print(f"  columns={cols}")
        print(f"  distinct traits ({trait_col})={n_traits} (expected {expected_n})")
        if n_traits != expected_n:
            print(f"  *** DISCREPANCY: expected {expected_n} traits, got {n_traits} ***")

        # sanity check matched_9 ratio: each match_group should have 1 positive + 9 controls
        grp_sizes = df.groupby("match_group").size()
        grp_pos = df.groupby("match_group")["label"].sum()
        ratio_ok = (grp_sizes == 10).all() and (grp_pos == 1).all()
        print(f"  matched_9 structure (10 rows/group, 1 positive/group) holds: {ratio_ok}")

        summary[cfg] = dict(
            path=out_path, n_rows=n_rows, n_pos=n_pos, n_neg=n_neg,
            columns=cols, n_traits=n_traits, matched_9_ok=bool(ratio_ok),
        )
        print(f"  saved -> {out_path}\n")

    print("=== DONE ===")
    for cfg, s in summary.items():
        print(cfg, s)


if __name__ == "__main__":
    main()
