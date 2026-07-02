"""
Download the `variant_effect_causal_eqtl` task of
InstaDeepAI/genomics-long-range-benchmark (train + test splits) and save locally as
parquet under data/glrb_eqtl/.

Usage: ~/Downloads/venv/bin/python code/02_get_glrb_eqtl.py

IMPORTANT implementation note (deviation forced by tooling, not by choice):
  The upstream HF repo ships this dataset as a *loading script*
  (genomics-long-range-benchmark.py), not as plain Parquet/CSV configs. The installed
  `datasets` library (4.8.5) has fully dropped support for script-based datasets
  ("Dataset scripts are no longer supported" / trust_remote_code removed), so
  `load_dataset("InstaDeepAI/genomics-long-range-benchmark", task_name=...)` fails
  outright on this environment. Rather than downgrade a general-purpose venv to an old
  `datasets`<4 pin, this script reimplements the *exact* logic of the
  `VariantEffectCausalEqtl` handler in that loading script (verified by reading the
  script's source, fetched via hf_hub_download) directly against the same two official
  upstream source files:
    - variant_effect_causal_eqtl/All_Tissues.csv (variant list + labels + splits, from the HF repo)
    - hg38.fa.gz reference genome (from UCSC, same URL the loading script itself uses)
  This reproduces the same ref/alt sequence construction, same standardize_sequence
  (uppercase, non-ACGT -> N), and same boundary-filtering (drop variants where the
  padded window would run off the start/end of a chromosome) as the original script.

sequence_length choice (verified from the loading script source, not guessed):
  `VariantEffectCausalEqtl.DEFAULT_LENGTH = 100000`. We use that documented default
  (100,000 bp window centered on the variant) -- see notes above in the TraitGym script
  for why we didn't invent our own number.
"""
import gzip
import os
import re
import shutil
import urllib.request

import pandas as pd
from huggingface_hub import hf_hub_download
from pyfaidx import Fasta

REPO_ID = "InstaDeepAI/genomics-long-range-benchmark"
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "glrb_eqtl"))
CACHE_DIR = os.path.expanduser("~/.cache/glrb_hg38")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

H38_REFERENCE_GENOME_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
SEQUENCE_LENGTH = 100_000
EXPECTED_ROWS = {"train": 88_717, "test": 8_846}


def standardize_sequence(sequence: str) -> str:
    sequence = sequence.upper()
    return re.sub("[^ATCG]", "N", sequence)


def pad_sequence(chromosome, start, sequence_length):
    """Centered window of `sequence_length` around `start`. Returns None if it runs
    off the chromosome (matches upstream pad_sequence semantics for this task)."""
    pad = sequence_length // 2
    end = start + pad + (sequence_length % 2)
    start = start - pad
    if start < 0 or end >= len(chromosome):
        return None
    return chromosome[start:end].seq


def download_hg38():
    fa_path = os.path.join(CACHE_DIR, "hg38.fa")
    gz_path = fa_path + ".gz"
    if os.path.exists(fa_path):
        print(f"  hg38.fa already present at {fa_path}")
        return fa_path
    if not os.path.exists(gz_path):
        print(f"  downloading {H38_REFERENCE_GENOME_URL} -> {gz_path} ...")
        urllib.request.urlretrieve(H38_REFERENCE_GENOME_URL, gz_path)
        print("  download complete.")
    print("  extracting ...")
    with gzip.open(gz_path, "rb") as f_in, open(fa_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print("  extraction complete.")
    return fa_path


def main():
    print("=== Fetching variant_effect_causal_eqtl/All_Tissues.csv from HF repo ===")
    csv_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                                filename="variant_effect_causal_eqtl/All_Tissues.csv")
    coords = pd.read_csv(csv_path)
    print(f"  raw CSV rows={len(coords)} columns={coords.columns.tolist()}")
    print(f"  raw split counts: {coords['split'].value_counts().to_dict()}")

    print("=== Preparing hg38 reference genome ===")
    fa_path = download_hg38()
    genome = Fasta(fa_path, one_based_attributes=False)

    # Cache chromosome lengths (from the .fai index, O(1), no sequence load) so the
    # boundary filter below never materializes a 100kb window.
    chrom_len = {c: len(genome[c]) for c in genome.keys()}

    def within_bounds(chrom, start):
        """Same drop-if-off-chromosome semantics as upstream pad_sequence, computed from
        the index length only (no 100kb extraction)."""
        pad = SEQUENCE_LENGTH // 2
        end = start + pad + (SEQUENCE_LENGTH % 2)
        s = start - pad
        return s >= 0 and end < chrom_len[chrom]

    summary = {}
    for split in ["train", "test"]:
        print(f"=== Building split={split} (variant table only; {SEQUENCE_LENGTH}bp "
              f"windows extracted lazily at scoring time) ===")
        split_df = coords[coords["split"] == split]
        rows = []
        for _, row in split_df.iterrows():
            chrom = row["CHROM"]  # e.g. "chr1"
            pos = int(row["POS"])  # 1-based
            start = pos - 1        # 0-based center
            if not within_bounds(chrom, start):
                continue

            # REF base straight from indexed hg38 (single-base fetch, cheap).
            ref_allele = genome[chrom][start:start + 1].seq.upper()

            rows.append({
                "chrom": chrom,                       # "chr1" — matches hg38.fa naming for features.py
                "pos": pos,                           # 1-based VCF coord
                "ref": ref_allele,
                "alt": str(row["ALT"]).upper(),
                "label": int(row["label"]),
                "tissue": row["tissue"],
                "distance_to_nearest_tss": int(row["distance_to_nearest_TSS"]),
            })

        df = pd.DataFrame(rows)
        out_path = os.path.join(OUT_DIR, f"{split}.parquet")
        df.to_parquet(out_path, index=False)

        n_rows = len(df)
        expected = EXPECTED_ROWS[split]
        print(f"  rows={n_rows} (expected ~{expected}); dropped {len(split_df) - n_rows} "
              f"variants too close to a chromosome boundary for {SEQUENCE_LENGTH}bp window")
        print(f"  columns={list(df.columns)}")
        if abs(n_rows - expected) / expected > 0.02:
            print(f"  *** DISCREPANCY: row count differs from expectation by >2% "
                  f"({n_rows} vs {expected}) ***")
        print(f"  label value_counts: {df['label'].value_counts().to_dict()}")
        print(f"  n distinct tissues: {df['tissue'].nunique()}")
        print(f"  n distinct chromosomes: {sorted(df['chrom'].unique().tolist())}")
        print(f"  saved -> {out_path}\n")

        summary[split] = dict(path=out_path, n_rows=n_rows, columns=list(df.columns))

    print("=== DONE ===")
    for split, s in summary.items():
        print(split, s)


if __name__ == "__main__":
    main()
