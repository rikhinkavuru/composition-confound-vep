"""
Build data/kircher_mpra/satmut_mpra_combined.parquet

Source: kircherlab/MPRA_SaturationMutagenesis GitHub repo (the source code for
kircherlab.bihealth.org/satMutMPRA/), which bundles the site's underlying data
at data/elements.tsv.gz (raw variant-effect table across ALL elements the site
serves, both GRCh37 and GRCh38 coordinates) plus data/enhancers.tsv /
data/promoters.tsv (element name lists).

The bundled file covers MORE constructs (29) than the canonical 21 loci from
the original Kircher et al. 2019 Nat Commun paper (10 promoters, 10 enhancers,
UC88), because several loci were assayed under multiple conditions in
follow-up work hosted on the same site:
  - TERT: 4 cell-line/allele contexts (TERT-HEK, TERT-GBM, TERT-GAa, TERT-GSc)
  - PKLR: 2 timepoints (PKLR-24h, PKLR-48h)
  - LDLR: 2 constructs (LDLR, LDLR.2)
  - SORT1: 3 constructs (SORT1, SORT1.2, SORT1-flip)
  - ZRS: 2 haplotypes (ZRSh-13, ZRSh-13h2)
All other 12 loci (BCL11A, F9, FOXE1, GP1BA, HBB, HBG1, HNF4A, IRF4, IRF6,
MYCrs11986220, MYCrs6983267, RET, TCF7L2, UC88, ZFAND3 minus overlaps) have a
single construct. All 21 canonical loci are present -> nothing missing.

We keep GRCh38 rows only (the file has both builds) and construct-level
granularity (different conditions of the same locus have genuinely different
measured effects), while also recording a `locus` column that collapses
constructs back to the 21 canonical element names for grouping.
"""
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kircher_mpra"
RAW_TSV = DATA_DIR / "elements.tsv"
RAW_TSV_GZ = DATA_DIR / "elements.tsv.gz"
ENHANCERS_META = DATA_DIR / "enhancers_meta.tsv"
PROMOTERS_META = DATA_DIR / "promoters_meta.tsv"
OUT_PARQUET = DATA_DIR / "satmut_mpra_combined.parquet"

BASE_URL = (
    "https://raw.githubusercontent.com/kircherlab/MPRA_SaturationMutagenesis/"
    "master/data/{}"
)

# Canonical 21 elements from Kircher et al. 2019 Nat Commun 10:3583
# (10 promoters + 10 enhancers + UC88); construct-name -> canonical locus.
LOCUS_MAP = {
    "TERT-HEK": "TERT", "TERT-GBM": "TERT", "TERT-GAa": "TERT", "TERT-GSc": "TERT",
    "PKLR-24h": "PKLR", "PKLR-48h": "PKLR",
    "LDLR": "LDLR", "LDLR.2": "LDLR",
    "SORT1": "SORT1", "SORT1.2": "SORT1", "SORT1-flip": "SORT1",
    "ZRSh-13": "ZRS", "ZRSh-13h2": "ZRS",
}
CANONICAL_21 = {
    # 10 promoters
    "F9", "FOXE1", "GP1BA", "HBB", "HBG1", "HNF4A", "LDLR", "MSMB", "PKLR", "TERT",
    # 10 enhancers
    "BCL11A", "IRF4", "IRF6", "MYCrs11986220", "MYCrs6983267", "RET", "SORT1",
    "TCF7L2", "ZFAND3", "ZRS",
    # +1
    "UC88",
}


def download(name):
    dest = DATA_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already have {name}", file=sys.stderr)
        return
    url = BASE_URL.format(name)
    print(f"  downloading {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, dest)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Fetching bundled site data from kircherlab/MPRA_SaturationMutagenesis ...", file=sys.stderr)
    download("elements.tsv.gz")
    download("enhancers.tsv")
    download("promoters.tsv")

    import gzip, shutil
    if not RAW_TSV.exists():
        with gzip.open(RAW_TSV_GZ, "rb") as f_in, open(RAW_TSV, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    df = pd.read_csv(RAW_TSV, sep="\t", low_memory=False)
    print(f"Raw rows (both GRCh37 + GRCh38, all constructs): {len(df)}", file=sys.stderr)
    print(f"Distinct construct-level Element labels: {df['Element'].nunique()}", file=sys.stderr)

    elements_seen = set(df["Element"].unique())
    loci_seen = {LOCUS_MAP.get(e, e) for e in elements_seen}
    missing = CANONICAL_21 - loci_seen
    extra = loci_seen - CANONICAL_21
    print(f"Canonical 21 loci covered: {len(loci_seen & CANONICAL_21)} / 21", file=sys.stderr)
    if missing:
        print(f"  MISSING canonical loci: {sorted(missing)}", file=sys.stderr)
    if extra:
        print(f"  Extra loci beyond canonical 21 (unexpected): {sorted(extra)}", file=sys.stderr)

    g38 = df[df["Release"] == "GRCh38"].copy()
    print(f"GRCh38-only rows: {len(g38)}", file=sys.stderr)

    before_na = len(g38)
    g38 = g38.dropna(subset=["Chrom", "Pos", "Ref", "Alt"])
    print(f"Dropped {before_na - len(g38)} rows with unparseable coordinates (NaN chrom/pos/ref/alt)", file=sys.stderr)
    print(f"Rows after coordinate cleanup: {len(g38)}", file=sys.stderr)

    n_unique_variants = g38[["Chrom", "Pos", "Ref", "Alt"]].drop_duplicates().shape[0]
    print(f"Unique physical (chrom,pos,ref,alt) variants across all 21 loci: {n_unique_variants}", file=sys.stderr)

    out = pd.DataFrame({
        "element_name": g38["Element"].astype(str),
        "locus": g38["Element"].astype(str).map(lambda e: LOCUS_MAP.get(e, e)),
        "chrom": g38["Chrom"].astype(str),
        "pos": g38["Pos"].astype(int),
        "ref": g38["Ref"].astype(str),
        "alt": g38["Alt"].astype(str),  # "-" denotes a 1bp deletion
        "log2_effect": g38["Coefficient"].astype(float),
        "pvalue": g38["pValue"].astype(float),
        "n_barcodes": g38["Barcodes"].astype(int),
        "dna_count": g38["DNA"].astype(int),
        "rna_count": g38["RNA"].astype(int),
        "is_deletion": g38["Alt"].astype(str).eq("-"),
    })

    print(f"Final combined table rows: {len(out)}", file=sys.stderr)
    print(out["locus"].value_counts(), file=sys.stderr)
    print(f"SNVs: {(~out['is_deletion']).sum()}, 1bp deletions: {out['is_deletion'].sum()}", file=sys.stderr)

    out.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {len(out)} rows to {OUT_PARQUET}", file=sys.stderr)


if __name__ == "__main__":
    main()
