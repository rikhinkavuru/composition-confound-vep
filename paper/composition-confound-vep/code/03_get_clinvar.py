"""
Build data/clinvar/clinvar_noncoding_labeled.parquet

Pipeline:
1. Parse ClinVar VCF (GRCh38) with pysam.VariantFile.
2. Filter to SNVs only (single ref base, single alt base, both in ACGT).
3. Extract CLNSIG / CLNREVSTAT.
4. Keep only >=2-star review status:
     - criteria_provided,_multiple_submitters,_no_conflicts
     - reviewed_by_expert_panel
     - practice_guideline
5. Label Pathogenic/Likely_pathogenic vs Benign/Likely_benign. Drop VUS,
   conflicting, and anything else.
6. Intersect with ENCODE cCRE regions (GRCh38-cCREs.bed) via bedtools intersect
   (chr-prefix normalized: ClinVar VCF uses "1","2",...; cCRE bed uses "chr1","chr2",...).
7. Write labeled + region-annotated table to parquet.

Prints real counts at every filtering stage (no estimates).
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pysam

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "clinvar"
VCF_PATH = DATA_DIR / "clinvar.vcf.gz"
CCRE_BED = DATA_DIR / "GRCh38-cCREs.bed"
OUT_PARQUET = DATA_DIR / "clinvar_noncoding_labeled.parquet"

TWO_STAR_PLUS = {
    "criteria_provided,_multiple_submitters,_no_conflicts",
    "reviewed_by_expert_panel",
    "practice_guideline",
}

PATHOGENIC_LABELS = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
BENIGN_LABELS = {"Benign", "Likely_benign", "Benign/Likely_benign"}

ACGT = set("ACGT")


def main():
    print(f"Reading VCF: {VCF_PATH}", file=sys.stderr)
    vf = pysam.VariantFile(str(VCF_PATH))

    total_records = 0
    total_alt_alleles = 0
    snv_count = 0
    two_star_count = 0
    labeled_rows = []

    for rec in vf:
        total_records += 1
        info = rec.info
        clnsig_raw = info.get("CLNSIG")
        clnrevstat_raw = info.get("CLNREVSTAT")

        # CLNSIG / CLNREVSTAT are Number=. (comma-joined) fields; pysam returns
        # tuples split on the raw commas in the string, so rejoin with ","
        # to recover the exact original token.
        clnsig = ",".join(clnsig_raw) if clnsig_raw else None
        clnrevstat = ",".join(clnrevstat_raw) if clnrevstat_raw else None

        ref = rec.ref
        alts = rec.alts or ()
        for alt in alts:
            total_alt_alleles += 1
            if ref is None or len(ref) != 1 or len(alt) != 1:
                continue
            if ref not in ACGT or alt not in ACGT:
                continue
            snv_count += 1

            if clnrevstat not in TWO_STAR_PLUS:
                continue
            two_star_count += 1

            if clnsig in PATHOGENIC_LABELS:
                label = "Pathogenic"
            elif clnsig in BENIGN_LABELS:
                label = "Benign"
            else:
                continue  # VUS / conflicting / other -> drop

            labeled_rows.append(
                {
                    "chrom": rec.chrom,
                    "pos": rec.pos,  # 1-based VCF position
                    "ref": ref,
                    "alt": alt,
                    "clnsig": clnsig,
                    "clnrevstat": clnrevstat,
                    "label": label,
                    "variant_id": info.get("ALLELEID", None),
                }
            )

        if total_records % 500000 == 0:
            print(f"  ...{total_records} VCF records processed", file=sys.stderr)

    print(f"Total VCF records: {total_records}", file=sys.stderr)
    print(f"Total alt alleles seen: {total_alt_alleles}", file=sys.stderr)
    print(f"SNVs (single ref/alt base, ACGT only): {snv_count}", file=sys.stderr)
    print(f"SNVs with >=2-star review status: {two_star_count}", file=sys.stderr)
    print(f"SNVs P/LP vs B/LB after dropping VUS/conflicting: {len(labeled_rows)}", file=sys.stderr)

    df = pd.DataFrame(labeled_rows)
    print(df["label"].value_counts(), file=sys.stderr)

    # ---- cCRE intersection via bedtools ----
    df["chrom_bed"] = "chr" + df["chrom"].astype(str)
    # BED is 0-based half-open; VCF pos is 1-based -> start = pos-1, end = pos
    df["start0"] = df["pos"] - 1
    df["end0"] = df["pos"]
    df["row_id"] = range(len(df))

    variants_bed = DATA_DIR / "_variants_for_intersect.bed"
    df[["chrom_bed", "start0", "end0", "row_id"]].to_csv(
        variants_bed, sep="\t", header=False, index=False
    )

    intersect_out = DATA_DIR / "_variants_ccre_intersect.bed"
    cmd = [
        "bedtools",
        "intersect",
        "-a",
        str(variants_bed),
        "-b",
        str(CCRE_BED),
        "-loj",
    ]
    print("Running:", " ".join(cmd), file=sys.stderr)
    with open(intersect_out, "w") as fh:
        subprocess.run(cmd, stdout=fh, check=True)

    # bedtools -loj output columns:
    # a: chrom_bed, start0, end0, row_id
    # b: chrom, start, end, EH38D..., EH38E..., ccre_class  (or "." x6 if no overlap)
    inter_cols = [
        "chrom_bed", "start0", "end0", "row_id",
        "ccre_chrom", "ccre_start", "ccre_end", "ccre_dcc", "ccre_encode_acc", "ccre_class",
    ]
    inter = pd.read_csv(intersect_out, sep="\t", header=None, names=inter_cols)

    # collapse multiple cCRE class hits per variant into a single semicolon list
    inter["ccre_class"] = inter["ccre_class"].fillna(".")
    hits = (
        inter[inter["ccre_class"] != "."]
        .groupby("row_id")["ccre_class"]
        .agg(lambda s: ";".join(sorted(set(s))))
    )

    df["ccre_class"] = df["row_id"].map(hits)
    df["in_ccre"] = df["ccre_class"].notna()
    df["ccre_class"] = df["ccre_class"].fillna("")

    n_in_ccre = int(df["in_ccre"].sum())
    print(f"Variants overlapping >=1 cCRE: {n_in_ccre} / {len(df)}", file=sys.stderr)
    print(
        df.loc[df["in_ccre"], "ccre_class"].str.split(";").explode().value_counts(),
        file=sys.stderr,
    )

    df = df.drop(columns=["chrom_bed", "start0", "end0", "row_id"])
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PARQUET}", file=sys.stderr)

    variants_bed.unlink(missing_ok=True)
    intersect_out.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
