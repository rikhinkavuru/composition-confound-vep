"""
Build a gnomAD-common control arm for the ClinVar pathogenic noncoding SNVs, to test whether
the composition confound is real or an artifact of ClinVar-benign ascertainment (the intro's
own caveat: CADD/GPN-MSA use gnomAD-common controls, not ClinVar-benign).

Controls = gnomAD v4.1 genome common SNVs (AF>=0.05) lying in the regulatory neighbourhoods of
the pathogenic variants (within merged +/-1kb spans), excluding any position that is itself a
ClinVar pathogenic variant. These are frequency-based, region-matched proxy-benign controls.

Then: score both arms with GPN-MSA (local lookup) + C1/C2, and compare composition recovery
against pathogenic under (a) ClinVar-benign controls and (b) gnomAD-common controls.

Output: data/clinvar/gnomad_common_controls.parquet (chrom,pos,ref,alt)
Usage: ~/Downloads/venv/bin/python code/16_gnomad_controls.py
"""
import os
import sys

import pandas as pd
import pysam

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GNOMAD = ("https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/vcf/genomes/"
          "gnomad.genomes.v4.1.sites.chr{c}.vcf.bgz")
PAD = 1000
AF_MIN = 0.05


def merge_spans(positions, gap=20000):
    """positions sorted -> merged [start,end] spans covering pos +/- PAD."""
    spans = []
    for p in positions:
        s, e = p - PAD, p + PAD
        if spans and s <= spans[-1][1] + gap:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    return spans


def main():
    cv = pd.read_parquet(os.path.join(ROOT, "data", "clinvar", "clinvar_scored.parquet"))
    path = cv[cv["label"] == "Pathogenic"].copy()
    path["chrom_c"] = path["chrom"].astype(str).str.replace("chr", "", regex=False)
    path_pos = set(zip(path["chrom_c"], path["pos"]))
    # subsample pathogenic *regions* for tractable remote querying (controls still ample);
    # positions kept for exclusion above use the full set.
    if len(path) > 2500:
        path = path.sample(n=2500, random_state=0)
    print(f"pathogenic noncoding variants (region seeds): {len(path)}", flush=True)

    rows = []
    for chrom, g in path.groupby("chrom_c"):
        if chrom in ("X", "Y", "MT", "M"):
            continue
        url = GNOMAD.format(c=chrom)
        try:
            vf = pysam.VariantFile(url)
        except Exception as e:
            print(f"  chr{chrom}: open failed {e}"); continue
        spans = merge_spans(sorted(g["pos"].unique()))
        if len(spans) > 40:  # cap spans/chrom to bound remote-streaming runtime
            import random
            random.Random(0).shuffle(spans)
            spans = spans[:40]
        found = 0
        for s, e in spans:
            try:
                for rec in vf.fetch(f"chr{chrom}", max(0, s), e):
                    af = rec.info.get("AF")
                    if (af and af[0] is not None and af[0] >= AF_MIN
                            and len(rec.ref) == 1 and rec.alts and len(rec.alts[0]) == 1
                            and rec.ref in "ACGT" and rec.alts[0] in "ACGT"
                            and (chrom, rec.pos) not in path_pos):
                        rows.append((f"chr{chrom}", rec.pos, rec.ref, rec.alts[0], af[0]))
                        found += 1
            except Exception:
                continue
        print(f"  chr{chrom}: {len(spans)} spans -> {found} common SNVs", flush=True)

    ctl = pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt", "gnomad_af"])
    ctl = ctl.drop_duplicates(["chrom", "pos", "ref", "alt"])
    outp = os.path.join(ROOT, "data", "clinvar", "gnomad_common_controls.parquet")
    ctl.to_parquet(outp, index=False)
    print(f"\ngnomAD-common controls: {len(ctl)} -> {outp}")


if __name__ == "__main__":
    main()
