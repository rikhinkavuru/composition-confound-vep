"""
Build a genome-wide neutral pentanucleotide score table for GPN-MSA by sampling random
autosomal SNVs from the released genome-wide scores (data/gpn_msa/scores.tsv.bgz). This is
the label-free, genome-representative neutral baseline that GPN-Star's calibration uses, and
it is a stronger E5 flagship than the control-variant estimate.

For each sampled position we read the pentamer from hg38 and GPN-MSA's score for each of the
three alternate alleles, then average by context (pentamer_ref, alt) -> LLR_neutral(context).
calibrated = raw - LLR_neutral(context).

Output: results/neutral_5mer_gpnmsa.csv (context, n, mean_score); and an E5 recheck on ClinVar
of raw vs genome-neutral-calibrated GPN-MSA (mutation-rate coupling + AUROC).

Usage: ~/Downloads/venv/bin/python code/14_gpn_neutral_table.py [n_positions]
"""
import os
import sys

import numpy as np
import pandas as pd
import pysam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_utils as A  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BGZ = os.path.join(ROOT, "data", "gpn_msa", "scores.tsv.bgz")
REF = os.path.join(ROOT, "data", "reference", "hg38.fa")
RES = os.path.join(ROOT, "results")

N_WINDOWS = int(sys.argv[1]) if len(sys.argv) > 1 else 250
WIN = 50_000  # bp per window; contiguous streamed reads are far faster than random seeks
SEED = 0


def main():
    fa = pysam.FastaFile(REF)
    tbx = pysam.TabixFile(BGZ)
    autosomes = [str(c) for c in range(1, 23)]
    lengths = {c: fa.get_reference_length("chr" + c) for c in autosomes}
    total = sum(lengths.values())
    weights = np.array([lengths[c] for c in autosomes], dtype=float) / total
    rng = np.random.default_rng(SEED)

    from collections import defaultdict
    acc = defaultdict(lambda: [0.0, 0])  # context -> [sum, n]
    done = 0
    for w in range(N_WINDOWS):
        c = autosomes[rng.choice(len(autosomes), p=weights)]
        start = int(rng.integers(3, lengths[c] - WIN - 3))  # 1-based
        end = start + WIN
        seq = fa.fetch("chr" + c, start - 3, end + 2).upper()  # pad for pentamers
        try:
            recs = tbx.fetch(c, start - 1, end)  # streamed, contiguous
        except Exception:
            continue
        for r in recs:
            f = r.split("\t")
            if len(f) < 5:
                continue
            p = int(f[1])
            off = p - start + 2  # index of center base in `seq` (fetch is 0-based half-open)
            if off < 2 or off + 3 > len(seq):
                continue
            pent = seq[off - 2:off + 3]
            if len(pent) != 5 or pent[2] != f[2] or any(b not in "ACGT" for b in pent):
                continue
            try:
                s = float(f[4])
            except ValueError:
                continue
            acc[pent + ">" + f[3]][0] += s
            acc[pent + ">" + f[3]][1] += 1
        done += 1
        if done % 25 == 0:
            tot = sum(v[1] for v in acc.values())
            print(f"  windows {done}/{N_WINDOWS}, {len(acc)} contexts, {tot} scores")

    rows = [(k, v[1], v[0] / v[1]) for k, v in acc.items() if v[1] > 0]
    tab = pd.DataFrame(rows, columns=["context", "n", "mean_score"]).sort_values("context")
    outp = os.path.join(RES, "neutral_5mer_gpnmsa.csv")
    tab.to_csv(outp, index=False)
    print(f"neutral table: {len(tab)} contexts, {tab['n'].sum()} scores -> {outp}")

    # E5 recheck on ClinVar with the genome-wide neutral calibration
    ctxcache = os.path.join(ROOT, "data", "clinvar", "clinvar_scored_ctx.parquet")
    if not os.path.exists(ctxcache):
        print("(clinvar ctx cache missing; skipping recheck)")
        return
    d = pd.read_parquet(ctxcache)
    d["label_bin"] = (d["label"] == "Pathogenic").astype(int)
    d["chrom_norm"] = d["chrom"].astype(str).str.replace("chr", "", regex=False)
    nmap = dict(zip(tab["context"], tab["mean_score"]))
    neutral = d["context5"].map(nmap)
    raw = d["gpn_msa_full"].astype("float64")
    cal = raw - neutral
    cov = neutral.notna().mean()
    r_raw, _, _ = A.partial_spearman(np.abs(raw.values), d["mu_5mer"].values)
    r_cal, _, _ = A.partial_spearman(np.abs(cal.values), d["mu_5mer"].values)
    a_raw = A.per_chrom_auc(d.assign(_s=raw), "_s", "label_bin", "chrom_norm")["auroc"]
    a_cal = A.per_chrom_auc(d.assign(_s=cal), "_s", "label_bin", "chrom_norm")["auroc"]
    print(f"\nGenome-wide-neutral calibration of GPN-MSA on ClinVar (context coverage {cov:.3f}):")
    print(f"  |rho|(|LLR|,mu): raw {abs(r_raw):.3f} -> calibrated {abs(r_cal):.3f}")
    print(f"  AUROC:           raw {a_raw:.3f} -> calibrated {a_cal:.3f}")
    pd.DataFrame([dict(model="GPN-MSA", dataset="ClinVar", corr_mu_raw=r_raw, corr_mu_cal=r_cal,
                       auroc_raw=a_raw, auroc_cal=a_cal, ctx_coverage=cov)]
                 ).to_csv(os.path.join(RES, "e5_genome_neutral_gpnmsa.csv"), index=False)


if __name__ == "__main__":
    main()
