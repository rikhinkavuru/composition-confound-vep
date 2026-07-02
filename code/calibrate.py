"""
calibrate_llr: a model-agnostic, post-hoc pentanucleotide calibration for zero-shot genomic
language-model variant-effect scores. Released as the constructive deliverable of the
composition-confound VEP audit.

The transform removes the mutation-expectedness component of a model's log-likelihood-ratio
score by subtracting a per-pentanucleotide-context neutral baseline. Two estimators:

  method="genome"  : subtract LLR_neutral(context) from a precomputed genome-wide neutral
                     table (results/neutral_5mer_gpnmsa.csv style; label-free, the GPN-Star
                     -faithful estimator). Pass neutral_table.
  method="control" : estimate LLR_neutral(context) leave-one-chromosome-out from the control
                     variants (label==0) in the evaluation set itself; no external table and
                     no test-fold label leakage. Pass label and chrom.

context is the 5-mer reference pentamer centred on the variant plus '>' plus the alt base,
e.g. "ACGTA>C". Use `mutation_context(chrom, pos, ref, alt, fasta)` to build it from hg38.

Example
-------
>>> from calibrate import calibrate_llr, mutation_context
>>> ctx = [mutation_context(c, p, r, a, fa) for c, p, r, a in variants]
>>> cal = calibrate_llr(raw_llr, context=ctx, method="genome", neutral_table=table)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mutation_context(chrom, pos, ref, alt, fasta):
    """5-mer reference pentamer (centered, 1-based pos) + '>' + alt, using a pysam FastaFile.
    Returns None if the pentamer center does not match ref or runs off the contig."""
    c = str(chrom)
    if not c.startswith("chr"):
        c = "chr" + c
    try:
        pent = fasta.fetch(c, int(pos) - 3, int(pos) + 2).upper()
    except Exception:
        return None
    if len(pent) != 5 or pent[2] != str(ref).upper():
        return None
    return f"{pent}>{str(alt).upper()}"


def calibrate_llr(score, context, method="genome", neutral_table=None,
                  label=None, chrom=None, is_transition=None, cpg=None):
    """Return the calibrated score array (raw minus per-context neutral baseline).

    Parameters
    ----------
    score : array-like of float           raw model LLR per variant.
    context : array-like of str           mutation context per variant (see mutation_context).
    method : {"genome", "control"}
    neutral_table : DataFrame or dict      for method="genome"; columns context, mean_score
                                           (or a {context: mean} mapping).
    label, chrom : array-like              for method="control"; binary label and chromosome.
    is_transition, cpg : array-like        optional fallback strata for unseen contexts.
    """
    s = np.asarray(score, dtype="float64")
    ctx = np.asarray(context, dtype=object)
    n = len(s)

    if method == "genome":
        if neutral_table is None:
            raise ValueError("method='genome' requires neutral_table")
        if isinstance(neutral_table, pd.DataFrame):
            nmap = dict(zip(neutral_table["context"], neutral_table["mean_score"]))
        else:
            nmap = dict(neutral_table)
        glob = np.nanmean(list(nmap.values())) if nmap else 0.0
        base = np.array([nmap.get(c, glob) for c in ctx], dtype="float64")
        return s - base

    if method == "control":
        if label is None or chrom is None:
            raise ValueError("method='control' requires label and chrom")
        y = np.asarray(label).astype(int)
        ch = np.asarray(chrom)
        ts = np.asarray(is_transition).astype(int) if is_transition is not None else np.zeros(n, int)
        cg = np.asarray(cpg).astype(int) if cpg is not None else np.zeros(n, int)
        out = np.full(n, np.nan)
        for test in pd.unique(ch):
            te = ch == test
            tr = (~te) & (y == 0) & ~np.isnan(s)
            if tr.sum() < 50:
                tr = (~te) & ~np.isnan(s)
            dtr = pd.DataFrame({"ctx": ctx[tr], "s": s[tr], "ts": ts[tr], "cpg": cg[tr]})
            cmean = dtr.groupby("ctx")["s"].mean()
            smean = dtr.groupby(["ts", "cpg"])["s"].mean()
            glob = dtr["s"].mean()
            for j in np.where(te)[0]:
                if np.isnan(s[j]):
                    continue
                b = cmean.get(ctx[j], np.nan)
                if np.isnan(b):
                    b = smean.get((ts[j], cg[j]), glob)
                out[j] = s[j] - b
        return out

    raise ValueError(f"unknown method {method!r}")
