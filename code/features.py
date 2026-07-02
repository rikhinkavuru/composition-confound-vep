"""Composition- and mutation-expectedness confound features for SNV variants.

This module implements the two "zero-learning" confound feature families defined in
``notes/00_original_outline.md`` section 5 ("The confounder, formally"):

C1 -- local composition change
    For window half-widths ``w`` (default 11, 21, 51 bp) around a variant, the reference
    window ``s_ref`` and the alternate window ``s_alt`` (single center base substituted)
    differ only at the center position. We report, per ``w``:
        * ``dGC_w{w}``     = |GC(s_alt) - GC(s_ref)|
        * ``dkmer{k}_w{w}`` = L1 distance between the k-mer frequency spectra (as probability
                              distributions over all 4^k k-mers) of ``s_ref`` and ``s_alt``,
                              for k in {1, 2, 3}.

C2 -- mutational expectedness (uses a REAL external mutation-rate table, not invented numbers)
        * ``mu_5mer``       = pentanucleotide (5-mer) neutral relative mutation rate for the
                              specific substitution, from Carlson et al. 2018 ERV rates
                              (see data/mutation_rates/SOURCE.md).
        * ``cpg_ref``/``cpg_alt`` = CpG-site flags for the reference and alternate allele.
        * ``is_transition`` = 1 for purine<->purine or pyrimidine<->pyrimidine, else 0.
        * ``flank_gc_w51``  = GC content of the reference w=51 window (region-level proxy).

All features are computed on the reference (forward) strand as supplied via VCF-convention
chrom/pos/ref/alt. Every feature here is strand-invariant by construction -- GC content and
CpG status are palindromic, the Carlson table is strand-symmetric (both strands enumerated
with equal rate), transition/transversion class is symmetric, and the L1 k-mer-spectrum
distance is invariant under the reverse-complement bijection on k-mers. Hence NO strand
averaging is required; forward-strand values are the canonical unstranded features.

Efficiency: because a SNV changes only the center base, Delta_k is determined entirely by the
(2k-1)-base neighborhood of the variant plus the window length (which enters only as a
normalization constant). We exploit this closed form so each variant costs O(k) work per
(w, k) instead of rebuilding full k-mer spectra. Sequence extraction from a reference FASTA
is the only real cost and can be parallelized with ``n_jobs``.

Main entry point:
    compute_features(df, ref_fasta_path, window_sizes=(11, 21, 51)) -> pd.DataFrame
"""

from __future__ import annotations

import os
from collections import Counter
from functools import lru_cache
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

_VALID_BASES = frozenset("ACGT")
_PURINES = frozenset("AG")
_PYRIMIDINES = frozenset("CT")
_COMPLEMENT = str.maketrans("ACGTacgtN", "TGCAtgcaN")

_DEFAULT_WINDOWS = (11, 21, 51)
_KMER_SIZES = (1, 2, 3)

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MUTRATE_PATH = os.path.join(
    _MODULE_DIR, "..", "data", "mutation_rates", "carlson2018_ERV_5mer_edit.txt"
)


# --------------------------------------------------------------------------------------
# Sequence helpers
# --------------------------------------------------------------------------------------


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of an uppercase/lowercase DNA string."""
    return seq.translate(_COMPLEMENT)[::-1]


def gc_content(seq: str) -> float:
    """Fraction of G/C bases among A/C/G/T positions in ``seq`` (case-insensitive).

    Non-ACGT characters (e.g. N) are ignored in both numerator and denominator. Returns
    ``float('nan')`` for a sequence with no valid bases.
    """
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    at = seq.count("A") + seq.count("T")
    denom = gc + at
    if denom == 0:
        return float("nan")
    return gc / denom


def kmer_spectrum(seq: str, k: int) -> np.ndarray:
    """Return the k-mer frequency spectrum of ``seq`` as a probability vector of length 4**k.

    This is a brute-force reference implementation (used by the fast closed-form delta as a
    correctness oracle in the tests). k-mers containing non-ACGT characters are skipped; the
    remaining counts are normalized to sum to 1. Returns an all-zero vector if no valid k-mer.
    """
    seq = seq.upper()
    index = {b: i for i, b in enumerate("ACGT")}
    vec = np.zeros(4 ** k, dtype=np.float64)
    n = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        idx = 0
        ok = True
        for ch in kmer:
            j = index.get(ch)
            if j is None:
                ok = False
                break
            idx = idx * 4 + j
        if ok:
            vec[idx] += 1
            n += 1
    if n > 0:
        vec /= n
    return vec


def _delta_kmer(window: str, center: int, alt: str, k: int) -> float:
    """L1 distance between k-mer spectra of ``window`` (ref) and its center-substituted alt.

    Closed form: only k-mers overlapping ``center`` change, so we aggregate the count delta
    over just those affected k-mers and divide by the total number of k-mers in the window
    (both spectra share the same denominator because a substitution preserves length).

    L1 over probability distributions = (sum of |aggregated count deltas|) / n_kmers.
    """
    L = len(window)
    n_kmers = L - k + 1
    if n_kmers <= 0:
        return float("nan")
    lo = max(0, center - k + 1)
    hi = min(center, L - k)  # inclusive last valid start covering the center
    if lo > hi:
        return 0.0
    alt_window = window[:center] + alt + window[center + 1 :]
    delta: Counter[str] = Counter()
    for s in range(lo, hi + 1):
        ref_kmer = window[s : s + k]
        alt_kmer = alt_window[s : s + k]
        # Skip k-mers containing non-ACGT so the delta matches kmer_spectrum's normalization
        # only when both are valid; if either side has an invalid base treat as NaN.
        if not (_all_valid(ref_kmer) and _all_valid(alt_kmer)):
            return float("nan")
        delta[ref_kmer] -= 1
        delta[alt_kmer] += 1
    l1_counts = sum(abs(v) for v in delta.values())
    return l1_counts / n_kmers


def _all_valid(seq: str) -> bool:
    return all(ch in _VALID_BASES for ch in seq)


# --------------------------------------------------------------------------------------
# C2: mutation-rate table
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=4)
def load_mutation_rate_table(path: Optional[str] = None) -> dict:
    """Load the Carlson 2018 ERV pentanucleotide relative-mutation-rate table.

    Returns a dict keyed by ``(ref_5mer, alt_base)`` -> relative mutation rate, where
    ``ref_5mer`` is the uppercase reference pentamer (central base = variant position) and
    ``alt_base`` is the substituted center base. Both strands are present in the source file,
    so the lookup is strand-complete; a reverse-complement fallback is nonetheless applied at
    query time for robustness. See data/mutation_rates/SOURCE.md for provenance.
    """
    if path is None:
        path = _DEFAULT_MUTRATE_PATH
    table: dict[tuple[str, str], float] = {}
    df = pd.read_csv(path, sep="\t")
    for wt_motif, mt_motif, rate in zip(df["wtMotif"], df["mtMotif"], df["ERV_rel_rate"]):
        ref5 = str(wt_motif).upper()
        alt5 = str(mt_motif).upper()
        if len(ref5) != 5 or len(alt5) != 5:
            continue
        alt_base = alt5[2]
        table[(ref5, alt_base)] = float(rate)
    return table


def lookup_mutation_rate(ref_5mer: str, alt_base: str, table: dict) -> float:
    """Look up the pentanucleotide relative mutation rate for a substitution.

    ``ref_5mer`` is the 5-base reference context (center = variant); ``alt_base`` is the
    alternate center base. Tries the forward orientation, then the reverse complement.
    Returns ``float('nan')`` if the context is invalid or absent from the table.
    """
    ref_5mer = ref_5mer.upper()
    alt_base = alt_base.upper()
    if len(ref_5mer) != 5 or not _all_valid(ref_5mer) or alt_base not in _VALID_BASES:
        return float("nan")
    rate = table.get((ref_5mer, alt_base))
    if rate is not None:
        return rate
    rc_ref = reverse_complement(ref_5mer)
    rc_alt = reverse_complement(alt_base)
    return table.get((rc_ref, rc_alt), float("nan"))


# --------------------------------------------------------------------------------------
# C2: CpG and transition/transversion
# --------------------------------------------------------------------------------------


def is_transition(ref: str, alt: str) -> bool:
    """True if the substitution is a transition (purine<->purine or pyrimidine<->pyrimidine)."""
    ref, alt = ref.upper(), alt.upper()
    return (ref in _PURINES and alt in _PURINES) or (
        ref in _PYRIMIDINES and alt in _PYRIMIDINES
    )


def cpg_status(pentamer: str, center_base: str) -> bool:
    """True if ``center_base`` together with an immediate neighbor forms a CpG dinucleotide.

    ``pentamer`` is the 5-base context (index 2 = center). We substitute ``center_base`` at
    index 2 and test whether ``seq[center-1:center+1] == 'CG'`` or ``seq[center:center+2] == 'CG'``
    (the ``ref[center-1:center+1] or ref[center:center+2] == "CG"`` rule from the outline).
    """
    if len(pentamer) != 5:
        return False
    seq = pentamer[:2] + center_base.upper() + pentamer[3:]
    seq = seq.upper()
    left = seq[1:3]
    right = seq[2:4]
    return left == "CG" or right == "CG"


# --------------------------------------------------------------------------------------
# Window extraction
# --------------------------------------------------------------------------------------


class _FastaWindowFetcher:
    """Fetch reference windows centered on a variant from an indexed FASTA (pysam)."""

    def __init__(self, fasta_path: str):
        import pysam  # local import so the module imports without pysam when seqs are provided

        self.fasta_path = fasta_path
        self._fa = pysam.FastaFile(fasta_path)
        self._contigs = set(self._fa.references)

    def _resolve_contig(self, chrom: str) -> Optional[str]:
        chrom = str(chrom)
        if chrom in self._contigs:
            return chrom
        alt = chrom[3:] if chrom.startswith("chr") else "chr" + chrom
        if alt in self._contigs:
            return alt
        return None

    def fetch(self, chrom: str, pos: int, half_width: int) -> Optional[str]:
        """Fetch the [pos-half_width, pos+half_width] window (1-based, inclusive), or None.

        Returns None if the contig is unknown or the window runs off the contig end.
        """
        contig = self._resolve_contig(chrom)
        if contig is None:
            return None
        start = pos - 1 - half_width  # 0-based, inclusive
        end = pos + half_width  # 0-based, exclusive
        if start < 0:
            return None
        try:
            seq = self._fa.fetch(contig, start, end)
        except (ValueError, KeyError):
            return None
        if len(seq) != 2 * half_width + 1:
            return None
        return seq.upper()

    def close(self):
        self._fa.close()


def _center_window_from_seq(seq: str, half_width: int) -> Optional[str]:
    """Extract the centered [-half_width, +half_width] sub-window from an odd-length seq."""
    n = len(seq)
    if n % 2 == 0:
        return None
    center = n // 2
    if center - half_width < 0 or center + half_width >= n:
        return None
    return seq[center - half_width : center + half_width + 1].upper()


# --------------------------------------------------------------------------------------
# Core per-variant feature computation
# --------------------------------------------------------------------------------------


def _feature_columns(window_sizes: Sequence[int]) -> list:
    cols = []
    for w in window_sizes:
        cols.append(f"dGC_w{w}")
    for w in window_sizes:
        for k in _KMER_SIZES:
            cols.append(f"dkmer{k}_w{w}")
    cols += ["mu_5mer", "cpg_ref", "cpg_alt", "is_transition", "flank_gc_w51"]
    return cols


def _compute_one(
    max_window: Optional[str],
    ref: str,
    alt: str,
    window_sizes: Sequence[int],
    max_half: int,
    mut_table: dict,
) -> dict:
    """Compute all features for a single variant given its centered max-width ref window.

    ``max_window`` is the reference sequence of length ``2*max_half+1`` centered on the
    variant (center base == ref), or None if unavailable. Returns a dict of feature ->
    value with NaN where a feature cannot be computed, plus ``feat_ok`` and ``ref_match``.
    """
    out = {c: float("nan") for c in _feature_columns(window_sizes)}
    out["feat_ok"] = False
    out["ref_match"] = False

    ref = str(ref).upper()
    alt = str(alt).upper()

    # Only single-base substitutions with valid alleles are supported.
    if ref not in _VALID_BASES or alt not in _VALID_BASES or ref == alt:
        return out
    if max_window is None:
        return out

    center = max_half
    ref_match = max_window[center] == ref
    out["ref_match"] = ref_match

    # C1: per-window composition-change features.
    for w in window_sizes:
        sub = max_window[center - w : center + w + 1]
        c = w  # center index within the sub-window
        L = len(sub)
        # dGC: only the center base can flip GC membership.
        ref_is_gc = 1 if sub[c] in "GC" else 0
        alt_is_gc = 1 if alt in "GC" else 0
        # GC is defined over valid bases; use window length as denom (windows are all-ACGT
        # for real hg38 loci; N-containing windows yield NaN via the k-mer path anyway).
        if _all_valid(sub):
            out[f"dGC_w{w}"] = abs(alt_is_gc - ref_is_gc) / L
        for k in _KMER_SIZES:
            out[f"dkmer{k}_w{w}"] = _delta_kmer(sub, c, alt, k)

    # C2: mutational expectedness.
    pentamer = max_window[center - 2 : center + 3]
    if len(pentamer) == 5 and _all_valid(pentamer):
        out["mu_5mer"] = lookup_mutation_rate(pentamer, alt, mut_table)
        out["cpg_ref"] = int(cpg_status(pentamer, pentamer[2]))
        out["cpg_alt"] = int(cpg_status(pentamer, alt))
    out["is_transition"] = int(is_transition(ref, alt))

    # Flanking GC background: reference GC of the w=51 window (or the max window available).
    flank_w = 51 if 51 in window_sizes else max_half
    flank = max_window[center - flank_w : center + flank_w + 1]
    out["flank_gc_w51"] = gc_content(flank)

    out["feat_ok"] = ref_match and not np.isnan(out.get("mu_5mer", float("nan")))
    return out


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def compute_features(
    df: pd.DataFrame,
    ref_fasta_path: Optional[str] = None,
    window_sizes: Sequence[int] = _DEFAULT_WINDOWS,
    seq_col: Optional[str] = None,
    mutation_rate_path: Optional[str] = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute C1 + C2 confound features for a table of SNVs.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ``chrom, pos, ref, alt`` (pos is 1-based, VCF convention).
        ``ref``/``alt`` are single reference-strand bases.
    ref_fasta_path : str, optional
        Path to an indexed hg38 FASTA (``.fai`` alongside). Required unless ``seq_col`` is
        given. See ``code/get_reference.py`` for a fetch+index helper.
    window_sizes : sequence of int, default (11, 21, 51)
        Window half-widths ``w`` for the C1 features.
    seq_col : str, optional
        Name of a column holding a pre-extracted reference window per variant, centered on
        the variant (odd length, center base == ref), at least ``2*max(window_sizes)+1`` long.
        If provided, ``ref_fasta_path`` is not used.
    mutation_rate_path : str, optional
        Override path to the pentanucleotide rate table (defaults to the bundled Carlson table).
    n_jobs : int, default 1
        Parallel workers for FASTA window extraction. Ignored when ``seq_col`` is used.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with the feature columns appended (see ``_feature_columns``) plus
        ``ref_match`` (center base matched the stated ref allele) and ``feat_ok`` (all core
        features computable). Rows that cannot be featurized carry NaN feature values.
    """
    required = {"chrom", "pos", "ref", "alt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")
    if seq_col is None and ref_fasta_path is None:
        raise ValueError("Provide either ref_fasta_path or seq_col.")

    window_sizes = tuple(int(w) for w in window_sizes)
    max_half = max(window_sizes)
    mut_table = load_mutation_rate_table(mutation_rate_path)

    df = df.reset_index(drop=True)
    windows = _extract_windows(df, ref_fasta_path, seq_col, max_half, n_jobs)

    records = []
    for i in range(len(df)):
        records.append(
            _compute_one(
                windows[i],
                df.at[i, "ref"],
                df.at[i, "alt"],
                window_sizes,
                max_half,
                mut_table,
            )
        )
    feat_df = pd.DataFrame.from_records(records)
    return pd.concat([df, feat_df], axis=1)


def _extract_windows(
    df: pd.DataFrame,
    ref_fasta_path: Optional[str],
    seq_col: Optional[str],
    max_half: int,
    n_jobs: int,
) -> list:
    """Return a list of centered max-width reference windows (or None) per variant."""
    n = len(df)
    if seq_col is not None:
        if seq_col not in df.columns:
            raise ValueError(f"seq_col '{seq_col}' not in df.")
        return [
            _center_window_from_seq(str(s), max_half) if isinstance(s, str) else None
            for s in df[seq_col].tolist()
        ]

    chroms = df["chrom"].tolist()
    positions = [int(p) for p in df["pos"].tolist()]

    if n_jobs is not None and n_jobs > 1 and n > 1:
        return _extract_windows_parallel(
            ref_fasta_path, chroms, positions, max_half, n_jobs
        )

    fetcher = _FastaWindowFetcher(ref_fasta_path)
    try:
        return [fetcher.fetch(chroms[i], positions[i], max_half) for i in range(n)]
    finally:
        fetcher.close()


def _extract_windows_worker(args):
    fasta_path, chroms, positions, max_half = args
    fetcher = _FastaWindowFetcher(fasta_path)
    try:
        return [
            fetcher.fetch(chroms[i], positions[i], max_half)
            for i in range(len(chroms))
        ]
    finally:
        fetcher.close()


def _extract_windows_parallel(fasta_path, chroms, positions, max_half, n_jobs):
    import multiprocessing as mp

    n = len(chroms)
    n_jobs = min(n_jobs, n)
    bounds = np.array_split(np.arange(n), n_jobs)
    chunks = [
        (
            fasta_path,
            [chroms[i] for i in idx],
            [positions[i] for i in idx],
            max_half,
        )
        for idx in bounds
        if len(idx) > 0
    ]
    # Use a 'fork' context: workers inherit the interpreter state and only need the top-level
    # worker function, so parallelism is robust regardless of how the caller was launched
    # (avoids the 'spawn' re-import of __main__ that Python 3.14 defaults to on macOS).
    try:
        ctx = mp.get_context("fork")
    except ValueError:  # pragma: no cover - platforms without fork
        ctx = mp.get_context()
    with ctx.Pool(processes=len(chunks)) as pool:
        results = pool.map(_extract_windows_worker, chunks)
    windows = []
    for r in results:
        windows.extend(r)
    return windows
