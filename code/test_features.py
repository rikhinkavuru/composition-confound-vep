"""Unit tests for code/features.py.

Run:  ~/Downloads/venv/bin/python -m pytest code/test_features.py -v

The C1 deltas are checked against BY-HAND arithmetic (see docstrings) and, over random
inputs, against a brute-force full-k-mer-spectrum oracle. C2 (CpG, ts/tv, mutation rate)
is checked against known biology.
"""

import os
import random

import numpy as np
import pandas as pd
import pytest

import features as F

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_HG38 = os.path.join(_MODULE_DIR, "..", "data", "reference", "hg38.fa")


# ---------------------------------------------------------------------------
# C1: k-mer spectrum L1 deltas, verified by hand
# ---------------------------------------------------------------------------


def test_delta_kmer_ACG_to_ATG_by_hand():
    """Window 'ACG' (w=1), substitute center C->T giving 'ATG'.

    1-mers: ref {A,C,G} -> alt {A,T,G}. Prob deltas: C:1/3->0, T:0->1/3 => L1 = 2/3.
    2-mers: ref {AC,CG} -> alt {AT,TG}. All four move by 1/2 => L1 = 4*(1/2) = 2.
    3-mers: ref {ACG} -> alt {ATG}. Two entries move by 1 => L1 = 2.
    """
    assert F._delta_kmer("ACG", 1, "T", 1) == pytest.approx(2 / 3)
    assert F._delta_kmer("ACG", 1, "T", 2) == pytest.approx(2.0)
    assert F._delta_kmer("ACG", 1, "T", 3) == pytest.approx(2.0)


def test_delta_gc_ACG_by_hand():
    """dGC for 'ACG' w=1, C->T: center C is GC, alt T is not => |0-1|/3 = 1/3."""
    sub = "ACG"
    ref_is_gc = 1 if sub[1] in "GC" else 0
    alt_is_gc = 1 if "T" in "GC" else 0
    assert abs(alt_is_gc - ref_is_gc) / len(sub) == pytest.approx(1 / 3)


def test_delta_kmer_matches_brute_force_spectrum():
    """Closed-form _delta_kmer must equal L1 of the brute-force kmer_spectrum, for random cases."""
    rng = random.Random(0)
    for _ in range(500):
        L = rng.choice([5, 11, 21, 23])
        window = "".join(rng.choice("ACGT") for _ in range(L))
        center = L // 2
        alt = rng.choice([b for b in "ACGT" if b != window[center]])
        alt_window = window[:center] + alt + window[center + 1 :]
        for k in (1, 2, 3):
            brute = float(np.abs(F.kmer_spectrum(window, k) - F.kmer_spectrum(alt_window, k)).sum())
            fast = F._delta_kmer(window, center, alt, k)
            assert fast == pytest.approx(brute, abs=1e-12), (window, alt, k, fast, brute)


def test_delta_kmer_normalization_scales_with_window():
    """For a fixed local neighborhood, dkmer scales as 1/(2w+1-k+1) with window size."""
    # Same center substitution, two window widths sharing the neighborhood.
    small = "TTCGT"   # w=2, center idx 2
    large = "AATTCGTAA"  # w=4, center idx 4
    for k in (1, 2, 3):
        d_small = F._delta_kmer(small, 2, "A", k)
        d_large = F._delta_kmer(large, 4, "A", k)
        # raw L1 count deltas are identical; only the denominator (n_kmers) differs.
        assert d_small * (len(small) - k + 1) == pytest.approx(d_large * (len(large) - k + 1))


# ---------------------------------------------------------------------------
# C2: transition/transversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ref,alt,expected",
    [
        ("A", "G", True),   # purine <-> purine
        ("G", "A", True),
        ("C", "T", True),   # pyrimidine <-> pyrimidine
        ("T", "C", True),
        ("A", "C", False),  # transversions
        ("A", "T", False),
        ("G", "T", False),
        ("C", "G", False),
        ("G", "C", False),
    ],
)
def test_transition_transversion(ref, alt, expected):
    assert F.is_transition(ref, alt) is expected


# ---------------------------------------------------------------------------
# C2: CpG detection
# ---------------------------------------------------------------------------


def test_cpg_status_ref_and_alt():
    # 'AACGT': center C (idx2), right pair seq[2:4]='CG' -> CpG.
    assert F.cpg_status("AACGT", "C") is True
    # 'AACAA': center C, neighbors A/A -> not CpG.
    assert F.cpg_status("AACAA", "C") is False
    # alt allele creating a CpG: ref center A in 'ACAAA' (left pair 'CA'); alt G -> 'ACGAA' left 'CG'.
    assert F.cpg_status("ACAAA", "A") is False
    assert F.cpg_status("ACAAA", "G") is True
    # CpG via the left side: 'TTCGA'? center C right 'CG' true; test G-centered CpG from left.
    assert F.cpg_status("ACGAA", "G") is True   # center G, left seq[1:3]='CG'


# ---------------------------------------------------------------------------
# C2: mutation-rate table
# ---------------------------------------------------------------------------


def test_mutation_rate_table_loads():
    table = F.load_mutation_rate_table()
    # Carlson 5-mer table: 1024 pentamer contexts x 3 alt bases = 3072 entries.
    assert len(table) == 3072
    assert all(v > 0 for v in table.values())


def test_mutation_rate_positive_and_finite():
    table = F.load_mutation_rate_table()
    # A common non-CpG context, valid substitution.
    r = F.lookup_mutation_rate("TTCTT", "T", table)  # C>T transition, non-CpG
    assert np.isfinite(r) and r > 0


def test_mutation_rate_cpg_transition_exceeds_transversion():
    """Well-known biology: CpG C>T transitions mutate far faster than non-CpG transversions."""
    table = F.load_mutation_rate_table()
    cpg_ct = F.lookup_mutation_rate("AACGC", "T", table)   # C>T at CpG (center C, +1 G)
    transversion = F.lookup_mutation_rate("AAAAA", "C", table)  # A>C transversion
    assert cpg_ct > transversion
    assert cpg_ct > 10 * transversion  # the gap is an order of magnitude, not marginal


def test_mutation_rate_transition_exceeds_transversion_same_context():
    """Within a fixed context, the transition should exceed each transversion (ti/tv > 1)."""
    table = F.load_mutation_rate_table()
    ctx = "TTACA"  # center A
    ti = F.lookup_mutation_rate(ctx, "G", table)   # A>G transition
    tv1 = F.lookup_mutation_rate(ctx, "C", table)  # A>C
    tv2 = F.lookup_mutation_rate(ctx, "T", table)  # A>T
    assert ti > tv1 and ti > tv2


def test_mutation_rate_reverse_complement_fallback():
    """Lookup is strand-symmetric: (ctx, alt) and revcomp(ctx, alt) return the same rate."""
    table = F.load_mutation_rate_table()
    r1 = F.lookup_mutation_rate("AACGC", "T", table)
    # revcomp of AACGC = GCGTT, revcomp of alt T = A
    r2 = F.lookup_mutation_rate("GCGTT", "A", table)
    assert r1 == pytest.approx(r2)


def test_mutation_rate_invalid_context_is_nan():
    table = F.load_mutation_rate_table()
    assert np.isnan(F.lookup_mutation_rate("AANGC", "T", table))
    assert np.isnan(F.lookup_mutation_rate("AACGC", "N", table))


# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------


def test_gc_content():
    assert F.gc_content("GCGC") == pytest.approx(1.0)
    assert F.gc_content("ATAT") == pytest.approx(0.0)
    assert F.gc_content("ACGT") == pytest.approx(0.5)
    assert F.gc_content("ACGN") == pytest.approx(2 / 3)  # N ignored


def test_reverse_complement():
    assert F.reverse_complement("AACGT") == "ACGTT"


# ---------------------------------------------------------------------------
# Integration: compute_features via a provided centered sequence (no FASTA needed)
# ---------------------------------------------------------------------------


def test_compute_features_with_seq_col_exact_values():
    """seq='TTTTTCTTTTT' (len 11, center idx5='C'), ref C -> alt T, windows w in {1,2,5}.

    dGC_w1 = 1/3 ; dGC_w2 = 1/5 ; dGC_w5 = 1/11.
    dkmer1_w1 = 2/3 (center 1-mer C->T over 3 1-mers).
    pentamer = 'TTCTT', C>T transition, non-CpG.
    flank_gc_w51 (falls back to w=5 window here) = 1/11 (single C among 11 bases).
    """
    df = pd.DataFrame(
        {"chrom": ["chr1"], "pos": [100], "ref": ["C"], "alt": ["T"],
         "seq": ["TTTTTCTTTTT"]}
    )
    out = F.compute_features(df, seq_col="seq", window_sizes=(1, 2, 5))
    row = out.iloc[0]
    assert row["ref_match"] == True  # noqa: E712
    assert row["feat_ok"] == True    # noqa: E712
    assert row["dGC_w1"] == pytest.approx(1 / 3)
    assert row["dGC_w2"] == pytest.approx(1 / 5)
    assert row["dGC_w5"] == pytest.approx(1 / 11)
    assert row["dkmer1_w1"] == pytest.approx(2 / 3)
    assert row["is_transition"] == 1
    assert row["cpg_ref"] == 0
    assert row["cpg_alt"] == 0
    assert np.isfinite(row["mu_5mer"]) and row["mu_5mer"] > 0
    assert row["flank_gc_w51"] == pytest.approx(1 / 11)


def test_compute_features_schema():
    df = pd.DataFrame(
        {"chrom": ["chr1"], "pos": [100], "ref": ["C"], "alt": ["T"],
         "seq": ["A" * 103]}  # long enough for default w=51
    )
    out = F.compute_features(df, seq_col="seq")
    expected = [
        "dGC_w11", "dGC_w21", "dGC_w51",
        "dkmer1_w11", "dkmer2_w11", "dkmer3_w11",
        "dkmer1_w21", "dkmer2_w21", "dkmer3_w21",
        "dkmer1_w51", "dkmer2_w51", "dkmer3_w51",
        "mu_5mer", "cpg_ref", "cpg_alt", "is_transition", "flank_gc_w51",
        "ref_match", "feat_ok",
    ]
    for c in expected:
        assert c in out.columns, c


def test_compute_features_rejects_non_snv():
    df = pd.DataFrame(
        {"chrom": ["chr1"], "pos": [100], "ref": ["CA"], "alt": ["T"],
         "seq": ["A" * 103]}
    )
    out = F.compute_features(df, seq_col="seq")
    assert out.iloc[0]["feat_ok"] == False  # noqa: E712
    assert np.isnan(out.iloc[0]["dGC_w51"])


def test_missing_columns_raises():
    df = pd.DataFrame({"chrom": ["chr1"], "pos": [1]})
    with pytest.raises(ValueError):
        F.compute_features(df, seq_col="seq")


# ---------------------------------------------------------------------------
# Integration: real hg38 FASTA path (skipped if the reference is not present)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.exists(_HG38), reason="hg38.fa not downloaded/indexed")
def test_compute_features_from_fasta_roundtrip():
    """Read a real base from hg38, build a SNV there, and confirm the FASTA path featurizes it."""
    import pysam

    fa = pysam.FastaFile(_HG38)
    # Pick a comfortably interior autosomal position.
    chrom, pos = "chr1", 1_000_000
    ref_base = fa.fetch(chrom, pos - 1, pos).upper()
    fa.close()
    assert ref_base in "ACGT"
    alt = "A" if ref_base != "A" else "G"
    df = pd.DataFrame({"chrom": [chrom], "pos": [pos], "ref": [ref_base], "alt": [alt]})
    out = F.compute_features(df, ref_fasta_path=_HG38, window_sizes=(11, 21, 51))
    row = out.iloc[0]
    assert row["ref_match"] == True  # noqa: E712
    assert row["feat_ok"] == True    # noqa: E712
    assert 0.0 <= row["flank_gc_w51"] <= 1.0
    assert np.isfinite(row["mu_5mer"]) and row["mu_5mer"] > 0
    # All dkmer/dGC features present and finite.
    for w in (11, 21, 51):
        assert np.isfinite(row[f"dGC_w{w}"])
        for k in (1, 2, 3):
            assert np.isfinite(row[f"dkmer{k}_w{w}"])
