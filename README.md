# How Much of Genomic Language Model Variant-Effect Prediction Is Base Composition?

A composition- and mutation-rate confound audit of zero-shot genomic language model (gLM)
variant-effect prediction (VEP), with a model-agnostic post-hoc calibration. Target venue:
MLCB (PMLR).

## Summary of findings

A context-free composition/mutation-expectedness classifier (no sequence context, no model)
recovers about half of the above-chance AUROC of every leading model on unmatched noncoding
ClinVar, and about a fifth on composition-matched TraitGym. The confound bites hardest on
smaller single-sequence gLMs (Nucleotide Transformer, Caduceus, HyenaDNA), which do not
exceed the composition baseline on any benchmark. Alignment-based (GPN-MSA) and large
autoregressive (Evo2-40B) models keep signal that survives composition matching and
concentrates on TF-motif-disrupting variants. Within fixed-background saturation mutagenesis
the leakage is negligible, so the confound is an across-set ascertainment phenomenon. A
post-hoc pentanucleotide calibration halves the mutation-rate coupling of the most affected
model at negligible AUROC cost.

## Layout

```
code/        analysis + data-acquisition scripts (numbered by pipeline order)
  features.py            C1/C2 composition & mutation-expectedness features (unit-tested)
  analysis_utils.py      orientation, per-chrom CV AUROC, bootstrap, partial Spearman, DeLong, BH
  calibrate.py           released calibrate_llr() post-hoc calibration transform
  05_compute_features.py .. 14_gpn_neutral_table.py   pipeline steps
data/        raw + processed datasets (large; not versioned)
results/     e{1..6}_*.csv result tables + figures/
paper/       main.tex (PMLR jmlr class), references.bib, main.pdf
notes/       scientific outline (source of truth)
```

## Reproduce

Python env with numpy/scipy/pandas/scikit-learn/lightgbm/pysam/matplotlib/pypdf.

```bash
python code/05_compute_features.py        # C1/C2 features on all benchmarks
python code/06_build_traitgym_scored.py   # TraitGym precomputed model roster
python code/07_build_clinvar_scored.py    # ClinVar + Evo2-ClinVar scores
python code/08_gpn_msa_lookup.py          # GPN-MSA genome-wide tabix lookup
python code/10_e1_e2.py                   # E1 confound, E2 recovery
python code/11_e3_e4.py                   # E3 matched eval, E4 within-element
python code/12_e5_e6.py                   # E5 calibration, E6 surviving signal
python code/13_figures.py                 # figures
cd paper && tectonic main.tex             # compile paper
```

## Data sources (all public)

TraitGym (`songlab/TraitGym`), Evo2-ClinVar (`goodarzilab/evo2-clinvar`), genome-wide GPN-MSA
scores (`songlab/gpn-msa-hg38-scores`), ENCODE cCREs, Carlson et al. 2018 pentanucleotide
mutation rates, GRCh38. Model scores are authors' released per-variant values; no local model
inference is performed.
