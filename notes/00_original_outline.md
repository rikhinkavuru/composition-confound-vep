# How Much of Genomic Language Model Variant Effect Prediction Is Base Composition?

**A composition- and mutation-rate confound audit of zero-shot gLM VEP, with a model-agnostic calibration fix**

Target venue: MLCB (Machine Learning in Computational Biology), PMLR. Format: ~10–15 pp, single focused contribution, critique + constructive fix + regime characterization.

---

## 0. One-sentence thesis

A substantial, quantifiable fraction of the headline zero-shot variant-effect-prediction (VEP) performance of leading genomic language models (gLMs) is attributable to a **base-composition / mutational-expectedness confound** rather than learned regulatory function; after a model-agnostic pentanucleotide + composition calibration and composition-matched evaluation, gLM advantage over trivial composition baselines and one-hot encodings collapses everywhere except a characterizable subset of composition-neutral, motif-disrupting variants.

Every outcome is publishable: strong confound → "leaderboards measure composition"; weak confound → "gLM VEP is more robust to composition confounding than synthetic-sequence critiques imply." No dead-end.

---

## 1. Motivation

Three convergent results from the last ~6 months establish that gLMs frequently exploit statistical shortcuts rather than regulatory mechanism:

- **Positional grammar failure (synthetic).** The Mechanistic Invariance Test showed gLM compositional sensitivity is driven almost entirely by AT-content correlation (r = 0.78–0.96), not positional logic; a 100-parameter position-aware PWM beats billion-parameter gLMs; scale amplifies the compositional bias. But this is on *designed synthetic promoter classes*, not real VEP.
- **Region-heterogeneity inflation (clinical).** Stratifying ClinVar by region type collapses Evo2's aggregate noncoding AUROC (~0.975) to ~0.6–0.9 within specific noncoding categories; aggregate scores are inflated because pathogenic variants are disproportionately concentrated in easy region types. This is a *region-type* confound.
- **Context-artifact failure (mechanistic).** Cyclic permutation of flanking sequence collapses Evo2 tRNA-pathogenicity sensitivity from 65.8% → 5.1%, i.e. predictions are driven by irrelevant genomic context.

**The gap.** No one has isolated and quantified the **local base-composition / mutation-type axis** as a confound on *real clinical and functional* VEP benchmarks, across models, with a constructive fix. This axis is distinct from region type and from positional grammar, and it is mechanistically motivated:

- gLM VEP scores are log-likelihood ratios, LLR = log p(alt) − log p(ref); a model's likelihood partly encodes **how mutationally expected** a substitution is given local context, independent of function.
- Mutational expectedness is tightly coupled to composition: CpG sites mutate 10–50× faster; transition/transversion ratio is a sigmoidal function of local CpG content; GC content and mutation subtypes exhibit Simpson's-paradox relationships. So "composition change" and "mutation propensity" are entangled and both leak into LLR.
- The confound has a **label pathway**: pathogenic/causal variants are ascertained and distributed non-randomly across mutation types and composition (which is exactly why CADD and GPN-MSA use gnomAD *common* variants as controls rather than ClinVar benign — an explicit ascertainment-bias mitigation). So LLR's mutation-expectedness component correlates with labels for reasons unrelated to regulatory understanding.

**Direct precedent for the fix.** GPN-Star reported raw scores correlate with Roulette mutation-rate estimates at ρ = 0.31–0.34, and introduced a pentanucleotide-context neutral-score calibration that reduced this to negligible and improved most downstream benchmarks. This was applied to a **single** model as a training-adjacent step. Generalizing it into a **model-agnostic, post-hoc, benchmark-wide** audit is the constructive contribution.

---

## 2. Why this wins at MLCB specifically

- **Genre match.** MLCB 2025 (PMLR v311) accepted "Representation Learning Methods for Single-Cell Microscopy are Confounded by Background Cells" — structurally identical (hidden confound inflates a model class; measure it; fix evaluation), different modality. Also accepted multiple gLM-evaluation and interpretability papers (zero-shot promoter indel gLM; SAEs for gene-expression models; DNABERT multimodal augmentation).
- **Editor agenda alignment.** Co-editor Peter Koo's group published that pretrained gLM representations give little-to-no advantage over one-hot encoding on cell-type-specific regulatory tasks; this proposal sits inside the reviewer taste that agenda shapes.
- **Right scale.** No foundation-model training; single falsifiable claim; rigorous stratified evaluation; clinical stakes (variant interpretation). Fits the 10–30pp single-contribution profile the venue accepts.

---

## 3. Central hypotheses (falsifiable)

- **H1 (confound presence).** Each gLM's zero-shot LLR correlates with a context-free composition/mutation-expectedness score, at genome-wide neutral sites and within benchmark variant sets, with |partial ρ| ≳ 0.2–0.35.
- **H2 (baseline recovery).** A context-free composition-only classifier recovers a large fraction (pre-registered threshold, e.g. ≥ 50%) of each gLM's benchmark AUROC/AUPRC on aggregate benchmarks; the fraction is larger on region-heterogeneous benchmarks (ClinVar all-noncoding) than on matched benchmarks (TraitGym).
- **H3 (matching collapse).** Under composition- and pentanucleotide-matched evaluation, gLM advantage over the composition baseline and over one-hot shrinks materially and cross-model rankings reorder.
- **H4 (calibration generalizes).** Post-hoc pentanucleotide+composition calibration applied to every model drives LLR–mutation-rate correlation to ≈0, changes uncalibrated-leaderboard rankings, and has small effect on already-matched benchmarks (evidence the aggregate leaderboard was measuring the confound).
- **H5 (surviving signal).** Genuine gLM advantage (post-calibration, post-matching) concentrates in composition-neutral, motif-disrupting variants — characterizable by TF-motif/footprint overlap and splice signals — not uniformly across the genome.

---

## 4. Positioning vs concurrent work (differentiation)

| Work | Confound / claim | Data | What this paper adds |
|---|---|---|---|
| Mechanistic Invariance Test | positional grammar ≈ AT content | synthetic promoter classes | real clinical + quantitative-MPRA VEP; a fix; surviving-signal map |
| Genomic-heterogeneity inflation | **region-type** stratification collapses AUROC | ClinVar by region | orthogonal **composition/mutation-type** axis, finer than region; calibration fix |
| GPN-Star calibration | scores track mutation rate; pentanucleotide fix | one model, training-side | **cross-model, post-hoc** audit; show it reorders leaderboards |
| TraitGym | matched causal-variant benchmark | Mendelian+complex, matched chr/consequence/TSS(/MAF/LD) | show matching still **leaks composition**; propose composition-matched extension |
| Blind-Spots / cyclic permutation | context-artifact, codon/tRNA | mtDNA, tRNA | complementary; add composition axis + constructive eval protocol |
| Koo representational-power | gLM reps ≈ one-hot | regulatory tasks (fine-tuned) | zero-shot VEP variant-level decomposition + calibration |

The clean story: *region-type inflation is known; composition/mutation-type inflation is not; it survives even careful matching; here is a model-agnostic fix and a map of where real signal lives.*

---

## 5. The confounder, formally

Two entangled but separable axes, both computed with **zero learning**:

**C1 — local composition change.** For a variant at position p with window half-width w, let s_ref, s_alt be the ±w sequences. Define Δ_k = L1 distance between the k-mer frequency spectra of s_ref and s_alt, for k ∈ {1,2,3}. Special case ΔGC = |GC(s_alt) − GC(s_ref)|. Sweep w ∈ {11, 21, 51} bp (a single SNV shifts local composition meaningfully only in short windows, matching the effective receptive field for a point substitution).

**C2 — mutational expectedness.** Features derived from the substitution and its context, independent of function:
- pentanucleotide (5-mer) neutral mutation rate for the specific substitution (from Roulette or ERV 7-mer/5-mer tables),
- CpG status of the site (ref and alt), transition/transversion class,
- flanking GC background (element/region GC).

**Mechanism model.** LLR ≈ α·(selection/function) + β·(mutational expectedness) + noise. The confound is the β term: it correlates with labels via ascertainment, not regulatory understanding. Matching and calibration target β; the residual α is what a gLM should actually contribute.

---

## 6. Datasets (all public)

1. **ClinVar noncoding SNVs** (clinical, region-heterogeneous). P/LP vs B/LB SNVs, ≥2-star review, MANE/Ensembl+ENCODE-cCRE region annotation. ~10^5 SNVs. Use both ClinVar-benign controls and gnomAD-common controls (contrast ascertainment regimes). This is the benchmark where the confound should be **largest**.
2. **TraitGym** (causal regulatory variants, matched). Mendelian (113 traits) + complex (83 traits); PIP>0.9 positives, PIP<0.01 controls; existing matching = chr, consequence, distance-to-TSS (+MAF+LD for complex); AUPRC metric, per-chromosome CV. Use as the "already-controlled" benchmark to test whether composition still leaks past their matching.
3. **Kircher satMutMPRA** (quantitative, background-controlled). 21 disease-associated elements (10 promoters, 11 enhancers incl. UC88), ~30k SNVs/dels, GC 28–73%, length 187–601 bp; per-variant label = log2 expression effect (continuous) + p-value. **Within-element saturation is the key asset**: fixed background composition and fixed local context up to the substitution, so the composition axis can be isolated cleanly. Caveat to handle: error-prone PCR construction is itself substitution-biased — note and control for it.
4. **GTEx causal eQTL** (regulatory, matched negatives). SuSiE PIP>0.9 positive / <0.01 negative, matched negatives per Avsec/Enformer + Genomics Long-Range Benchmark; note zero-shot DNA LMs hover near AUROC 0.50 and are beaten by CADD here — useful stress test where there may be *little* signal to confound.

Optional extension: Manzo et al. MPRA/raQTL/eQTL compendium (54,859 SNPs, 4 cell lines) for cross-assay replication.

---

## 7. Models & baselines

**gLMs (zero-shot LLR = log p(alt) − log p(ref), strand-averaged):**
- **GPN-MSA** — precomputed genome-wide scores for all ~9B SNVs (HF `songlab/gpn-msa-hg38-scores`). **Pure CPU lookup.**
- **Evo2** (evo2_7b_base, 8192 bp, `score_sequences`) — precomputed ClinVar + satMut score sets already public (e.g. `jang1563/evo2-*` incl. TERT MPRA); otherwise single-GPU inference on the ~30k satMutMPRA + benchmark SNVs.
- **Nucleotide Transformer** (v2, cosine-similarity or LLR variant), **Caduceus**, **DNABERT-2** — small enough for single-GPU / Colab.
- (Optional) **PhyloGPN / GPN-Star** to test calibration on a model that already ships it.

**Baselines (the crux):**
- **Composition-only** classifier: logistic regression + gradient-boosted trees on C1+C2 features only — no sequence context, no embedding.
- **One-hot + small CNN** (matched capacity) — the "does pretraining help at all" control (Koo-style).
- **Conservation**: PhyloP, PhastCons (per-position, cheap, strong noncoding baseline).
- **CADD** (integrative) — the reference the Genomics Long-Range Benchmark shows beats zero-shot DNA LMs.

---

## 8. Experiments

Each experiment states the claim, procedure, metric, and what would falsify it.

**E1 — Confound characterization.**
Claim: LLR encodes C1/C2. Procedure: at genome-wide neutral sites (ancestral-repeat, PhyloP∈[−0.05,0.05], PhastCons=0) and within each benchmark, compute Spearman(LLR, C1) and partial correlations controlling for region/consequence. Metric: ρ, partial ρ per model. Falsify: |ρ| ≈ 0 for all models (then the confound is absent — itself a finding).

**E2 — Trivial-baseline recovery.**
Claim: composition-only recovers most gLM benchmark performance. Procedure: train composition-only classifier (C1+C2), evaluate on each benchmark under identical CV; compute recovery ratio = AUROC(composition-only − 0.5) / (AUROC(gLM) − 0.5). Metric: AUROC, AUPRC, recovery ratio; per benchmark. Expect recovery high on ClinVar-all-noncoding, lower on TraitGym. Falsify: composition-only ≈ chance while gLM strong.

**E3 — Stratified + matched evaluation.**
(a) Bin variants by C1/C2; plot AUROC vs composition-delta bin (the collapse curve). (b) Build composition- and pentanucleotide-matched case-control sets (extend TraitGym matching with 5-mer context + ΔGC via nearest-neighbor / propensity matching); re-evaluate and re-rank all models. Metric: matched AUROC/AUPRC, ranking changes (Kendall τ pre/post). Falsify: rankings and gaps stable under matching.

**E4 — Within-element clean test (satMutMPRA).**
Claim: raw model–assay correlation is partly substitution-type structure. Procedure: within each of the 21 elements (fixed background), regress model LLR on measured log2 effect; report Spearman and **partial** Spearman controlling for substitution type + Δcomposition + PCR-bias indicator. Metric: raw vs partial ρ, per element, pooled with element random effects. Expect partial ρ < raw ρ; residual = true functional signal. Falsify: raw ≈ partial (no composition leakage).

**E5 — Calibration generalization (constructive core).**
Claim: GPN-Star's pentanucleotide neutral-score calibration generalizes as a post-hoc, model-agnostic transform. Procedure: for each model, estimate LLR_neutral per 5-mer context at neutral sites; define calibrated score = LLR − LLR_neutral(context); (optionally residualize on C1 too). Re-run E1–E3. Metrics: (i) post-calibration LLR–mutation-rate ρ → ≈0 for all models; (ii) leaderboard reordering on aggregate benchmarks; (iii) small delta on already-matched benchmarks. Deliverable: released `calibrate_llr()` + per-context tables. Falsify: calibration changes nothing (confound wasn't driving aggregate scores).

**E6 — Surviving-signal characterization (the positive result).**
Claim: real gLM value is localized. Procedure: on matched+calibrated evaluation, find variants where gLM beats composition baseline; test enrichment for TF-motif disruption (PWM/FIMO or footprint overlap), splice-site proximity, and composition-neutral status; contrast with the baseline-solved set. Metric: odds ratios / enrichment, ROC within strata. Output: "gLMs add value specifically on composition-neutral, motif-disrupting variants."

**E7 — Mechanistic probe (optional depth).**
Claim: models over-encode composition. Procedure: linear-probe GC content and 5-mer mutation rate from frozen embeddings (GPN-MSA last layer, NT), reference vs alt. Metric: probe R². High R² ties the black-box confound to the entropy/Fisher-information story (gLMs concentrate information in embedding layers on DNA). Falsify: composition not linearly decodable.

---

## 9. Statistical methodology

- **Metrics:** report AUROC **and** AUPRC (imbalance per GPN-MSA/TraitGym convention); for satMutMPRA use Spearman/partial-Spearman on continuous effects.
- **Cross-validation:** per-chromosome CV, sample-size-weighted average across chromosomes, bootstrap SEs (TraitGym protocol).
- **Confound control:** partial Spearman / point-biserial; matching via nearest-neighbor or propensity score on (chr, consequence, distance-to-TSS, 5-mer context, ΔGC).
- **Significance of AUROC gaps:** DeLong or paired bootstrap; multiple-comparison control across models×benchmarks (Benjamini–Hochberg).
- **Pre-registration:** fix the recovery-ratio threshold, window w, and matching covariates before running, to preempt "p-hacked confound" criticism.

---

## 10. Figures (paper spine)

1. LLR vs composition-delta scatter per model, with ρ (E1).
2. AUROC vs composition-delta bin — the collapse curve, per benchmark (E3a).
3. Composition-only baseline vs each gLM: aggregate vs matched (E2/E3b), recovery ratios.
4. Pre- vs post-matching model rankings (bump chart) (E3b).
5. satMutMPRA within-element raw vs partial correlation (E4).
6. Calibration effect: LLR–mutation-rate ρ before/after + leaderboard reordering (E5).
7. Surviving-signal enrichment: motif-disruption / splice ORs for gLM-wins vs baseline-wins (E6).

---

## 11. Compute & feasibility

CPU-dominant. GPN-MSA is a precomputed lookup; Evo2 scores for the target sets are largely pre-released; NT/Caduceus/DNABERT-2 run on a single GPU or free Colab for the ~10^5 benchmark SNVs + ~30k MPRA variants. All composition/mutation-rate features, matching, baselines, calibration, and stats are pure CPU. No model training beyond small baselines. End-to-end reproducible on a laptop + occasional GPU.

---

## 12. Risks & mitigations

- **"Composition and function are genuinely correlated — you're discarding real biology."** Central subtlety; address head-on. The point is not that composition is non-biological, but that a model advertised as understanding regulatory *grammar* should not be reducible to a composition/mutation-rate detector; composition-neutral variants are precisely where the grammar claim is testable. Frame matching/calibration as a **diagnostic**, not a replacement metric, and always report surviving signal (E6).
- **"This is just the MIT / heterogeneity / GPN-Star paper."** Differentiation table (§4): real clinical+quantitative VEP (not synthetic), composition axis (not region), cross-model post-hoc audit (not single-model training trick), constructive matched benchmark + fix.
- **"Descriptive only."** E5 (drop-in calibration) + E6 (regime map) + E7 (mechanism) make it constructive and mechanistic.
- **Weak-confound outcome.** Still publishable as a robustness result (see thesis). Pre-registration protects interpretation.
- **PCR-bias artifact in satMutMPRA.** Include the construction bias as an explicit covariate; treat satMutMPRA as one of four datasets, not the sole basis.
- **Strand / window conventions differ across models.** Standardize LLR (strand-averaged), sweep w, report sensitivity.

---

## 13. Anticipated reviewer objections → responses

- *Ascertainment already handled by gnomAD-common controls.* We test both control regimes and show composition leaks in both; ascertainment (frequency) and composition/mutation-type are distinct confounds, and gnomAD-common controls do not match 5-mer context or ΔGC.
- *Zero-shot LLR is a weak use of gLMs; fine-tuning is the real test.* Zero-shot LLR is the deployed clinical setting and the headline claim in every gLM VEP paper (GPN-MSA, Evo2); we scope to it explicitly and note fine-tuning as future work. We include a one-hot CNN to bound the "does pretraining help" question.
- *Matching throws away data / changes the estimand.* We report both stratified (all data, binned) and matched analyses; the estimand shift is the point (confound-free comparison).

---

## 14. Contributions (what the paper claims to deliver)

1. First variant-level decomposition of zero-shot gLM VEP into functional vs composition/mutation-expectedness components across models and four benchmark families.
2. Evidence that a context-free composition baseline recovers a large fraction of aggregate gLM VEP performance, and that even carefully matched benchmarks (TraitGym) leak this confound.
3. A model-agnostic, post-hoc pentanucleotide+composition calibration (generalizing GPN-Star) shown to reorder leaderboards — a drop-in evaluation standard with released code.
4. A composition-matched evaluation protocol extending existing matching schemes.
5. A characterization of where gLMs genuinely add value (composition-neutral, motif-disrupting variants), turning a critique into a usage guideline.

---

## 15. Stretch extensions (not required for acceptance)

- Repeat for **indels** (Evo2/GPN handle these; composition change is larger and cleaner).
- Cross-species: does the confound shrink for GPN-MSA (alignment-based) vs single-sequence Evo2/NT? Prediction: alignment/conservation models are less composition-confounded.
- Tie calibrated scores back to **rare-variant burden testing** to show a downstream benefit of decontaminating the score.

---

## 16. Key references (for the related-work section)

- Mechanistic Invariance Test — gLMs fail positional regulatory logic (OpenReview, 2026).
- Genomic heterogeneity inflates variant pathogenicity predictions (bioRxiv 2025.09.05.674459).
- GPN-MSA — Benegas et al., *Nat. Biotechnol.* 2025; precomputed genome-wide scores.
- GPN-Star / phylogeny-informed GPN — pentanucleotide mutation-rate calibration (PMC12458161).
- TraitGym — Benegas, Eraslan, Song, *Benchmarking DNA Sequence Models for Causal Regulatory Variant Prediction* (bioRxiv 2025.02.11.637758).
- Evo2 — Brixi et al. / Arc Institute, *Nature* 2026 (genome modelling across domains of life).
- Kircher et al., *Saturation mutagenesis of twenty disease-associated regulatory elements*, *Nat. Commun.* 2019.
- Genomics Long-Range Benchmark — zero-shot DNA LMs vs CADD on causal eQTL/ClinVar/OMIM.
- Blind Spots in Evo2 VEP — cyclic-permutation context artifact (bioRxiv 2026.03).
- Koo et al., *Evaluating the representational power of pre-trained DNA language models for regulatory genomics*, *Genome Biol.* 2025.
- Entropy/Fisher limits of genomic FMs (arXiv 2604.04287).
- CpG/GC ↔ mutation-spectrum coupling (Simpson's paradox in rare-variant GC trends; ts/tv sigmoidal in CpG content).
