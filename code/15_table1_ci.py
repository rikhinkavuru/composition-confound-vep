"""Regenerate Table 1 rows with per-chromosome stratified-bootstrap 95% CIs on every AUROC
(composition baseline + each model), matching the per-chrom point estimator. Emits LaTeX."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_utils as A  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SPECS = [
    ("ClinVar (unmatched)", "data/clinvar/clinvar_scored.parquet", "path",
     [("gpn_msa_full", "GPN-MSA"), ("evo2_7b_llr", "Evo2-7B"), ("evo2_40b_llr", "Evo2-40B"),
      ("cadd_csv", "CADD"), ("phylop_100way", "phyloP")]),
    ("TraitGym Mendelian (matched)", "data/traitgym/mendelian_scored.parquet", "bool",
     [("gpn_msa_llr", "GPN-MSA"), ("evo2_7b_llr", "Evo2-7B"), ("evo2_40b_llr", "Evo2-40B"),
      ("nt_llr", "NT-2.5B"), ("caduceus_llr", "Caduceus"), ("cadd_rawscore", "CADD"),
      ("phylop_100v", "phyloP")]),
    ("TraitGym complex (matched)", "data/traitgym/complex_scored.parquet", "bool",
     [("gpn_msa_llr", "GPN-MSA"), ("evo2_40b_llr", "Evo2-40B"), ("nt_llr", "NT-2.5B"),
      ("caduceus_llr", "Caduceus"), ("cadd_rawscore", "CADD"), ("phylop_100v", "phyloP")]),
    ("GLRB eQTL", "data/glrb_eqtl/test_scored.parquet", "bool",
     [("gpn_msa_llr", "GPN-MSA")]),
]


def ci(lo, hi):
    return f"[{lo:.3f},{hi:.3f}]"


def main():
    out = []
    for dsname, path, labkind, models in SPECS:
        d = pd.read_parquet(os.path.join(ROOT, path))
        d["label_bin"] = ((d["label"] == "Pathogenic").astype(int) if labkind == "path"
                          else d["label"].astype(int))
        d["chrom_norm"] = d["chrom"].astype(str).str.replace("chr", "", regex=False)
        comp = A.composition_cv(d, "label_bin", "chrom_norm", return_oof=True)
        dco = d.assign(_oof=comp["oof"])
        cci = A.per_chrom_auc_ci(dco, "_oof", "label_bin", "chrom_norm", orient_score=False)
        out.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{dsname}}}}}\\\\")
        out.append(f"\\quad Composition-only & {comp['auroc']:.3f} {ci(cci['lo'], cci['hi'])} "
                   f"& {comp['auprc']:.3f} & -- \\\\")
        for col, pretty in models:
            if col not in d.columns or d[col].notna().sum() < 100:
                continue
            r = A.per_chrom_auc_ci(d, col, "label_bin", "chrom_norm")
            m = A.per_chrom_auc(d, col, "label_bin", "chrom_norm")
            rr = A.recovery_ratio(comp["auroc"], m["auroc"])
            rrs = f"{rr:.2f}" if 0 < rr < 1.5 else "$>$1"
            out.append(f"\\quad {pretty} & {r['auroc']:.3f} {ci(r['lo'], r['hi'])} "
                       f"& {m['auprc']:.3f} & {rrs} \\\\")
        out.append("\\addlinespace")
    print("\n".join(out))


if __name__ == "__main__":
    main()
