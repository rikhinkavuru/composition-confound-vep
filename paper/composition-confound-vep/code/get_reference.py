"""Fetch and index the hg38 reference genome for feature extraction.

Canonical source (chosen for stability and standard 'chr'-prefixed contig names):
    UCSC hg38 : https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz  (~1 GB gzip)

pysam's FASTA index (``.fai``) requires an uncompressed FASTA or a bgzf-compressed one; the
UCSC file is plain gzip, so we decompress to ``hg38.fa`` and build ``hg38.fa.fai``. The
uncompressed genome is ~3.2 GB. Run once:

    python code/get_reference.py

Then pass ``data/reference/hg38.fa`` as ``ref_fasta_path`` to ``features.compute_features``.
"""

from __future__ import annotations

import gzip
import os
import shutil
import sys
import urllib.request

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_REF_DIR = os.path.join(_MODULE_DIR, "..", "data", "reference")
UCSC_HG38_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"


def ensure_reference(ref_dir: str = _REF_DIR, url: str = UCSC_HG38_URL) -> str:
    """Ensure an indexed hg38 FASTA exists under ``ref_dir``; return the path to ``hg38.fa``.

    Downloads the gzip if absent, decompresses it, and builds the ``.fai`` index. Idempotent:
    skips any step whose output already exists.
    """
    import pysam

    os.makedirs(ref_dir, exist_ok=True)
    gz_path = os.path.join(ref_dir, "hg38.fa.gz")
    fa_path = os.path.join(ref_dir, "hg38.fa")
    fai_path = fa_path + ".fai"

    if not os.path.exists(fa_path):
        if not os.path.exists(gz_path):
            print(f"Downloading {url} -> {gz_path} ...", file=sys.stderr)
            urllib.request.urlretrieve(url, gz_path)
        print(f"Decompressing {gz_path} -> {fa_path} ...", file=sys.stderr)
        with gzip.open(gz_path, "rb") as fin, open(fa_path, "wb") as fout:
            shutil.copyfileobj(fin, fout, length=1 << 24)

    if not os.path.exists(fai_path):
        print(f"Indexing {fa_path} ...", file=sys.stderr)
        pysam.faidx(fa_path)

    print(f"Reference ready: {fa_path}", file=sys.stderr)
    return fa_path


if __name__ == "__main__":
    ensure_reference()
