"""
Conceptual overview figure (Figure 1): the thesis as a schematic. Zero-shot gLM VEP score
(LLR) decomposes into a regulatory-function term and a mutation-expectedness/composition term;
the second is capturable with no learning and inflates unmatched leaderboards; a
pentanucleotide calibration removes it and the surviving signal is motif-disrupting.

Output: results/figures/fig0_concept.pdf (+ .png)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "savefig.bbox": "tight"})

FUNC = "#2a9d8f"    # regulatory function (teal)
CONF = "#d1495b"    # confound / composition (red)
CAL = "#2e86ab"     # calibration (blue)
INK = "#22303a"
GREY = "#6b7785"


def box(ax, x, y, w, h, text, fc, ec, tc="white", fs=9, weight="normal", round=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.006,rounding_size={round}",
                 linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight=weight, zorder=3, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=INK, style="-|>", lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=13, linewidth=lw, color=color,
                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(7.2, 2.75))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # --- top row: the score ---
    box(ax, 0.005, 0.66, 0.14, 0.24, "Variant\nref $\\rightarrow$ alt", "#f2f3f5", GREY, INK, 8.5)
    box(ax, 0.185, 0.66, 0.15, 0.24, "genomic\nLM", INK, INK, "white", 9, "bold")
    arrow(ax, 0.145, 0.78, 0.185, 0.78)
    box(ax, 0.37, 0.66, 0.30, 0.24,
        "$\\mathrm{LLR}=\\log p(\\mathrm{alt})-\\log p(\\mathrm{ref})$",
        "#eef2f4", GREY, INK, 9)
    arrow(ax, 0.335, 0.78, 0.37, 0.78)

    # decomposition
    ax.text(0.52, 0.60, "decomposes into", ha="center", va="center", fontsize=8,
            color=GREY, style="italic")
    arrow(ax, 0.45, 0.655, 0.30, 0.50, GREY, rad=0.2)
    arrow(ax, 0.59, 0.655, 0.72, 0.50, GREY, rad=-0.2)

    # --- middle row: two terms ---
    box(ax, 0.11, 0.30, 0.30, 0.20,
        "$\\alpha\\cdot$ regulatory function\n(what a gLM should learn)", FUNC, FUNC, "white", 8.5)
    box(ax, 0.55, 0.30, 0.34, 0.20,
        "$\\beta\\cdot$ mutation-expectedness\n5-mer rate, CpG, GC composition", CONF, CONF, "white", 8.5)

    # confound path down from beta term
    arrow(ax, 0.72, 0.295, 0.72, 0.20, CONF)
    box(ax, 0.55, 0.005, 0.34, 0.18,
        "composition baseline\n(no learning, no sequence)", "#fdeef1", CONF, CONF, 8.5)
    ax.text(0.72, 0.205, "captured with no learning", ha="center", va="bottom",
            fontsize=7.3, color=CONF, style="italic")

    # function path down
    arrow(ax, 0.26, 0.295, 0.26, 0.20, FUNC)
    box(ax, 0.10, 0.005, 0.32, 0.18,
        "surviving gLM signal:\nmotif-disrupting variants", "#e8f4f2", FUNC, FUNC, 8.5)

    # outcome callout in the empty top-right corner
    ax.text(0.995, 0.78, "composition alone recovers\n~half the gLM AUROC on\n"
            "unmatched benchmarks; a\npentanucleotide calibration\nremoves it (released)",
            ha="right", va="center", fontsize=7.6, color=INK, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.45", fc="#f7f8f9", ec=CAL, lw=1.1))

    save(fig, "fig0_concept")


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


if __name__ == "__main__":
    main()
