"""
Conceptual overview figure (Figure 1): the thesis as a clean, compact schematic. The zero-shot
gLM score (LLR) decomposes into a regulatory-function term and a mutation-expectedness /
composition term; the second is capturable with no learning. The quantitative message (recovers
~half of gLM AUROC; removed by calibration; surviving signal is motif-disrupting) lives in the
caption, keeping the diagram uncluttered.

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
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "savefig.bbox": "tight",
                     "font.family": "sans-serif"})

FUNC = "#2a9d8f"   # regulatory function
CONF = "#d1495b"   # confound / composition
INK = "#22303a"
GREY = "#7a8791"
LW = 1.3
ROUND = 0.018


def box(ax, cx, cy, w, h, text, fc, ec, tc, fs=9, weight="normal"):
    """Centered rounded box; text auto-centered with margin."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle=f"round,pad=0.004,rounding_size={ROUND}",
                 linewidth=LW, edgecolor=ec, facecolor=fc, zorder=2,
                 mutation_aspect=0.5))
    ax.text(cx, cy, text, ha="center", va="center", color=tc, fontsize=fs,
            fontweight=weight, zorder=3, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=12, linewidth=lw, color=color,
                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(7.0, 2.25))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # aligned two-column grid
    LX, RX = 0.255, 0.745      # column centres (function | confound)
    CW = 0.46                  # shared column-box width (even borders)
    R_TOP, R_MID, R_BOT = 0.855, 0.50, 0.145
    HTOP, HMID, HBOT = 0.20, 0.21, 0.21

    # top pipeline row (three boxes, consistent height)
    box(ax, 0.085, R_TOP, 0.15, HTOP, "variant\nref $\\rightarrow$ alt", "#f1f3f4", GREY, INK, 8.5)
    box(ax, 0.265, R_TOP, 0.155, HTOP, "genomic LM", INK, INK, "white", 9, "bold")
    box(ax, 0.66, R_TOP, 0.40, HTOP, "$\\mathrm{LLR}=\\log p(\\mathrm{alt})-\\log p(\\mathrm{ref})$",
        "#eef1f3", GREY, INK, 9)
    arrow(ax, 0.162, R_TOP, 0.185, R_TOP)
    arrow(ax, 0.345, R_TOP, 0.457, R_TOP)

    # decompose fan-out
    ax.text(0.5, 0.685, "decomposes into", ha="center", va="center", fontsize=8,
            color=GREY, style="italic")
    arrow(ax, 0.60, 0.75, LX, R_MID + HMID / 2, GREY, rad=0.22)
    arrow(ax, 0.72, 0.75, RX, R_MID + HMID / 2, GREY, rad=-0.15)

    # middle row: the two terms (equal boxes, aligned)
    box(ax, LX, R_MID, CW, HMID, "$\\alpha\\cdot$ regulatory function", FUNC, FUNC, "white", 9.5, "bold")
    box(ax, RX, R_MID, CW, HMID,
        "$\\beta\\cdot$ mutation-expectedness\n(5-mer rate, CpG, GC)", CONF, CONF, "white", 9, "bold")

    # arrows to outcomes
    arrow(ax, LX, R_MID - HMID / 2, LX, R_BOT + HBOT / 2, FUNC)
    arrow(ax, RX, R_MID - HMID / 2, RX, R_BOT + HBOT / 2, CONF)
    ax.text(RX + 0.015, (R_MID + R_BOT) / 2, "captured with\nno learning", ha="left", va="center",
            fontsize=7.4, color=CONF, style="italic", linespacing=1.2)

    # bottom row: outcomes (equal boxes, aligned under the terms)
    box(ax, LX, R_BOT, CW, HBOT, "surviving gLM signal:\nmotif-disrupting variants",
        "#e7f3f1", FUNC, FUNC, 8.8)
    box(ax, RX, R_BOT, CW, HBOT, "composition baseline\n(no learning, no sequence)",
        "#fdecef", CONF, CONF, 8.8)

    fig.savefig(os.path.join(FIG, "fig0_concept.pdf"))
    fig.savefig(os.path.join(FIG, "fig0_concept.png"), dpi=220)
    plt.close(fig)
    print("wrote fig0_concept")


if __name__ == "__main__":
    main()
