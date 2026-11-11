"""Shared chart style for all thesis figures (validated categorical palette, fixed order)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # blue, orange, aqua, yellow
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False})

def finish(fig, ax, title, path, legend=True, ncol=4, grid_axis="y"):
    n_items = len(ax.get_legend_handles_labels()[0]) if legend else 0
    rows = -(-n_items // ncol) if n_items else 0
    ax.set_title(title, loc="left", color=INK, fontsize=10.5, pad=10 + 18 * rows)
    if legend:
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, fontsize=8.5)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
