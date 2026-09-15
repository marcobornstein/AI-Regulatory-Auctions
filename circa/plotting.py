"""Shared figure output for the experiment scripts."""

from pathlib import Path

import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parents[1] / "figures"
LABEL_FONT = {"fontsize": 14, "fontweight": "bold"}


def add_output_args(parser):
    parser.add_argument("--out-dir", type=Path, default=FIGURES, help="directory for the PNG (default: figures/)")
    parser.add_argument("--show", action="store_true", help="also open the figure in a window")


def finish(fig, ax, path, show=False, **legend_kw):
    """Add grid and legend, save to path, and optionally display."""
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14, **legend_kw)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    print(f"saved {path}")
    if show:
        plt.show()
    plt.close(fig)
