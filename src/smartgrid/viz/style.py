"""Shared chart styling.

Colour is assigned by the job it does rather than picked per chart:

* categorical — telling distinct series apart, in a fixed slot order
* sequential — magnitude, one hue from light to dark
* emphasis — one series is the point, the rest are context: slot 1 against grey

The categorical slots are ordered for colour-vision-deficiency separation
(worst adjacent pair dE 9.2, normal-vision dE 24.0 on this surface). Series are
also direct-labelled or legended, so identity never rests on hue alone.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

from smartgrid.config import PROJECT_ROOT

#: Committed, so figures render on GitHub and in the README without running the
#: notebook. Written by `save()` as each chart is drawn, so there is one source
#: for both the inline output and the file.
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

#: Fixed slot order. Never cycled past the end.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE, ORANGE, GREEN, YELLOW = CATEGORICAL

CRITICAL = "#d03b3b"

#: Single hue, light to dark, for magnitude.
SEQUENTIAL = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]


def sequential_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list("smartgrid", SEQUENTIAL)


def apply_style() -> None:
    """Set global rcParams. Call once before plotting."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": SURFACE,

        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),

        # Hairline, solid, one shade off the surface. Never dashed.
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",

        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,

        "lines.linewidth": 1.8,
        "lines.markersize": 5,

        "legend.frameon": False,
        "legend.fontsize": 9,

        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10,

        "figure.constrained_layout.use": True,
    })


def titled(ax, title: str, caption: str | None = None) -> None:
    """Title, plus a caption above the axes stating what to take from the chart.

    Both are set together because they share vertical space: the caption sits
    between the title and the plot, so the title's padding depends on it.
    """
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=24 if caption else 12)
    if caption:
        ax.text(
            0.0, 1.015, caption,
            transform=ax.transAxes,
            fontsize=9, color=INK_MUTED, ha="left", va="bottom",
        )


def save(fig, name: str) -> None:
    """Write a figure to docs/figures. Call before `plt.show()`."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")


def require_data(data, what: str):
    """Fail loudly on an empty selection.

    Matplotlib renders an empty frame as blank axes rather than raising, which is
    easy to commit without noticing.
    """
    if len(data) == 0:
        raise ValueError(f"no data for {what}")
    return data
