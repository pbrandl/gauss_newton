"""Shared matplotlib styling for the example figures, with light and dark themes.

The colors are validated for colorblind safety and contrast against GitHub's light
(``#ffffff``) and dark (``#0d1117``) page backgrounds. Figures are saved with a
transparent background so they blend into either page.
"""

from dataclasses import dataclass

import matplotlib as mpl
from matplotlib.axes import Axes


@dataclass(frozen=True)
class Theme:
    surface: str
    text_primary: str
    text_secondary: str
    text_muted: str
    grid: str
    axis: str
    sample: str
    series: tuple[str, ...]
    """Categorical colors, assigned in this fixed order."""
    ramp: tuple[str, ...]
    """Ordinal blue ramp from early (least emphasis) to late (most emphasis)."""


THEMES = {
    "light": Theme(
        surface="#ffffff",
        text_primary="#0b0b0b",
        text_secondary="#52514e",
        text_muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        sample="#898781",
        series=("#2a78d6", "#eb6834"),
        ramp=("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"),
    ),
    "dark": Theme(
        surface="#0d1117",
        text_primary="#f0efec",
        text_secondary="#c3c2b7",
        text_muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        sample="#898781",
        series=("#3987e5", "#d95926"),
        ramp=("#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"),
    ),
}

LINE_WIDTH = 1.75
MARKER_SIZE = 7.0
RING_WIDTH = 1.5


def apply_theme(theme: Theme) -> None:
    """Set matplotlib defaults for ``theme``."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": theme.text_primary,
            "axes.labelcolor": theme.text_muted,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlecolor": theme.text_primary,
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.facecolor": "none",
            "figure.facecolor": "none",
            "savefig.facecolor": "none",
            "xtick.color": theme.axis,
            "ytick.color": theme.axis,
            "xtick.labelcolor": theme.text_muted,
            "ytick.labelcolor": theme.text_muted,
            "legend.frameon": False,
            "legend.labelcolor": theme.text_secondary,
            "legend.fontsize": 9,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "svg.hashsalt": "gauss-newton",
        }
    )


def style_axes(ax: Axes, theme: Theme) -> None:
    """Hairline horizontal grid, baseline only, no tick marks on the value axis."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme.axis)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="y", color=theme.grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=4, width=0.8)
