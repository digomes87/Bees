"""Reusable chart panels drawn onto a single matplotlib axis."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

GRID_OPACITY = 0.3
FILL_OPACITY = 0.7
Values = Mapping[str, float]


@dataclass(frozen=True)
class Labels:
    title: str
    horizontal: str = ""
    vertical: str = ""

    def apply(self, axis: Axes) -> None:
        axis.set_title(self.title, fontsize=14, fontweight="bold")
        axis.set_xlabel(self.horizontal)
        axis.set_ylabel(self.vertical)
        axis.grid(True, alpha=GRID_OPACITY)


def fitted_line(horizontal: Sequence[float], vertical: Sequence[float]) -> np.ndarray:
    return np.polyval(np.polyfit(horizontal, vertical, 1), horizontal)


def trend_panel(axis: Axes, series: pd.Series, color: str, labels: Labels) -> None:
    """A yearly series with its linear trend."""
    axis.plot(series.index, series.to_numpy(), "o-", color=color, linewidth=2)
    fit = fitted_line(series.index, series.to_numpy())
    axis.plot(series.index, fit, "--", color="black", alpha=0.8, linewidth=2)
    labels.apply(axis)


def area_trend_panel(axis: Axes, series: pd.Series, color: str, labels: Labels) -> None:
    axis.fill_between(series.index, series.to_numpy(), alpha=GRID_OPACITY, color=color)
    trend_panel(axis, series, color, labels)


def scatter_panel(
    axis: Axes, horizontal: pd.Series, vertical: pd.Series, color: str, labels: Labels
) -> None:
    """Two variables against each other with the fitted regression line."""
    axis.scatter(horizontal, vertical, alpha=FILL_OPACITY, s=60, color=color)
    axis.plot(horizontal, fitted_line(horizontal, vertical), "r--", alpha=0.8)
    labels.apply(axis)


def bar_panel(
    axis: Axes, values: Values, colors: Sequence[str], labels: Labels, unit: str = ""
) -> None:
    bars = axis.bar(
        list(values), list(values.values()), color=colors, alpha=FILL_OPACITY
    )
    axis.bar_label(bars, fmt=lambda value: f"{value:g}{unit}", padding=2)
    labels.apply(axis)


def horizontal_bar_panel(
    axis: Axes, values: Values, colors: Sequence[str] | str, labels: Labels
) -> None:
    bars = axis.barh(
        list(values), list(values.values()), color=colors, alpha=FILL_OPACITY
    )
    axis.bar_label(bars, fmt="%g", padding=2)
    labels.apply(axis)


def grouped_bar_panel(
    axis: Axes, groups: Mapping[str, Values], colors: Sequence[str], labels: Labels
) -> None:
    """Side-by-side bars, one colour per group."""
    table = pd.DataFrame(groups)
    table.plot.bar(ax=axis, color=list(colors), alpha=FILL_OPACITY, rot=0, width=0.7)
    labels.apply(axis)


def lines_panel(
    axis: Axes, table: pd.DataFrame, colors: Sequence[str], labels: Labels
) -> None:
    """One line per column, sharing the table's index as the horizontal axis."""
    table.plot(ax=axis, style="o-", color=list(colors), linewidth=2)
    labels.apply(axis)
