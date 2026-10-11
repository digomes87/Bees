"""Charts of the continental series and of the projected range shift."""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bee_migration.charts.panels import (
    Labels,
    area_trend_panel,
    lines_panel,
    scatter_panel,
    trend_panel,
)

PROJECTION_HORIZON = 2030
IMPACTED = ["temperature", "species_count", "honey_production"]
IMPACT_NAMES = {
    "temperature": "Temperature",
    "latitude": "Southward Migration",
    "species_count": "Diversity",
    "honey_production": "Honey Production",
}
IMPACT_COLORS = {
    "Temperature": "red",
    "Southward Migration": "blue",
    "Diversity": "green",
    "Honey Production": "orange",
}
SCATTERS = (
    ("latitude", "purple", Labels("Temperature vs Bee Latitude", "", "Latitude (°)")),
    (
        "species_count",
        "green",
        Labels("Temperature vs Species Diversity", "", "Number of Species"),
    ),
    (
        "honey_production",
        "orange",
        Labels("Temperature vs Honey Production", "", "Honey Production (kg/hive)"),
    ),
)
RANGE_COLORMAPS = {"Current (2020-2023)": "Reds", "Projected (2040-2050)": "Blues"}


def temperature_migration_trend(series: pd.DataFrame) -> Figure:
    figure = Figure(figsize=(14, 10))
    warming, migration = figure.subplots(2, 1)
    trend_panel(
        warming,
        series["temperature"],
        "red",
        Labels(
            "Annual Average Temperature Increase in South America",
            "Year",
            "Temperature (°C)",
        ),
    )
    trend_panel(
        migration,
        series["latitude"],
        "blue",
        Labels("Southward Bee Migration (Average Latitude)", "Year", "Latitude (°)"),
    )
    return figure


def correlation_analysis(series: pd.DataFrame) -> Figure:
    figure = Figure(figsize=(16, 12))
    matrix, *scatters = figure.subplots(2, 2).flat
    sns.heatmap(
        series.corr(), annot=True, cmap="RdYlBu_r", center=0, square=True, ax=matrix
    )
    matrix.set_title("Correlation Matrix Between Variables", fontweight="bold")
    for axis, (column, color, labels) in zip(scatters, SCATTERS, strict=True):
        described = Labels(labels.title, "Temperature (°C)", labels.vertical)
        scatter_panel(axis, series["temperature"], series[column], color, described)
    return figure


def species_honey_decline(series: pd.DataFrame) -> Figure:
    figure = Figure(figsize=(16, 6))
    species, honey = figure.subplots(1, 2)
    area_trend_panel(
        species,
        series["species_count"],
        "darkred",
        Labels(
            "Decline in Bee Species Diversity", "Year", "Number of Observed Species"
        ),
    )
    area_trend_panel(
        honey,
        series["honey_production"],
        "darkorange",
        Labels("Impact on Honey Production", "Year", "Average Production (kg/hive)"),
    )
    return figure


def geographic_distribution(distribution: pd.DataFrame) -> Figure:
    figure = Figure(figsize=(18, 8))
    axes = figure.subplots(1, 2)
    for axis, period in zip(axes, RANGE_COLORMAPS, strict=True):
        _range_panel(axis, distribution[distribution["period"] == period], period)
    return figure


def climate_impact_summary(series: pd.DataFrame) -> Figure:
    figure = Figure(figsize=(16, 12))
    normalized, spread, change, projection = figure.subplots(2, 2).flat
    _normalized_panel(normalized, series)
    _spread_panel(spread, series)
    _change_panel(change, series)
    _projection_panel(projection, series[IMPACTED])
    return figure


def _range_panel(axis: Axes, points: pd.DataFrame, period: str) -> None:
    dots = axis.scatter(
        points["longitude"],
        points["latitude"],
        c=points["abundance"],
        s=60,
        alpha=0.7,
        cmap=RANGE_COLORMAPS[period],
        edgecolors="black",
        linewidth=0.5,
    )
    axis.figure.colorbar(dots, ax=axis, label="Relative Abundance")
    title = period.replace(" (", " Bee Distribution\n(")
    Labels(title, "Longitude", "Latitude").apply(axis)


def _named(table: pd.DataFrame) -> pd.DataFrame:
    return table.rename(columns=IMPACT_NAMES)


def _colors(table: pd.DataFrame) -> list[str]:
    return [IMPACT_COLORS[name] for name in table.columns]


def _scaled(table: pd.DataFrame) -> pd.DataFrame:
    """Rescale every column to run from 0 to 100."""
    return (table - table.min()) / (table.max() - table.min()) * 100


def _normalized_panel(axis: Axes, series: pd.DataFrame) -> None:
    southward = series.assign(latitude=-series["latitude"])
    table = _named(_scaled(southward))
    labels = Labels("Normalized Trends (0-100)", "Year", "Normalized Index")
    lines_panel(axis, table, _colors(table), labels)


def _spread_panel(axis: Axes, series: pd.DataFrame) -> None:
    magnitudes = _named(series.assign(latitude=series["latitude"].abs()))
    magnitudes.plot.box(ax=axis)
    Labels("Variable Distribution").apply(axis)


def _change_panel(axis: Axes, series: pd.DataFrame) -> None:
    first_year = series.index.min()
    table = _named((series[IMPACTED] / series[IMPACTED].iloc[0] - 1) * 100)
    axis.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    labels = Labels(f"Percentage Change Since {first_year}", "Year", "Change (%)")
    lines_panel(axis, table, _colors(table), labels)


def _extrapolated(series: pd.DataFrame) -> pd.DataFrame:
    """Continue every column's linear trend up to the projection horizon."""
    future = np.arange(series.index.max() + 1, PROJECTION_HORIZON + 1)
    columns = {
        name: np.polyval(np.polyfit(series.index, values, 1), future)
        for name, values in series.items()
    }
    return pd.DataFrame(columns, index=future)


def _projection_panel(axis: Axes, series: pd.DataFrame) -> None:
    last_year = series.index.max()
    table = _named(_scaled(pd.concat([series, _extrapolated(series)])))
    colors = _colors(table)
    labels = Labels(f"Projection to {PROJECTION_HORIZON}", "Year", "Normalized Index")
    table.loc[last_year:].plot(ax=axis, style="--", color=colors, legend=False)
    lines_panel(axis, table.loc[:last_year], colors, labels)
    axis.axvline(x=last_year, color="black", linestyle=":", alpha=0.7)
