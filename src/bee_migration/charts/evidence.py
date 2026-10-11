"""Charts of the figures gathered from the cited studies."""

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bee_migration.charts.panels import (
    Labels,
    bar_panel,
    grouped_bar_panel,
    horizontal_bar_panel,
    lines_panel,
)
from bee_migration.evidence import literature


def _titled(title: str) -> tuple[Figure, list[Axes]]:
    figure = Figure(figsize=(16, 12))
    figure.suptitle(title, fontsize=16, fontweight="bold")
    return figure, list(figure.subplots(2, 2).flat)


def ufpr_migration_analysis() -> Figure:
    figure, (species, directions, factors, intensity) = _titled(
        "UFPR Migration Analysis: South American Bees Moving South"
    )
    grouped_bar_panel(
        species,
        literature.SPECIES_BY_REGION,
        ["lightblue", "darkblue"],
        Labels("Species Distribution by Region", "Brazilian Regions", "Species"),
    )
    bar_panel(
        directions,
        literature.MIGRATION_DIRECTIONS,
        ["darkgreen", "green", "lightgreen", "gray", "red"],
        Labels("Migration Direction Distribution", "Direction", "Species (%)"),
        unit="%",
    )
    horizontal_bar_panel(
        factors,
        literature.IMPACT_FACTORS,
        "orange",
        Labels("Migration Impact Factors", "Impact Intensity (1-10)"),
    )
    _intensity_panel(intensity)
    return figure


def comparative_temperature_analysis() -> Figure:
    figure, (warming, timeline, thresholds, biomes) = _titled(
        "Temperature Trends Comparison: Local Data vs Scientific Studies"
    )
    bar_panel(
        warming,
        literature.TEMPERATURE_INCREASE,
        ["red", "orange", "yellow", "blue", "purple"],
        Labels("Projected Temperature Increase by Region", "", "Increase (°C)"),
        unit="°C",
    )
    _timeline_panel(timeline)
    _threshold_panel(thresholds)
    horizontal_bar_panel(
        biomes,
        literature.BIOME_VULNERABILITY,
        ["darkgreen", "brown", "green", "yellow", "lightgreen"],
        Labels("Climate Vulnerability Index by Biome", "Vulnerability Index (1-10)"),
    )
    return figure


def migration_patterns_synthesis() -> Figure:
    figure, (studies, scenarios, groups, priorities) = _titled(
        "Bee Migration Patterns: Scientific Evidence Synthesis"
    )
    bar_panel(
        studies,
        literature.SOUTHWARD_SHARE_BY_STUDY,
        ["green", "blue", "orange", "purple"],
        Labels("Species with Southward Migration", "", "Southward Migration (%)"),
        unit="%",
    )
    _scenario_panel(scenarios)
    grouped_bar_panel(
        groups,
        literature.FUNCTIONAL_RESPONSE,
        ["red", "blue"],
        Labels("Functional Group Response to Climate Change", "", "Percentage (%)"),
    )
    _priority_panel(priorities)
    return figure


def _intensity_panel(axis: Axes) -> None:
    intensity = pd.Series(literature.MIGRATION_INTENSITY)
    axis.fill_between(intensity.index, intensity.to_numpy(), alpha=0.3, color="blue")
    axis.plot(intensity.index, intensity.to_numpy(), "bo-", linewidth=3)
    labels = Labels(
        "Southward Migration Intensity Projection", "Year", "Migrating Species (%)"
    )
    labels.apply(axis)


def _timeline_panel(axis: Axes) -> None:
    table = pd.DataFrame(literature.ABUNDANCE_TIMELINE, index=literature.TIMELINE_YEARS)
    labels = Labels("Species Abundance Timeline", "Year", "Relative Abundance (%)")
    lines_panel(axis, table, ["red", "blue"], labels)
    axis.axvline(x=literature.PRESENT_YEAR, color="black", linestyle="--", alpha=0.7)


def _threshold_panel(axis: Axes) -> None:
    bar_panel(
        axis,
        literature.CRITICAL_WARMING,
        ["green", "blue", "orange", "red"],
        Labels("Critical Warming by Functional Group", "", "Critical Increase (°C)"),
        unit="°C",
    )
    axis.axhline(
        y=literature.AVERAGE_CRITICAL_WARMING,
        color="red",
        linestyle="--",
        label="Global Average Threshold",
    )
    axis.legend()


def _scenario_panel(axis: Axes) -> None:
    table = pd.DataFrame(
        literature.MIGRATION_SCENARIOS, index=literature.SCENARIO_YEARS
    )
    lowest, highest = table.min(axis=1), table.max(axis=1)
    axis.fill_between(table.index, lowest, highest, alpha=0.2, color="gray")
    labels = Labels(
        "Migration Velocity Projections", "Year", "Cumulative Migration (%)"
    )
    lines_panel(axis, table, ["green", "blue", "red"], labels)


def _priority_panel(axis: Axes) -> None:
    richness = axis.twinx()
    areas = list(literature.PRIORITY_SCORES)
    axis.bar(
        areas, list(literature.PRIORITY_SCORES.values()), color="darkgreen", alpha=0.7
    )
    richness.plot(areas, list(literature.SPECIES_RICHNESS.values()), "ro-", linewidth=2)
    richness.set_ylabel("Species Richness", color="red")
    axis.tick_params(axis="x", labelsize=8)
    Labels("Conservation Priority Areas", "", "Priority Score (1-10)").apply(axis)
