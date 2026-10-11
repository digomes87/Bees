from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from bee_migration.charts import evidence, observed

PUBLICATION_RESOLUTION = 300


class ChartGallery:
    """Renders every project chart as an image file in one directory."""

    def __init__(
        self, directory: Path, resolution: int = PUBLICATION_RESOLUTION
    ) -> None:
        self._directory = directory
        self._resolution = resolution

    def render(self, series: pd.DataFrame, distribution: pd.DataFrame) -> list[Path]:
        """Draw all charts from the continental series and the range points."""
        self._directory.mkdir(parents=True, exist_ok=True)
        figures = {**_observed(series.set_index("year"), distribution), **_evidence()}
        return [self._saved(name, figure) for name, figure in figures.items()]

    def _saved(self, name: str, figure: Figure) -> Path:
        path = self._directory / f"{name}.png"
        figure.tight_layout()
        figure.savefig(path, dpi=self._resolution, bbox_inches="tight")
        return path


def _observed(series: pd.DataFrame, distribution: pd.DataFrame) -> dict[str, Figure]:
    return {
        "temperature_migration_trend": observed.temperature_migration_trend(series),
        "correlation_analysis": observed.correlation_analysis(series),
        "species_honey_decline": observed.species_honey_decline(series),
        "geographic_distribution": observed.geographic_distribution(distribution),
        "climate_impact_summary": observed.climate_impact_summary(series),
    }


def _evidence() -> dict[str, Figure]:
    return {
        "ufpr_migration_analysis": evidence.ufpr_migration_analysis(),
        "comparative_temperature_analysis": evidence.comparative_temperature_analysis(),
        "migration_patterns_synthesis": evidence.migration_patterns_synthesis(),
    }
