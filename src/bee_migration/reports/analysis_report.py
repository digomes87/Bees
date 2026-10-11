from pathlib import Path

from bee_migration.analysis.correlation import Correlations
from bee_migration.analysis.temporal import TemporalPatterns
from bee_migration.climate.observations import Observations
from bee_migration.reports.layout import banner, document, section

LISTED_CORRELATIONS = 5


class AnalysisReport:
    """Plain-text summary of trends and correlations in the observations."""

    def __init__(self, observations: Observations) -> None:
        self._observations = observations

    def text(self) -> str:
        return document(
            [
                banner("ANALYSIS REPORT: BEES AND CLIMATE CHANGE"),
                self._coverage(),
                self._averages(),
                self._trends(),
                self._correlations(),
                self._findings(),
            ]
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text(), encoding="utf-8")

    def _coverage(self) -> str:
        years = self._observations.table()["year"]
        regions = ", ".join(self._observations.region_names())
        return "\n".join(
            [
                f"Period analysed: {years.min()} - {years.max()}",
                f"Regions analysed: {regions}",
                f"Total records: {len(self._observations)}",
            ]
        )

    def _averages(self) -> str:
        means = self._observations.table().mean(numeric_only=True)
        return section(
            "DATA SUMMARY",
            [
                f"Mean temperature: {means['temperature']:.1f} °C",
                f"Mean precipitation: {means['precipitation']:.0f} mm",
                f"Mean bee abundance: {means['bee_abundance']:.1f}",
                f"Mean species diversity: {means['species_diversity']:.1f}",
                f"Mean honey production: {means['honey_production']:.1f} kg",
            ],
        )

    def _trends(self) -> str:
        trends = TemporalPatterns(self._observations).trends()
        lines = [
            f"{region}: temperature {row['temperature']:+.3f} °C/year, "
            f"bee abundance {row['bee_abundance']:+.2f}/year"
            for region, row in trends.iterrows()
        ]
        return section("REGIONAL TRENDS", lines)

    def _correlations(self) -> str:
        pairs = Correlations(self._observations).notable_pairs()
        lines = [pair.describe() for pair in pairs[:LISTED_CORRELATIONS]]
        return section("STRONGEST CORRELATIONS", lines)

    def _findings(self) -> str:
        trends = TemporalPatterns(self._observations).trends()
        regions = len(trends)
        warming = int((trends["temperature"] > 0).sum())
        declining = int((trends["bee_abundance"] < 0).sum())
        return section(
            "KEY FINDINGS",
            [
                f"1. Warming in {warming} of {regions} regions",
                f"2. Bee abundance declining in {declining} of {regions} regions",
            ],
        )
