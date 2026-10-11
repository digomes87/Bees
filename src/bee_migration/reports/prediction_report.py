from pathlib import Path

from bee_migration.prediction.training import Leaderboard
from bee_migration.reports.layout import banner, document, section


class PredictionReport:
    """Plain-text comparison of the trained models."""

    def __init__(self, leaderboard: Leaderboard) -> None:
        self._leaderboard = leaderboard

    def text(self) -> str:
        best = self._leaderboard.best()
        performance = [
            line for result in self._leaderboard for line in result.describe()
        ]
        return document(
            [
                banner("PREDICTION REPORT: MACHINE LEARNING"),
                f"Models trained: {len(self._leaderboard)}",
                section("PERFORMANCE ON HELD-OUT YEARS", performance),
                section(f"MOST IMPORTANT FEATURES ({best.name})", best.top_features()),
                section("RECOMMENDATION", [f"Best model: {best.name}"]),
            ]
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text(), encoding="utf-8")
