"""Command line entry point: analyse, predict and chart bee migration."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from bee_migration.analysis.correlation import Correlations
from bee_migration.charts.gallery import ChartGallery
from bee_migration.climate.continental import continental_series, distribution_shift
from bee_migration.climate.observations import Observations
from bee_migration.climate.period import DEFAULT_PERIOD
from bee_migration.climate.region import BRAZILIAN_REGIONS
from bee_migration.climate.simulation import DEFAULT_SEED, ClimateSimulation
from bee_migration.prediction.catalog import DEFAULT_CANDIDATES
from bee_migration.prediction.training import MigrationPredictor
from bee_migration.reports.analysis_report import AnalysisReport
from bee_migration.reports.prediction_report import PredictionReport
from bee_migration.workspace import Workspace

logger = logging.getLogger(__name__)


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        options.command(options)
    except (OSError, ValueError) as error:
        logger.error("Could not complete '%s': %s", options.name, error)
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--root", type=Path, default=Path.cwd(), help="project folder")
    shared.add_argument(
        "--data", type=Path, help="observations CSV; simulated if absent"
    )
    shared.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser = argparse.ArgumentParser(prog="bee-migration", description=__doc__)
    commands = parser.add_subparsers(dest="name", required=True)
    for name, command in COMMANDS.items():
        commands.add_parser(name, parents=[shared], help=command.__doc__).set_defaults(
            command=command
        )
    return parser


def _observations(options: argparse.Namespace) -> Observations:
    if options.data is not None:
        return Observations.from_csv(options.data)
    simulation = ClimateSimulation(BRAZILIAN_REGIONS, DEFAULT_PERIOD)
    return simulation.observations(options.seed)


def analyze(options: argparse.Namespace) -> None:
    """Summarise trends and correlations in the observations."""
    workspace = Workspace(options.root)
    observations = _observations(options)
    observations.save(workspace.data("processed/bee_climate_processed.csv"))
    matrix = Correlations(observations).matrix()
    matrix.to_csv(workspace.result("correlation_analysis_correlation_matrix.csv"))
    report = workspace.result("analysis_summary_report.txt")
    AnalysisReport(observations).save(report)
    logger.info("Analysis of %d records written to %s", len(observations), report)


def predict(options: argparse.Namespace) -> None:
    """Train models forecasting bee abundance and compare them."""
    workspace = Workspace(options.root)
    leaderboard = MigrationPredictor(DEFAULT_CANDIDATES).train(_observations(options))
    leaderboard.save(workspace.result("models"))
    leaderboard.metrics().to_csv(workspace.result("model_metrics.csv"))
    leaderboard.importances().to_csv(workspace.result("feature_importance.csv"))
    report = workspace.result("ml_predictions_report.txt")
    PredictionReport(leaderboard).save(report)
    logger.info("Comparison of %d models written to %s", len(leaderboard), report)


def charts(options: argparse.Namespace) -> None:
    """Render every chart shown in the README."""
    workspace = Workspace(options.root)
    if options.data is not None:
        logger.warning("Charts draw the simulated continental series, not --data")
    series = continental_series(DEFAULT_PERIOD, options.seed)
    distribution = distribution_shift(options.seed)
    rendered = ChartGallery(workspace.images()).render(series, distribution)
    series.to_csv(workspace.data("bee_climate_data.csv"), index=False)
    distribution.to_csv(workspace.data("bee_geographic_data.csv"), index=False)
    logger.info("%d charts written to %s", len(rendered), workspace.images())


def everything(options: argparse.Namespace) -> None:
    """Run the analysis, the predictions and the charts."""
    analyze(options)
    predict(options)
    charts(options)


COMMANDS = {
    "analyze": analyze,
    "predict": predict,
    "charts": charts,
    "all": everything,
}
