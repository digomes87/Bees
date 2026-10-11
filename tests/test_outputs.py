import pytest
from sklearn.linear_model import LinearRegression

from bee_migration import cli
from bee_migration.charts.gallery import ChartGallery
from bee_migration.climate.continental import continental_series, distribution_shift
from bee_migration.climate.period import StudyPeriod
from bee_migration.prediction.catalog import Candidate
from bee_migration.prediction.training import MigrationPredictor
from bee_migration.reports.analysis_report import AnalysisReport
from bee_migration.reports.prediction_report import PredictionReport

FAST_CATALOG = (Candidate("Linear Regression", LinearRegression(), {}),)

EXPECTED_CHARTS = [
    "climate_impact_summary.png",
    "comparative_temperature_analysis.png",
    "correlation_analysis.png",
    "geographic_distribution.png",
    "migration_patterns_synthesis.png",
    "species_honey_decline.png",
    "temperature_migration_trend.png",
    "ufpr_migration_analysis.png",
]


def test_analysis_report_summarises_every_region(observations):
    text = AnalysisReport(observations).text()

    assert "2000 - 2023" in text
    assert all(region in text for region in observations.region_names())
    assert "Warming in 5 of 5 regions" in text


def test_prediction_report_names_the_best_model(observations):
    leaderboard = MigrationPredictor(FAST_CATALOG, holdout_years=4).train(observations)

    text = PredictionReport(leaderboard).text()

    assert "Best model: Linear Regression" in text
    assert "Test R2" in text


def test_gallery_renders_every_chart(tmp_path):
    series = continental_series(StudyPeriod(2000, 2023), seed=42)
    gallery = ChartGallery(tmp_path / "images", resolution=40)

    rendered = gallery.render(series, distribution_shift(seed=42))

    assert sorted(path.name for path in rendered) == EXPECTED_CHARTS
    assert all(path.stat().st_size > 0 for path in rendered)


def test_analyze_command_writes_report_and_processed_data(tmp_path):
    exit_code = cli.main(["analyze", "--root", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "results" / "analysis_summary_report.txt").exists()
    assert (tmp_path / "data" / "processed" / "bee_climate_processed.csv").exists()
    assert (
        tmp_path / "results" / "correlation_analysis_correlation_matrix.csv"
    ).exists()


def test_analyze_command_accepts_previously_saved_observations(observations, tmp_path):
    source = tmp_path / "input.csv"
    observations.save(source)

    exit_code = cli.main(["analyze", "--root", str(tmp_path), "--data", str(source)])

    assert exit_code == 0


def test_predict_command_writes_metrics_and_models(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DEFAULT_CANDIDATES", FAST_CATALOG)

    exit_code = cli.main(["predict", "--root", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "results" / "model_metrics.csv").exists()
    assert (tmp_path / "results" / "feature_importance.csv").exists()
    assert (tmp_path / "results" / "models" / "linear_regression_model.pkl").exists()


def test_missing_data_file_is_reported_without_a_traceback(tmp_path, caplog):
    exit_code = cli.main(
        ["analyze", "--root", str(tmp_path), "--data", str(tmp_path / "absent.csv")]
    )

    assert exit_code == 1
    assert "absent.csv" in caplog.text


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        cli.main(["swarm"])
