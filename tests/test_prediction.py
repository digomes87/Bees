import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from bee_migration.climate.observations import Observations
from bee_migration.prediction.catalog import DEFAULT_CANDIDATES, Candidate
from bee_migration.prediction.features import FeatureTable
from bee_migration.prediction.scores import RegressionScore
from bee_migration.prediction.training import MigrationPredictor
from tests.conftest import region_table

SMALL_CATALOG = (
    Candidate("Linear Regression", LinearRegression(), {}),
    Candidate(
        "Random Forest",
        RandomForestRegressor(n_estimators=20, random_state=42),
        {"max_depth": [2, 4]},
    ),
)


@pytest.fixture(scope="module")
def leaderboard(observations):
    return MigrationPredictor(SMALL_CATALOG, holdout_years=4).train(observations)


def test_features_exclude_anything_derived_from_the_current_target(observations):
    names = FeatureTable(observations).names()

    assert "bee_abundance" not in names
    assert "bee_abundance_change" not in names
    assert "bee_health_index" not in names
    assert "bee_abundance_previous_year" in names


def test_features_drop_the_first_year_which_has_no_previous_year(observations):
    split = FeatureTable(observations).split(holdout_years=4)

    rows = len(split.train.target) + len(split.test.target)

    assert rows == len(observations) - len(observations.region_names())


def test_every_training_year_precedes_every_test_year(observations):
    table = FeatureTable(observations)
    year = table.names().index("year_normalized")

    split = table.split(holdout_years=4)

    assert split.train.features[:, year].max() < split.test.features[:, year].min()


def test_previous_year_feature_carries_last_years_target():
    abundance = [80.0, 70.0, 65.0, 50.0, 45.0, 30.0]
    table = FeatureTable(Observations(region_table("North", bee_abundance=abundance)))
    previous = table.names().index("bee_abundance_previous_year")

    split = table.split(holdout_years=2)

    assert split.train.features[:, previous].tolist() == [80.0, 70.0, 65.0]
    assert split.train.target.tolist() == [70.0, 65.0, 50.0]
    assert split.test.features[:, previous].tolist() == [50.0, 45.0]


@pytest.mark.parametrize("target", ["temperature", "precipitation", "wingspan"])
def test_feature_table_only_predicts_bee_measurements(observations, target):
    with pytest.raises(ValueError, match="target"):
        FeatureTable(observations, target=target)


def test_split_holds_out_the_most_recent_years(observations):
    split = FeatureTable(observations).split(holdout_years=4)

    assert len(split.test.target) == 4 * len(observations.region_names())
    assert split.train.features.shape[1] == split.test.features.shape[1]


@pytest.mark.parametrize("holdout_years", [0, 23, 99])
def test_split_rejects_holdouts_leaving_nothing_to_train_or_test(
    observations, holdout_years
):
    with pytest.raises(ValueError, match="holdout"):
        FeatureTable(observations).split(holdout_years=holdout_years)


def test_perfect_predictions_score_without_error():
    actual = np.array([10.0, 20.0, 30.0])

    score = RegressionScore.of(actual, actual)

    assert score.rmse == 0.0
    assert score.mae == 0.0
    assert score.r2 == 1.0
    assert score.mape == 0.0


def test_score_measures_average_miss():
    score = RegressionScore.of(np.array([10.0, 20.0]), np.array([12.0, 18.0]))

    assert score.mae == pytest.approx(2.0)
    assert score.rmse == pytest.approx(2.0)
    assert score.mape == pytest.approx(15.0)


def test_leaderboard_ranks_every_candidate(leaderboard):
    metrics = leaderboard.metrics()

    assert sorted(metrics.index) == ["Linear Regression", "Random Forest"]
    assert leaderboard.best().name == metrics["test_r2"].idxmax()


def test_best_model_generalises_to_unseen_years(leaderboard):
    assert leaderboard.best().evaluation.test.r2 > 0.0


def test_importances_cover_every_feature_per_model(leaderboard, observations):
    importances = leaderboard.importances()

    assert list(importances.index) == FeatureTable(observations).names()
    assert list(importances.columns) == ["Linear Regression", "Random Forest"]


def test_saved_models_can_be_reloaded_to_predict(leaderboard, observations, tmp_path):
    leaderboard.save(tmp_path / "models")
    sample = FeatureTable(observations).split(holdout_years=4).test

    restored = joblib.load(tmp_path / "models" / "linear_regression_model.pkl")

    assert restored.predict(sample.features).shape == sample.target.shape


def test_default_catalog_offers_the_four_documented_models():
    names = [candidate.name for candidate in DEFAULT_CANDIDATES]

    assert names == ["Linear Regression", "Random Forest", "Gradient Boosting", "SVR"]
