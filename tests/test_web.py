import json

import pytest

from bee_migration.prediction.features import FeatureTable
from bee_migration.web.study import StudyRequest, run_study, simulated_study

REGIONS = 5


@pytest.fixture(scope="module")
def study() -> dict:
    return simulated_study(StudyRequest(seed=42, holdout_years=4))


def test_study_is_plain_json(study):
    assert json.loads(json.dumps(study)) == study


def test_study_lists_every_observation_with_its_measurements(study):
    first = study["observations"][0]

    assert len(study["observations"]) == REGIONS * 24
    assert {"year", "region", "temperature", "bee_abundance"} <= set(first)


def test_study_reports_one_trend_per_region(study):
    regions = [trend["region"] for trend in study["trends"]]

    assert regions == study["regions"]
    assert all(trend["temperature"] > 0 for trend in study["trends"])


def test_correlation_matrix_matches_its_variables(study):
    correlations = study["correlations"]

    size = len(correlations["variables"])

    assert len(correlations["matrix"]) == size
    assert all(len(row) == size for row in correlations["matrix"])


def test_study_forecasts_every_held_out_region_and_year(study):
    forecasts = study["forecasts"]
    models = [model["model"] for model in study["models"]]

    assert len(forecasts) == REGIONS * 4
    assert {row["year"] for row in forecasts} == {2020, 2021, 2022, 2023}
    assert all(name in forecasts[0] for name in models)
    assert study["best"] in models


def test_forecast_rows_carry_the_observed_value(study, observations):
    table = observations.table()
    row = study["forecasts"][0]

    observed = table[
        (table["year"] == row["year"]) & (table["region"] == row["region"])
    ]

    assert row["actual"] == pytest.approx(observed["bee_abundance"].iloc[0])


def test_importance_ranks_features_from_most_influential(study):
    values = [entry["importance"] for entry in study["importance"]]

    assert values == sorted(values, reverse=True)
    assert 0 < len(values) <= 8


def test_holdout_labels_align_with_the_test_sample(observations):
    features = FeatureTable(observations)

    labels = features.holdout(4)
    sample = features.split(4).test

    assert labels["actual"].tolist() == sample.target.tolist()
    assert list(labels.columns) == ["year", "region", "actual"]


def test_run_study_speaks_json_both_ways():
    answer = json.loads(run_study(json.dumps({"seed": 7, "holdout_years": 3})))

    assert answer["request"] == {
        "seed": 7,
        "holdout_years": 3,
        "target": "bee_abundance",
        "thorough": False,
    }
    assert len(answer["forecasts"]) == REGIONS * 3


def test_study_request_rejects_unknown_settings():
    with pytest.raises(ValueError, match="Unknown study settings"):
        StudyRequest.parsed(json.dumps({"seeds": 7}))
