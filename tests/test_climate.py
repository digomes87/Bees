import numpy as np
import pandas as pd
import pytest

from bee_migration.climate.continental import continental_series, distribution_shift
from bee_migration.climate.observations import HEALTH_WEIGHTS, Observations
from bee_migration.climate.period import StudyPeriod
from bee_migration.climate.region import BRAZILIAN_REGIONS
from bee_migration.climate.simulation import ClimateSimulation
from bee_migration.climate.trend import Drift, Trend
from tests.conftest import region_table


def test_study_period_lists_every_year_inclusive():
    years = StudyPeriod(2000, 2003).years()

    assert years.tolist() == [2000, 2001, 2002, 2003]


def test_study_period_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="before"):
        StudyPeriod(2010, 2000)


def test_trend_without_noise_follows_its_drift():
    trend = Trend(Drift(start=10.0, total_change=4.0), spread=0.0)

    sample = trend.sample(5, np.random.default_rng(0))

    assert sample.tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]


def test_trend_never_falls_below_its_floor():
    trend = Trend(Drift(start=5.0, total_change=-50.0), spread=3.0, floor=2.0)

    sample = trend.sample(100, np.random.default_rng(0))

    assert sample.min() >= 2.0


def test_simulation_yields_one_row_per_region_and_year(observations):
    assert len(observations) == len(BRAZILIAN_REGIONS) * 24
    assert observations.region_names() == sorted(
        profile.name for profile in BRAZILIAN_REGIONS
    )


def test_simulation_is_reproducible_for_the_same_seed():
    simulation = ClimateSimulation(BRAZILIAN_REGIONS, StudyPeriod(2000, 2005))

    first = simulation.observations(seed=7).table()
    second = simulation.observations(seed=7).table()
    other = simulation.observations(seed=8).table()

    pd.testing.assert_frame_equal(first, second)
    assert not first["temperature"].equals(other["temperature"])


def test_observations_reject_tables_missing_measurements():
    incomplete = region_table("North", temperature=[20.0, 21.0]).drop(
        columns="precipitation"
    )

    with pytest.raises(ValueError, match="precipitation"):
        Observations(incomplete)


def test_observations_reject_empty_tables():
    empty = region_table("North", temperature=[20.0]).iloc[0:0]

    with pytest.raises(ValueError, match="empty"):
        Observations(empty)


def test_observations_reject_non_numeric_measurements():
    table = region_table("North", temperature=[20.0, 21.0]).astype({"temperature": str})
    table.loc[0, "temperature"] = "warm"

    with pytest.raises(ValueError, match="non-numeric"):
        Observations(table)


def test_observations_reject_missing_values():
    table = region_table("North", temperature=[20.0, float("nan"), 22.0])

    with pytest.raises(ValueError, match="missing values"):
        Observations(table)


def test_observations_reject_a_year_repeated_within_a_region():
    table = region_table("North", temperature=[20.0, 21.0])

    with pytest.raises(ValueError, match="repeat a year"):
        Observations(pd.concat([table, table], ignore_index=True))


def test_observations_reject_gaps_between_years():
    table = region_table("North", temperature=[20.0, 21.0, 22.0, 23.0]).drop(index=1)

    with pytest.raises(ValueError, match="skip years"):
        Observations(table)


def test_anomalies_are_centred_within_each_region(observations):
    table = observations.table()

    centres = table.groupby("region")["temperature_anomaly"].mean()

    assert np.allclose(centres, 0.0)


def test_health_index_weighs_abundance_diversity_and_honey():
    table = region_table(
        "North",
        bee_abundance=[100.0, 50.0],
        species_diversity=[10.0, 20.0],
        honey_production=[30.0, 40.0],
    )

    health = Observations(table).table()["bee_health_index"]

    expected = (
        HEALTH_WEIGHTS["bee_abundance"] * 100.0
        + HEALTH_WEIGHTS["species_diversity"] * 10.0
        + HEALTH_WEIGHTS["honey_production"] * 30.0
    )
    assert health.iloc[0] == pytest.approx(expected)


def test_observations_survive_a_csv_round_trip(observations, tmp_path):
    path = tmp_path / "nested" / "observations.csv"

    observations.save(path)
    restored = Observations.from_csv(path)

    pd.testing.assert_frame_equal(restored.table(), observations.table())


def test_table_is_a_defensive_copy(observations):
    table = observations.table()
    table["temperature"] = 0.0

    assert observations.table()["temperature"].mean() > 0.0


def test_continental_series_drifts_south_while_warming():
    series = continental_series(StudyPeriod(2000, 2023), seed=42)

    assert series["temperature"].iloc[-1] > series["temperature"].iloc[0]
    assert series["latitude"].iloc[-1] < series["latitude"].iloc[0]
    assert list(series.columns) == [
        "year",
        "temperature",
        "latitude",
        "species_count",
        "honey_production",
    ]


def test_distribution_shift_projects_a_more_southern_range():
    points = distribution_shift(seed=42)

    latitude = points.groupby("period")["latitude"].mean()

    assert len(points) == 200
    assert latitude["Projected (2040-2050)"] < latitude["Current (2020-2023)"]
