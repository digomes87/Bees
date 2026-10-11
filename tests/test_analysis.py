import numpy as np
import pandas as pd
import pytest

from bee_migration.analysis.correlation import CorrelationPair, Correlations
from bee_migration.analysis.geographic import GeographicPatterns
from bee_migration.analysis.temporal import TemporalPatterns
from bee_migration.climate.observations import Observations
from tests.conftest import region_table


def test_trends_show_warming_and_bee_decline_in_every_region(observations):
    trends = TemporalPatterns(observations).trends()

    assert (trends["temperature"] > 0).all()
    assert (trends["bee_abundance"] < 0).all()
    assert list(trends.index) == observations.region_names()


def test_trend_is_the_yearly_slope():
    table = region_table("North", temperature=[10.0, 10.5, 11.0, 11.5])

    trends = TemporalPatterns(Observations(table)).trends()

    assert trends.loc["North", "temperature"] == pytest.approx(0.5)


def test_change_points_flag_the_year_of_an_abrupt_shift():
    steady = [20.0] * 20
    jump = steady[:12] + [value + 9.0 for value in steady[12:]]
    table = region_table("North", temperature=jump)

    changes = TemporalPatterns(Observations(table)).change_points()

    assert 2012 in changes.loc["North", "temperature"]


def test_correlation_matrix_is_symmetric_with_unit_diagonal(observations):
    matrix = Correlations(observations).matrix()

    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)


def test_notable_pairs_are_ordered_from_strongest(observations):
    pairs = Correlations(observations).notable_pairs(threshold=0.3)

    magnitudes = [abs(pair.coefficient) for pair in pairs]

    assert magnitudes == sorted(magnitudes, reverse=True)
    assert min(magnitudes) >= 0.3


def test_notable_pairs_never_pair_a_variable_with_itself(observations):
    pairs = Correlations(observations).notable_pairs(threshold=0.0)

    assert all(pair.variables[0] != pair.variables[1] for pair in pairs)


@pytest.mark.parametrize(
    ("coefficient", "strength"),
    [(0.9, "strong"), (-0.75, "strong"), (0.4, "moderate"), (0.1, "weak")],
)
def test_correlation_pair_names_its_strength(coefficient, strength):
    pair = CorrelationPair(("temperature", "bee_abundance"), coefficient)

    assert pair.strength() == strength


def test_lagged_correlation_pairs_temperature_with_later_abundance():
    temperature = [20.0, 23.0, 21.0, 26.0, 22.0, 27.0, 24.0, 29.0]
    delayed = [50.0] + [100.0 - value for value in temperature[:-1]]
    table = region_table("North", temperature=temperature, bee_abundance=delayed)

    lagged = Correlations(Observations(table)).lagged(lags=(0, 1))

    assert lagged.loc["North", 1] == pytest.approx(-1.0)
    assert abs(lagged.loc["North", 0]) < 0.9


def test_regional_correlations_cover_every_region(observations):
    regional = Correlations(observations).by_region()

    assert sorted(regional) == observations.region_names()


def test_regional_summary_reports_spread_per_measurement(observations):
    summary = GeographicPatterns(observations).summary()

    assert (
        summary.loc["North", ("temperature", "max")]
        >= summary.loc["North", ("temperature", "min")]
    )


def test_latitudinal_gradient_covers_every_measurement(observations):
    gradients = GeographicPatterns(observations).latitudinal_gradients()

    assert gradients.between(-1.0, 1.0).all()
    assert "bee_abundance" in gradients.index


def test_hotspots_rank_regions_by_combined_change():
    calm = region_table("Calm", temperature=[20.0, 20.1], bee_abundance=[80.0, 79.0])
    harsh = region_table("Harsh", temperature=[20.0, 24.0], bee_abundance=[80.0, 40.0])
    observations = Observations(pd.concat([calm, harsh], ignore_index=True))

    hotspots = GeographicPatterns(observations).hotspots()

    assert hotspots.index[0] == "Harsh"
    assert hotspots.loc["Harsh", "bee_abundance_change"] == pytest.approx(40.0)
