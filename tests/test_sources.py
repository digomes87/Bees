import json
from urllib.parse import parse_qs, urlparse

import pytest

from bee_migration.climate.period import StudyPeriod
from bee_migration.climate.region import BRAZILIAN_REGIONS
from bee_migration.sources.gbif import (
    LatitudeBand,
    latitude_bands,
    record_centroids,
    yearly_counts,
)
from bee_migration.sources.open_meteo import archive_url, yearly_climate
from bee_migration.web.field import field_report, field_requests

PERIOD = StudyPeriod(2000, 2001)


def location(temperatures: list[float | None]) -> dict:
    days = ["2000-12-30", "2000-12-31", "2001-01-01", "2001-01-02"]
    return {"daily": {"time": days, "temperature_2m_mean": temperatures}}


def facets(counts: dict[str, int]) -> dict:
    listed = [{"name": year, "count": count} for year, count in counts.items()]
    return {
        "count": sum(counts.values()),
        "facets": [{"field": "YEAR", "counts": listed}],
    }


def test_archive_url_asks_for_every_region_over_the_whole_period():
    query = parse_qs(urlparse(archive_url(BRAZILIAN_REGIONS, PERIOD)).query)

    assert query["latitude"] == ["-3.0,-9.0,-15.0,-20.0,-27.0"]
    assert query["start_date"] == ["2000-01-01"]
    assert query["end_date"] == ["2001-12-31"]
    assert query["daily"] == ["temperature_2m_mean"]


def test_yearly_climate_averages_the_daily_temperatures_of_each_year():
    climate = yearly_climate([location([20.0, 22.0, 30.0, 32.0])], ["North"])

    assert climate.to_dict("records") == [
        {"year": 2000, "region": "North", "temperature": 21.0},
        {"year": 2001, "region": "North", "temperature": 31.0},
    ]


def test_yearly_climate_accepts_the_single_location_shape():
    climate = yearly_climate(location([20.0, 22.0, 30.0, 32.0]), ["South"])

    assert climate["region"].unique().tolist() == ["South"]


def test_yearly_climate_ignores_days_without_a_reading():
    climate = yearly_climate([location([20.0, None, 30.0, 32.0])], ["North"])

    assert climate.loc[0, "temperature"] == 20.0


def test_yearly_climate_rejects_a_mismatched_number_of_locations():
    with pytest.raises(ValueError, match="locations"):
        yearly_climate([location([1.0] * 4)], ["North", "South"])


def test_yearly_climate_rejects_an_error_answer():
    with pytest.raises(ValueError, match="Open-Meteo"):
        yearly_climate({"error": True, "reason": "Too many requests"}, ["North"])


def test_latitude_bands_tile_the_continent_without_gaps():
    bands = latitude_bands(south=-10.0, north=5.0, height=5.0)

    assert bands == [
        LatitudeBand(-10.0, -5.0),
        LatitudeBand(-5.0, 0.0),
        LatitudeBand(0.0, 5.0),
    ]


def test_band_url_filters_bees_by_latitude_and_year():
    query = parse_qs(urlparse(LatitudeBand(-30.0, -25.0).url(PERIOD)).query)

    assert query["decimalLatitude"] == ["-30.0,-25.0"]
    assert query["year"] == ["2000,2001"]
    assert query["facet"] == ["year"]
    assert query["limit"] == ["0"]


def test_yearly_counts_reads_the_year_facet():
    assert yearly_counts(facets({"2000": 12, "2001": 30})) == {2000: 12, 2001: 30}


def test_yearly_counts_rejects_answers_without_the_year_facet():
    with pytest.raises(ValueError, match="GBIF"):
        yearly_counts({"count": 3, "facets": []})


def test_centroid_is_the_record_weighted_mean_latitude():
    counts = {
        LatitudeBand(-30.0, -20.0): {2000: 10, 2001: 30},
        LatitudeBand(-10.0, 0.0): {2000: 30, 2001: 10},
    }

    centroids = record_centroids(counts)

    assert centroids.to_dict("records") == [
        {"year": 2000, "records": 40, "mean_latitude": -10.0},
        {"year": 2001, "records": 40, "mean_latitude": -20.0},
    ]


def test_centroid_skips_years_without_records():
    counts = {LatitudeBand(-10.0, 0.0): {2001: 4}}

    assert record_centroids(counts)["year"].tolist() == [2001]


def test_field_requests_list_one_climate_url_and_one_url_per_band():
    requests = json.loads(field_requests())

    assert requests["climate"].startswith("https://archive-api.open-meteo.com/")
    assert len(requests["bands"]) == 14
    assert all(
        band["url"].startswith("https://api.gbif.org/") for band in requests["bands"]
    )


def test_field_report_summarises_warming_and_the_drift_of_records():
    names = [profile.name for profile in BRAZILIAN_REGIONS]
    climate = [location([20.0, 20.0, 21.0, 21.0]) for _ in names]
    bands = [
        {"south": -30.0, "north": -20.0, "answer": facets({"2000": 10, "2001": 30})},
        {"south": -10.0, "north": 0.0, "answer": facets({"2000": 30, "2001": 10})},
    ]

    report = json.loads(field_report(json.dumps({"climate": climate, "bands": bands})))

    assert report["regions"] == sorted(names)
    assert len(report["climate"]) == 2 * len(names)
    assert report["warming_per_decade"] == pytest.approx(10.0)
    assert report["latitude_shift_per_decade"] == pytest.approx(-100.0)
    assert report["total_records"] == 80
