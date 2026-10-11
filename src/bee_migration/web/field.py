"""Field data as plain JSON: observed weather and where bees were recorded."""

import json
from typing import Any

import numpy as np
import pandas as pd

from bee_migration.climate.period import DEFAULT_PERIOD
from bee_migration.climate.region import BRAZILIAN_REGIONS
from bee_migration.sources.gbif import (
    LatitudeBand,
    latitude_bands,
    record_centroids,
    yearly_counts,
)
from bee_migration.sources.open_meteo import archive_url, yearly_climate

YEARS_PER_DECADE = 10
DECIMALS = 4


def field_requests() -> str:
    """The addresses to fetch, as JSON, before calling `field_report`."""
    bands = [
        {"south": band.south, "north": band.north, "url": band.url(DEFAULT_PERIOD)}
        for band in latitude_bands()
    ]
    return json.dumps(
        {"climate": archive_url(BRAZILIAN_REGIONS, DEFAULT_PERIOD), "bands": bands}
    )


def field_report(document: str) -> str:
    """Summarise the fetched answers: `climate` and one `answer` per band."""
    fetched = json.loads(document)
    names = [region.name for region in BRAZILIAN_REGIONS]
    climate = yearly_climate(fetched["climate"], names)
    centroids = record_centroids(
        {
            LatitudeBand(band["south"], band["north"]): yearly_counts(band["answer"])
            for band in fetched["bands"]
        }
    )
    return json.dumps(
        {
            "regions": sorted(names),
            "climate": _records(climate),
            "warming_per_decade": _warming(climate),
            "centroids": _records(centroids),
            "latitude_shift_per_decade": _per_decade(
                centroids["year"], centroids["mean_latitude"]
            ),
            "total_records": int(centroids["records"].sum()),
        }
    )


def _per_decade(years: pd.Series, values: pd.Series) -> float:
    return float(np.polyfit(years, values, 1)[0]) * YEARS_PER_DECADE


def _warming(climate: pd.DataFrame) -> float:
    """Mean across regions of each region's temperature change per decade."""
    changes = [
        _per_decade(rows["year"], rows["temperature"])
        for _, rows in climate.groupby("region")
    ]
    return float(np.mean(changes))


def _records(table: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(table.to_json(orient="records", double_precision=DECIMALS + 2))
