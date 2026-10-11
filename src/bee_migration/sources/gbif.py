"""Bee occurrence records published through GBIF."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from bee_migration.climate.period import StudyPeriod

OCCURRENCES = "https://api.gbif.org/v1/occurrence/search"
APIDAE = 4334
CONTINENT = "SOUTH_AMERICA"
SOUTHERN_TIP = -55.0
NORTHERN_TIP = 15.0
BAND_HEIGHT = 5.0
MOST_YEARS = 200


@dataclass(frozen=True)
class LatitudeBand:
    """A strip of the continent between two parallels."""

    south: float
    north: float

    def centre(self) -> float:
        return (self.south + self.north) / 2

    def url(self, period: StudyPeriod) -> str:
        """Address counting the bee records of this band for each year."""
        query = {
            "taxonKey": APIDAE,
            "continent": CONTINENT,
            "hasCoordinate": "true",
            "decimalLatitude": f"{self.south},{self.north}",
            "year": f"{period.first_year},{period.last_year}",
            "facet": "year",
            "facetLimit": MOST_YEARS,
            "limit": 0,
        }
        return f"{OCCURRENCES}?{urlencode(query)}"


def latitude_bands(
    south: float = SOUTHERN_TIP,
    north: float = NORTHERN_TIP,
    height: float = BAND_HEIGHT,
) -> list[LatitudeBand]:
    edges = np.arange(south, north, height).tolist()
    return [LatitudeBand(edge, edge + height) for edge in edges]


def yearly_counts(answer: Mapping[str, Any]) -> dict[int, int]:
    """Records per year from an occurrence search faceted by year."""
    facets = [facet for facet in answer.get("facets", []) if facet["field"] == "YEAR"]
    if not facets:
        raise ValueError("GBIF answered without the yearly record counts")
    return {int(entry["name"]): int(entry["count"]) for entry in facets[0]["counts"]}


def record_centroids(
    counts: Mapping[LatitudeBand, Mapping[int, int]],
) -> pd.DataFrame:
    """Records per year and their mean latitude, weighing each band by its records."""
    table = pd.DataFrame(
        {
            band.centre(): pd.Series(yearly, dtype=float)
            for band, yearly in counts.items()
        }
    ).fillna(0.0)
    records = table.sum(axis=1)
    weighted = (table * table.columns.to_numpy()).sum(axis=1)
    centroids = pd.DataFrame(
        {
            "year": table.index.astype(int),
            "records": records.astype(int).to_numpy(),
            "mean_latitude": (weighted / records).to_numpy(),
        }
    )
    return centroids.sort_values("year").reset_index(drop=True)
