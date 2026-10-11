"""Historical weather from the Open-Meteo archive (ERA5 reanalysis)."""

from collections.abc import Sequence
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from bee_migration.climate.period import StudyPeriod
from bee_migration.climate.region import RegionProfile

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
# One variable keeps a 24-year request within the free per-minute allowance.
DAILY_VARIABLE = "temperature_2m_mean"


def archive_url(regions: Sequence[RegionProfile], period: StudyPeriod) -> str:
    """Address of the daily temperature at every region's centre over the period."""
    centres = [region.centre for region in regions]
    query = {
        "latitude": ",".join(str(centre.latitude) for centre in centres),
        "longitude": ",".join(str(centre.longitude) for centre in centres),
        "start_date": f"{period.first_year}-01-01",
        "end_date": f"{period.last_year}-12-31",
        "daily": DAILY_VARIABLE,
        "timezone": "GMT",
    }
    return f"{ARCHIVE}?{urlencode(query)}"


def yearly_climate(payload: Any, names: Sequence[str]) -> pd.DataFrame:
    """Mean temperature per year for each named place."""
    locations = _locations(payload)
    if len(locations) != len(names):
        raise ValueError(
            f"Open-Meteo answered for {len(locations)} locations, expected {len(names)}"
        )
    yearly = [
        _yearly(location, name) for location, name in zip(locations, names, strict=True)
    ]
    return pd.concat(yearly, ignore_index=True)


def _locations(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if "daily" not in payload:
        reason = payload.get("reason", "no daily values")
        raise ValueError(f"Open-Meteo did not return temperatures: {reason}")
    return [payload]


def _yearly(location: dict[str, Any], name: str) -> pd.DataFrame:
    daily = pd.DataFrame(location["daily"])
    years = pd.to_datetime(daily["time"]).dt.year.rename("year")
    readings = daily.groupby(years).agg(temperature=(DAILY_VARIABLE, "mean"))
    labelled = readings.reset_index().assign(region=name)
    return labelled[["year", "region", "temperature"]]
