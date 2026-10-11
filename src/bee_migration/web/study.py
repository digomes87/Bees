"""A whole study as plain JSON, for interfaces that cannot hold Python objects."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any

import pandas as pd

from bee_migration.analysis.correlation import CORRELATED_COLUMNS, Correlations
from bee_migration.analysis.temporal import TemporalPatterns
from bee_migration.climate.observations import MEASUREMENTS, Observations
from bee_migration.climate.period import DEFAULT_PERIOD
from bee_migration.climate.region import BRAZILIAN_REGIONS
from bee_migration.climate.simulation import DEFAULT_SEED, ClimateSimulation
from bee_migration.prediction.catalog import DEFAULT_CANDIDATES, QUICK_CANDIDATES
from bee_migration.prediction.features import DEFAULT_TARGET, FeatureTable
from bee_migration.prediction.training import (
    DEFAULT_HOLDOUT_YEARS,
    TOP_FEATURES,
    Leaderboard,
    MigrationPredictor,
)

LISTED_PAIRS = 5
LISTED_FEATURES = min(8, TOP_FEATURES)
DECIMALS = 4


@dataclass(frozen=True)
class StudyRequest:
    """The choices a visitor makes before running a study."""

    seed: int = DEFAULT_SEED
    holdout_years: int = DEFAULT_HOLDOUT_YEARS
    target: str = DEFAULT_TARGET
    thorough: bool = False

    @classmethod
    def parsed(cls, document: str) -> StudyRequest:
        settings = json.loads(document)
        unknown = set(settings) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown study settings: {', '.join(sorted(unknown))}")
        return cls(**settings)

    def predictor(self) -> MigrationPredictor:
        candidates = DEFAULT_CANDIDATES if self.thorough else QUICK_CANDIDATES
        return MigrationPredictor(candidates, self.holdout_years)


def run_study(document: str) -> str:
    """Run the study described by a JSON request and answer in JSON."""
    return json.dumps(simulated_study(StudyRequest.parsed(document)))


def simulated_study(request: StudyRequest) -> dict[str, Any]:
    simulation = ClimateSimulation(BRAZILIAN_REGIONS, DEFAULT_PERIOD)
    observations = simulation.observations(request.seed)
    return {
        "request": asdict(request),
        **described(observations),
        **forecasted(observations, request),
    }


def described(observations: Observations) -> dict[str, Any]:
    """What the observations show: values, trends and correlations."""
    correlations = Correlations(observations)
    trends = TemporalPatterns(observations).trends()
    pairs = correlations.notable_pairs()[:LISTED_PAIRS]
    return {
        "regions": observations.region_names(),
        "observations": _records(
            observations.table()[["year", "region", *MEASUREMENTS]]
        ),
        "trends": _records(trends.reset_index(names="region")),
        "correlations": {
            "variables": CORRELATED_COLUMNS,
            "matrix": correlations.matrix().round(DECIMALS).to_numpy().tolist(),
        },
        "pairs": [pair.record() for pair in pairs],
    }


def forecasted(observations: Observations, request: StudyRequest) -> dict[str, Any]:
    """How well each model predicts the held-out years, and what drives it."""
    features = FeatureTable(observations, request.target)
    leaderboard = request.predictor().train(observations, request.target)
    predictions = leaderboard.forecasts(features.split(request.holdout_years).test)
    rows = pd.concat([features.holdout(request.holdout_years), predictions], axis=1)
    return {
        "models": _records(leaderboard.metrics().reset_index(names="model")),
        "best": leaderboard.best().name,
        "forecasts": _records(rows),
        "importance": _importance(leaderboard),
    }


def _importance(leaderboard: Leaderboard) -> list[dict[str, Any]]:
    ranked = leaderboard.best().importance.sort_values(ascending=False)
    listed = ranked.head(LISTED_FEATURES).rename("importance")
    return _records(listed.rename_axis("feature").reset_index())


def _records(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as JSON-safe dictionaries (numpy scalars become plain numbers)."""
    return json.loads(table.to_json(orient="records", double_precision=DECIMALS + 2))
