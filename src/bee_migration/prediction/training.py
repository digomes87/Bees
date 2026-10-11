from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from bee_migration.climate.observations import Observations
from bee_migration.prediction.catalog import DEFAULT_CANDIDATES, Candidate
from bee_migration.prediction.features import (
    DEFAULT_TARGET,
    FeatureTable,
    Sample,
    TrainTestSplit,
)
from bee_migration.prediction.scores import Evaluation

DEFAULT_HOLDOUT_YEARS = 4
TOP_FEATURES = 10


@dataclass(frozen=True)
class ModelResult:
    """A fitted model with its accuracy and the features it relies on."""

    name: str
    pipeline: Pipeline
    evaluation: Evaluation
    importance: pd.Series

    @classmethod
    def trained(
        cls, candidate: Candidate, split: TrainTestSplit, names: Sequence[str]
    ) -> ModelResult:
        pipeline = candidate.tuned(split.train)
        evaluation = Evaluation(split.train.score(pipeline), split.test.score(pipeline))
        return cls(
            candidate.name,
            pipeline,
            evaluation,
            split.train.importance(pipeline, names),
        )

    def test_r2(self) -> float:
        return self.evaluation.test.r2

    def describe(self) -> list[str]:
        return [f"{self.name}:", *self.evaluation.test.describe("Test")]

    def top_features(self) -> list[str]:
        ranked = self.importance.sort_values(ascending=False).head(TOP_FEATURES)
        return [f"  {name}: {value:.4f}" for name, value in ranked.items()]

    def forecast(self, sample: Sample) -> np.ndarray:
        return self.pipeline.predict(sample.features)

    def save(self, directory: Path) -> None:
        slug = self.name.lower().replace(" ", "_")
        joblib.dump(self.pipeline, directory / f"{slug}_model.pkl")


class Leaderboard:
    """Every trained model, comparable by accuracy on the held-out years."""

    def __init__(self, results: Sequence[ModelResult]) -> None:
        if not results:
            raise ValueError("A leaderboard needs at least one trained model")
        self._results = tuple(results)

    def __iter__(self) -> Iterator[ModelResult]:
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def best(self) -> ModelResult:
        return max(self._results, key=ModelResult.test_r2)

    def metrics(self) -> pd.DataFrame:
        rows = {result.name: result.evaluation.labelled() for result in self._results}
        return pd.DataFrame.from_dict(rows, orient="index")

    def importances(self) -> pd.DataFrame:
        return pd.DataFrame(
            {result.name: result.importance for result in self._results}
        )

    def forecasts(self, sample: Sample) -> pd.DataFrame:
        """What every model predicts for the rows of `sample`."""
        return pd.DataFrame(
            {result.name: result.forecast(sample) for result in self._results}
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for result in self._results:
            result.save(directory)


class MigrationPredictor:
    """Trains candidate models to forecast a bee measurement from climate."""

    def __init__(
        self,
        candidates: Sequence[Candidate] = DEFAULT_CANDIDATES,
        holdout_years: int = DEFAULT_HOLDOUT_YEARS,
    ) -> None:
        self._candidates = tuple(candidates)
        self._holdout_years = holdout_years

    def train(
        self, observations: Observations, target: str = DEFAULT_TARGET
    ) -> Leaderboard:
        features = FeatureTable(observations, target)
        split = features.split(self._holdout_years)
        names = features.names()
        return Leaderboard(
            [
                ModelResult.trained(candidate, split, names)
                for candidate in self._candidates
            ]
        )
