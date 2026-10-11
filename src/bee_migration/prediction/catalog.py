from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from bee_migration.prediction.features import Sample

MODEL_SEED = 42
VALIDATION_FOLDS = 5
TUNING_SCORE = "neg_mean_squared_error"


@dataclass(frozen=True)
class Candidate:
    """A regression model and the hyperparameters worth trying for it."""

    name: str
    estimator: RegressorMixin
    grid: Mapping[str, Sequence[Any]]

    def tuned(self, sample: Sample) -> Pipeline:
        """Fit on `sample`, choosing hyperparameters on later-year folds."""
        pipeline = Pipeline(
            [("scale", StandardScaler()), ("model", clone(self.estimator))]
        )
        if not self.grid:
            return pipeline.fit(sample.features, sample.target)
        search = GridSearchCV(
            pipeline,
            {f"model__{name}": list(values) for name, values in self.grid.items()},
            cv=TimeSeriesSplit(n_splits=VALIDATION_FOLDS),
            scoring=TUNING_SCORE,
            n_jobs=-1,
        )
        return search.fit(sample.features, sample.target).best_estimator_


# Fixed hyperparameters: no search, for places where training must be quick.
QUICK_CANDIDATES = (
    Candidate("Linear Regression", LinearRegression(), {}),
    Candidate(
        "Random Forest",
        RandomForestRegressor(n_estimators=100, max_depth=5, random_state=MODEL_SEED),
        {},
    ),
    Candidate(
        "Gradient Boosting",
        GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=2, random_state=MODEL_SEED
        ),
        {},
    ),
    Candidate("SVR", SVR(C=10, kernel="linear"), {}),
)

DEFAULT_CANDIDATES = (
    Candidate("Linear Regression", LinearRegression(), {}),
    Candidate(
        "Random Forest",
        RandomForestRegressor(random_state=MODEL_SEED),
        {
            "n_estimators": [100, 300],
            "max_depth": [5, None],
            "min_samples_split": [2, 5],
        },
    ),
    Candidate(
        "Gradient Boosting",
        GradientBoostingRegressor(random_state=MODEL_SEED),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        },
    ),
    Candidate("SVR", SVR(), {"C": [1, 10, 100], "kernel": ["rbf", "linear"]}),
)
