from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Drift:
    """Steady change of a quantity from a starting value."""

    start: float
    total_change: float

    def path(self, length: int) -> np.ndarray:
        return np.linspace(self.start, self.start + self.total_change, length)

    def starting_at(self, start: float) -> Drift:
        return replace(self, start=start)


@dataclass(frozen=True)
class Trend:
    """A drift observed with random year-to-year variation."""

    drift: Drift
    spread: float
    floor: float = -np.inf

    def sample(self, length: int, generator: np.random.Generator) -> np.ndarray:
        noise = generator.normal(0.0, self.spread, length)
        return np.maximum(self.drift.path(length) + noise, self.floor)

    def starting_at(self, start: float) -> Trend:
        return replace(self, drift=self.drift.starting_at(start))
