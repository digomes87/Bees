from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StudyPeriod:
    """Inclusive range of years covered by a study."""

    first_year: int
    last_year: int

    def __post_init__(self) -> None:
        if self.last_year < self.first_year:
            raise ValueError(
                f"Last year {self.last_year} is before first year {self.first_year}"
            )

    def years(self) -> np.ndarray:
        return np.arange(self.first_year, self.last_year + 1)

    def length(self) -> int:
        return self.last_year - self.first_year + 1


DEFAULT_PERIOD = StudyPeriod(2000, 2023)
