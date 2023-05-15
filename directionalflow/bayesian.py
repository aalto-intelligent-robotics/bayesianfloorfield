from dataclasses import dataclass, field

import numpy as np


@dataclass
class BayesianMoD:
    resolution: float
    origin: list[float]
    alpha: float
    prior: np.ndarray
    observations: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        assert (self.prior >= 0).all(), "Prior contains negative probabilities"
        assert np.isclose(
            np.sum(self.prior, axis=2), 1
        ).all(), "Not all probability distributions in the prior sum to 1"
        assert self.alpha > 0, "Alpha should be positive"
        assert self.resolution > 0, "Resolution should be positive"
        self.observations = np.zeros_like(self.prior)

    def add_observations(self, cell: tuple[int, int, int], n: int = 1) -> None:
        assert n >= 1, "The number of observations should be positive"
        self.observations[cell] += n

    @property
    def posterior(self) -> np.ndarray:
        post = self.prior * self.alpha + self.observations
        return post / np.sum(post, axis=2, keepdims=True)
