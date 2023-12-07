from functools import cached_property

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    ValidationInfo,
    field_validator,
)

from mod.utils import _2PI, RCCoords, XYCoords


class Cell(BaseModel):
    # TODO previously flipped (x = coords[1], y = coords[0])
    coords: XYCoords = Field(frozen=True)
    index: RCCoords = Field(frozen=True)
    resolution: PositiveFloat = Field(frozen=True)
    probability: float = Field(default=0, ge=0, le=1)
    data: pd.DataFrame = Field(default=pd.DataFrame())
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True
    )

    @property
    def observation_count(self) -> int:
        return len(self.data.index)

    @cached_property
    def center(self) -> XYCoords:
        return XYCoords(
            x=self.coords.x + self.resolution / 2,
            y=self.coords.y + self.resolution / 2,
        )

    def add_data(self, data: pd.DataFrame) -> None:
        self.data = pd.concat([self.data, data])

    def compute_cell_probability(self, total_observations: int) -> None:
        if total_observations:
            self.probability = self.observation_count / total_observations
        elif self.observation_count:
            raise ValueError(
                "Total observations cannot be zero if cell observation count "
                "is not zero"
            )

    def update_model(self, total_observations: int) -> None:
        self.compute_cell_probability(total_observations)


class ProbabilityBin(BaseModel):
    probability: float = Field(default=0, ge=0, le=1)
    data: pd.DataFrame = Field(default=pd.DataFrame())
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True
    )


class DiscreteDirectional(Cell):
    """
    Floor Field
    """

    bin_count: int = Field(frozen=True, default=8)
    bins: list[ProbabilityBin] = Field(default=[], validate_default=True)

    @field_validator("bins")
    @classmethod
    def default_bins(
        cls, v: list[ProbabilityBin], values: ValidationInfo
    ) -> list[ProbabilityBin]:
        return (
            v
            if v
            else [
                ProbabilityBin(probability=1 / values.data["bin_count"])
                for _ in range(values.data["bin_count"])
            ]
        )

    @cached_property
    def half_split(self) -> float:
        return np.pi / self.bin_count

    @cached_property
    def directions(self) -> np.ndarray:
        return np.arange(0, _2PI, _2PI / self.bin_count)

    @property
    def bin_probabilities(self) -> list[float]:
        return [bin.probability for bin in self.bins]

    def bin_from_angle(self, rad: float) -> int:
        for i, d in enumerate(self.directions):
            diff = np.abs(d - (rad % _2PI)) - self.half_split
            if diff < 0 or np.isclose(diff, 0):
                return i
        return 0

    def update_bin_data(self) -> None:
        if not self.data.empty:
            data_bins = self.data["motion_angle"].apply(self.bin_from_angle)
            for i in data_bins.drop_duplicates():
                self.bins[i].data = self.data.loc[data_bins == i]

    def update_bin_probabilities(self) -> None:
        if not self.data.empty:
            for bin in self.bins:
                bin.probability = len(bin.data.index) / self.observation_count
            assert np.isclose(
                sum(self.bin_probabilities), 1
            ), f"Bin probability sum equal to {sum(self.bin_probabilities)}."

    def update_model(self, total_observations: int) -> None:
        self.compute_cell_probability(total_observations)
        self.update_bin_data()
        self.update_bin_probabilities()


class BayesianDiscreteDirectional(DiscreteDirectional):
    """
    Bayesian Floor Field
    """

    priors: np.ndarray = Field(default=np.array([]), validate_default=True)
    alpha: NonNegativeFloat = 0.0

    @field_validator("priors")
    @classmethod
    def default_priors(
        cls, v: np.ndarray, values: ValidationInfo
    ) -> np.ndarray:
        if v is not None and v.size != 0:
            if (
                v.size != values.data["bin_count"]
                or not (v >= 0).all()
                or not np.isclose(np.sum(v), 1)
            ):
                raise ValueError(f"Malformed prior = {v}")
            return v
        else:
            return np.array(
                [
                    1 / values.data["bin_count"]
                    for _ in range(values.data["bin_count"])
                ]
            )

    def update_prior(self, priors: np.ndarray, alpha: float) -> None:
        self.priors = priors
        self.alpha = alpha
        self.update_bin_probabilities()

    def update_bin_probabilities(self) -> None:
        if not self.data.empty:
            for i, bin in enumerate(self.bins):
                posterior = self.priors[i] * self.alpha + len(bin.data.index)
                bin.probability = posterior / (
                    np.sum(self.priors) * self.alpha + len(self.data.index)
                )
            assert np.isclose(
                sum(self.bin_probabilities), 1
            ), f"Bin probability sum equal to {sum(self.bin_probabilities)}."
