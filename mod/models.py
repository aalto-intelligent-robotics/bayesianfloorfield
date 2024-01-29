from functools import cached_property
from typing import Annotated

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

Probability = Annotated[float, Field(ge=0, le=1)]


class Cell(BaseModel):
    # TODO previously flipped (x = coords[1], y = coords[0])
    coords: XYCoords = Field(frozen=True)
    index: RCCoords = Field(frozen=True)
    resolution: PositiveFloat = Field(frozen=True)
    probability: Probability = 0
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


class DiscreteDirectional(Cell):
    """
    Floor Field
    """

    bin_count: int = Field(frozen=True, default=8)
    bins: list[Probability] = Field(default=[], validate_default=True)

    @field_validator("bins")
    @classmethod
    def default_bins(
        cls, v: list[Probability], values: ValidationInfo
    ) -> list[Probability]:
        if v is not None and len(v) != 0:
            if len(v) != values.data["bin_count"] or not np.isclose(
                np.sum(v), 1
            ):
                raise ValueError(f"Malformed prior = {v}")
            return v
        else:
            return [
                1 / values.data["bin_count"]
                for _ in range(values.data["bin_count"])
            ]

    @cached_property
    def half_split(self) -> float:
        return np.pi / self.bin_count

    @cached_property
    def directions(self) -> np.ndarray:
        return np.arange(0, _2PI, _2PI / self.bin_count)

    def add_data(self, data: pd.DataFrame) -> None:
        self.data = pd.concat(
            [
                self.data,
                data.assign(
                    bin=data["motion_angle"].apply(self.bin_from_angle)
                ),
            ]
        )

    def bin_from_angle(self, rad: float) -> int:
        a = rad % _2PI
        for i, d in enumerate(self.directions):
            s = (d - self.half_split) % _2PI
            e = d + self.half_split
            if (
                np.float64(a - s).round(8) % _2PI
                < np.float64(e - s).round(8) % _2PI
            ):
                return i
        raise ValueError(f"{rad} does not represent an angle")

    def update_bin_probabilities(self) -> None:
        if not self.data.empty:
            for i in range(self.bin_count):
                self.bins[i] = (
                    len(self.data[self.data.bin == i]) / self.observation_count
                )
            assert np.isclose(
                sum(self.bins), 1
            ), f"Bin probability sum equal to {sum(self.bins)}."

    def update_model(self, total_observations: int) -> None:
        self.compute_cell_probability(total_observations)
        self.update_bin_probabilities()


class BayesianDiscreteDirectional(DiscreteDirectional):
    """
    Bayesian Floor Field
    """

    priors: list[Probability] = Field(default=[], validate_default=True)
    alpha: NonNegativeFloat = 0.0

    @field_validator("priors")
    @classmethod
    def default_priors(
        cls, v: list[Probability], values: ValidationInfo
    ) -> list[Probability]:
        if v is not None and len(v) != 0:
            if len(v) != values.data["bin_count"] or not np.isclose(
                np.sum(v), 1
            ):
                raise ValueError(f"Malformed prior = {v}")
            return v
        else:
            return [
                1 / values.data["bin_count"]
                for _ in range(values.data["bin_count"])
            ]

    def update_prior(self, priors: list[Probability], alpha: float) -> None:
        self.priors = priors
        self.alpha = alpha
        self.update_bin_probabilities()

    def update_bin_probabilities(self) -> None:
        if not self.data.empty:
            for i in range(self.bin_count):
                posterior = self.priors[i] * self.alpha + len(
                    self.data.loc[self.data.bin == i]
                )
                self.bins[i] = posterior / (
                    np.sum(self.priors) * self.alpha + len(self.data.index)
                )
            assert np.isclose(
                sum(self.bins), 1
            ), f"Bin probability sum equal to {sum(self.bins)}."
