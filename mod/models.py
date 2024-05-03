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
    """Defines a single cell in a 2D grid representing a Map of Dynamics.

    Attributes:
        coords (XYCoords): The origin coordinates of the cell in 2D space.
        index (RCCoords): The row and column indices of the cell within the
        grid.
        resolution (PositiveFloat): The size of the sides of the cell's square.
        probability (Probability): Transition probabilities associated with the
        cell. Default is 0.
        data (pd.DataFrame): Stored data points that fall within the cell.
    """

    coords: XYCoords = Field(frozen=True)
    index: RCCoords = Field(frozen=True)
    resolution: PositiveFloat = Field(frozen=True)
    probability: Probability = 0
    data: pd.DataFrame = Field(default=pd.DataFrame(), repr=False)
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True
    )

    @property
    def observation_count(self) -> int:
        """The number of data points within the cell."""
        return len(self.data.index)

    @cached_property
    def center(self) -> XYCoords:
        """The center coordinates of the cell."""
        return XYCoords(
            x=self.coords.x + self.resolution / 2,
            y=self.coords.y + self.resolution / 2,
        )

    def add_data(self, data: pd.DataFrame) -> None:
        """Adds the data points in `data` to the cell."""
        self.data = pd.concat([self.data, data])

    def compute_cell_probability(self, total_observations: int) -> None:
        """Compute the probability associated with the cell.

        Args:
            total_observations (int): Total observations in the grid to which
            the cell belongs.
        """
        if total_observations:
            self.probability = self.observation_count / total_observations
        elif self.observation_count:
            raise ValueError(
                "Total observations cannot be zero if cell observation count "
                "is not zero"
            )

    def update_model(self, total_observations: int) -> None:
        """Update the cell model by computing the cell probability.

        Args:
            total_observations (int): Total observations in the grid to which
            the cell belongs.
        """
        self.compute_cell_probability(total_observations)


class DiscreteDirectional(Cell):
    """A type of cell that divides the stored data into directional bins,
    allowing for modeling of data with directional components. Corresponds to
    the Floor Field model.

    Attributes:
        bin_count (int): Number of directions, or bins, into which data is
        divided.
        bins (list[Probability]): Precomputed probabilities for each
        directional bin.
    """

    bin_count: int = Field(frozen=True, default=8)
    bins: list[Probability] = Field(default=[], validate_default=True)

    @field_validator("bins")
    @classmethod
    def default_bins(
        cls, v: list[Probability], values: ValidationInfo
    ) -> list[Probability]:
        """Validate the provided bins or create default even bins."""
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
        """Half of the angular coverage of each bin."""
        return np.pi / self.bin_count

    @cached_property
    def directions(self) -> np.ndarray:
        """An array representing the central directions for each bin."""
        return np.arange(0, _2PI, _2PI / self.bin_count)

    def add_data(self, data: pd.DataFrame) -> None:
        """Add data points to the cell, assigning them to a bin based on their
        motion angle.

        Args:
            data (pd.DataFrame): The data to add.
        """
        self.data = pd.concat(
            [
                self.data,
                data.assign(
                    bin=data["motion_angle"].apply(self.bin_from_angle)
                ),
            ]
        )

    def bin_from_angle(self, rad: float) -> int:
        """Calculates the bin index for a given angle."""
        a = rad % _2PI
        for i, d in enumerate(self.directions):
            s = (d - self.half_split) % _2PI
            e = d + self.half_split
            if round(a - s, ndigits=8) % _2PI < round(e - s, ndigits=8) % _2PI:
                return i
        raise ValueError(f"{rad} does not represent an angle")

    def update_bin_probabilities(self) -> None:
        """Update the probabilities of each bin."""
        if not self.data.empty:
            for i in range(self.bin_count):
                self.bins[i] = (
                    len(self.data[self.data.bin == i]) / self.observation_count
                )
            assert np.isclose(
                sum(self.bins), 1
            ), f"Bin probability sum equal to {sum(self.bins)}."

    def update_model(self, total_observations: int) -> None:
        """Updates both the cell probabilities and bin probabilities based on
        the current data added to the cell.

        Args:
            total_observations (int): Total observations in the grid to which
            the cell belongs.
        """
        self.compute_cell_probability(total_observations)
        self.update_bin_probabilities()


class BayesianDiscreteDirectional(DiscreteDirectional):
    """A subclass of `DiscreteDirectional` that extends the bin probability
    calculation with a Bayesian approach. We refer to this as Bayesian Floor
    Field.

    Attributes:
        priors (list[Probability]): Prior probabilities for each directional
        bin.
        alpha (NonNegativeFloat): Concentration hyperparameter to be applied
        during the Bayesian update.
    """

    priors: list[Probability] = Field(default=[], validate_default=True)
    alpha: NonNegativeFloat = 0.0

    @field_validator("priors")
    @classmethod
    def default_priors(
        cls, v: list[Probability], values: ValidationInfo
    ) -> list[Probability]:
        """Validate the provided priors or create default uniform priors."""
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
        """Update the prior probabilities and the alpha hyperparameter.

        Args:
            priors (list[Probability]): The new prior probabilities for each
            bin.
            alpha (float): The new alpha hyperparameter.
        """
        self.priors = priors
        self.alpha = alpha
        self.update_bin_probabilities()

    def update_bin_probabilities(self) -> None:
        """Updates the bin probabilities according to the current priors,
        alpha, and data using a Dirichlet conjugate prior."""
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
        else:
            self.bins = self.priors
