import pandas as pd
import pytest
from pydantic import ValidationError

from mod.grid import Grid, assign_prior_to_grid
from mod.models import BayesianDiscreteDirectional, Cell
from mod.utils import XYCoords


def test_grid_init() -> None:
    grid = Grid(resolution=1, origin=XYCoords(0.2, -30.2), model=Cell)
    assert len(grid.cells) == 0
    assert grid.dimensions == (0, 0)
    assert grid.total_count == 0

    with pytest.raises(ValidationError):
        grid = Grid(resolution=0, origin=XYCoords(0, 0), model=Cell)
    with pytest.raises(ValidationError):
        grid = Grid(resolution=-1, origin=XYCoords(0, 0), model=Cell)


def test_grid_add_data(sample_data: pd.DataFrame) -> None:
    grid = Grid(resolution=1000, origin=XYCoords(-40000, -40000), model=Cell)
    grid.add_data(sample_data)
    assert sum([len(cell.data.index) for cell in grid.cells.values()]) == 9
    assert grid.total_count == 9
    assert grid.dimensions != (0, 0)


def test_grid_assign_prior(bayesian_grid: Grid) -> None:
    alpha = 2
    priors = [1 / 4] * 2 + [1 / 12] * 6
    assign_prior_to_grid(bayesian_grid, prior=priors, alpha=alpha)
    assert all(
        [
            isinstance(cell, BayesianDiscreteDirectional)
            and cell.alpha == alpha
            and cell.priors == priors
            for cell in bayesian_grid.cells.values()
        ]
    )
