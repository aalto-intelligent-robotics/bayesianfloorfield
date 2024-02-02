import logging

import pandas as pd
import pytest
from pydantic import ValidationError

from mod.grid import Grid, assign_cell_priors_to_grid, assign_prior_to_grid
from mod.models import BayesianDiscreteDirectional, Cell
from mod.utils import RCCoords, XYCoords


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
    grid = Grid(resolution=1, origin=XYCoords(-40, -40), model=Cell)
    grid.add_data(sample_data)
    assert sum([len(cell.data.index) for cell in grid.cells.values()]) == 9
    assert grid.total_count == 9
    assert grid.dimensions != (0, 0)


def test_grid_update_model(sample_data: pd.DataFrame) -> None:
    grid = Grid(resolution=1, origin=XYCoords(-40, -40), model=Cell)
    grid.add_data(sample_data)
    grid.update_model()
    assert grid.cells[(RCCoords(17, 79))].probability == 3 / 9
    assert grid.cells[(RCCoords(47, 30))].probability == 2 / 9
    assert grid.cells[(RCCoords(22, 74))].probability == 3 / 9
    assert grid.cells[(RCCoords(47, 29))].probability == 1 / 9


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


def test_grid_assign_cell_prior(bayesian_grid: Grid) -> None:
    alpha = 2
    priors = {
        RCCoords(0, 0): [1 / 4] * 2 + [1 / 12] * 6,
        RCCoords(0, 1): [1 / 2] * 1 + [1 / 14] * 7,
    }
    assign_cell_priors_to_grid(bayesian_grid, priors=priors, alpha=alpha)
    assert all(
        [
            isinstance(cell, BayesianDiscreteDirectional)
            and cell.alpha == alpha
            and cell.priors == priors[cell.index]
            for cell in bayesian_grid.cells.values()
        ]
    )


def test_grid_assign_cell_prior_partial(bayesian_grid: Grid) -> None:
    alpha = 2
    priors = {RCCoords(0, 0): [1 / 4] * 2 + [1 / 12] * 6}
    assign_cell_priors_to_grid(bayesian_grid, priors=priors, alpha=alpha)
    cell1 = bayesian_grid.cells[RCCoords(0, 0)]
    cell2 = bayesian_grid.cells[RCCoords(0, 1)]
    assert isinstance(cell1, BayesianDiscreteDirectional)
    assert cell1.alpha == alpha
    assert cell1.priors == priors[RCCoords(0, 0)]
    assert isinstance(cell2, BayesianDiscreteDirectional)
    assert cell2.alpha == 0
    assert cell2.priors == [1 / 8] * 8


def test_grid_assign_cell_prior_missing(
    bayesian_grid: Grid, caplog: pytest.LogCaptureFixture
) -> None:
    alpha = 2
    priors = {RCCoords(0, 1): [1 / 4] * 2 + [1 / 12] * 6}
    with caplog.at_level(logging.WARNING):
        assign_cell_priors_to_grid(bayesian_grid, priors=priors, alpha=alpha)
        assert len(caplog.records) == 0

    priors = {RCCoords(0, 2): [1 / 4] * 2 + [1 / 12] * 6}
    with caplog.at_level(logging.WARNING):
        assign_cell_priors_to_grid(bayesian_grid, priors=priors, alpha=alpha)
        assert len(caplog.records) == 1
