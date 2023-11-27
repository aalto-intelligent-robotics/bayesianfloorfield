from copy import deepcopy

import numpy as np
import pandas as pd
from pydantic import BaseModel, PositiveFloat

from mod.models import BayesianDiscreteDirectional, Cell, RCCoords, XYCoords


class Grid(BaseModel):
    resolution: PositiveFloat
    origin: XYCoords
    model: type[Cell] = Cell
    cells: dict[RCCoords, Cell] = {}
    total_count: int = 0

    @property
    def dimensions(self) -> RCCoords:
        max_r = 0
        max_c = 0
        for c in self.cells:
            if c.row > max_r:
                max_r = c.row
            if c.column > max_c:
                max_c = c.column
        return RCCoords(max_r, max_c)

    def add_data(self, data: pd.DataFrame) -> None:
        for _, row in data.iterrows():
            col_index = int((row["x"] - self.origin[0]) / self.resolution)
            row_index = int((row["y"] - self.origin[1]) / self.resolution)
            key = RCCoords(row_index, col_index)
            if key not in self.cells:
                self.cells[key] = self.model(
                    index=RCCoords(row_index, col_index),
                    coords=XYCoords(
                        col_index * self.resolution + self.origin[0],
                        row_index * self.resolution + self.origin[1],
                    ),
                    resolution=self.resolution,
                )
            self.cells[key].add_data(row)
            self.total_count = self.total_count + 1

    def update_model(self) -> None:
        for cell in self.cells.values():
            cell.update_model(self.total_count)


def move_grid_origin(grid: Grid, new_origin: XYCoords) -> Grid:
    new_grid = deepcopy(grid)
    old_origin = new_grid.origin
    delta_origin = XYCoords(
        old_origin.x - new_origin.x,
        old_origin.y - new_origin.y,
    )
    for cell in new_grid.cells.values():
        cell.coords = XYCoords(
            cell.coords.x - delta_origin.x, cell.coords.y - delta_origin.y
        )
    new_grid.origin = new_origin
    return new_grid


def assign_prior_to_grid(grid: Grid, prior: np.ndarray, alpha: float):
    for cell in grid.cells.values():
        assert isinstance(cell, BayesianDiscreteDirectional)
        cell.update_prior(prior, alpha)


def assign_cell_priors_to_grid(
    grid: Grid, priors: dict[RCCoords, np.ndarray], alpha: float
):
    for cell_id in priors:
        cell = grid.cells.get(cell_id)
        if cell:
            assert isinstance(cell, BayesianDiscreteDirectional)
            cell.update_prior(priors[cell_id], alpha)
