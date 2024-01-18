import logging

import pandas as pd
from pydantic import BaseModel, PositiveFloat

from mod.models import BayesianDiscreteDirectional, Cell, Probability
from mod.utils import RCCoords, XY_from_RC, XYCoords

logger = logging.getLogger(__name__)


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
        data_with_row_col = data.assign(
            row=((data.y - self.origin.y) // self.resolution).astype(int),
            col=((data.x - self.origin.x) // self.resolution).astype(int),
        )
        row_cols = data_with_row_col[["row", "col"]].drop_duplicates()
        for _, cell_id in row_cols.iterrows():
            key = RCCoords(cell_id["row"], cell_id["col"])
            if key not in self.cells:
                self.cells[key] = self.model(
                    index=key,
                    coords=XY_from_RC(key, self.origin, self.resolution),
                    resolution=self.resolution,
                )
            cell_data = data.loc[
                (data_with_row_col[["row", "col"]] == cell_id).all(axis=1)
            ]
            self.cells[key].add_data(cell_data)
            self.total_count += len(cell_data.index)

    def update_model(self) -> None:
        for cell in self.cells.values():
            cell.update_model(self.total_count)


def assign_prior_to_grid(
    grid: Grid, prior: list[Probability], alpha: float
) -> None:
    for cell in grid.cells.values():
        assert isinstance(cell, BayesianDiscreteDirectional)
        cell.update_prior(prior, alpha)


def assign_cell_priors_to_grid(
    grid: Grid, priors: dict[RCCoords, list[Probability]], alpha: float
) -> None:
    for cell_id in priors:
        cell = grid.cells.get(cell_id)
        if cell:
            assert isinstance(cell, BayesianDiscreteDirectional)
            cell.update_prior(priors[cell_id], alpha)
        else:
            logger.warning(
                f"Unable to assign prior to non-existing cell {cell_id}"
            )
