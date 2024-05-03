import logging

import pandas as pd
from pydantic import BaseModel, Field, PositiveFloat

from mod.models import BayesianDiscreteDirectional, Cell, Probability
from mod.utils import RCCoords, XY_from_RC, XYCoords

logger = logging.getLogger(__name__)


class Grid(BaseModel):
    """Represents a 2D Map of Dynamics as a grid. Each cell of the grid is an
    instance of models.Cell or its subclasses.

    Attributes:
        resolution (PositiveFloat): The size of the sides of the grid's
        squares.
        origin (XYCoords): The coordinate reference for the origin point.
        model (type[Cell]): The type of cell used to fill the grid.
        cells (dict[RCCoords, Cell]): A mapping of grid coordinates to cell
        instances.
        total_count (int): The total number of data items added to grid.
    """

    resolution: PositiveFloat
    origin: XYCoords
    model: type[Cell] = Cell
    cells: dict[RCCoords, Cell] = Field(default={}, repr=False)
    total_count: int = 0

    @property
    def dimensions(self) -> RCCoords:
        """Calculate the extent of the grid in rows and columns.

        Returns:
            RCCoords: The maximum row and column values currently within the
            grid.
        """
        max_r = 0
        max_c = 0
        for c in self.cells:
            if c.row > max_r:
                max_r = c.row
            if c.column > max_c:
                max_c = c.column
        return RCCoords(max_r, max_c)

    def add_data(self, data: pd.DataFrame) -> None:
        """Add positional data to grid points by calculating which cell in the
        grid each dataset entry belongs to.

        Args:
            data (pd.DataFrame): Data with 'x' and 'y' positional entries
        """

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
        """Update all Cell models within the grid. Used after all data has been
        added to the grid to finalize model parameters."""

        for cell in self.cells.values():
            cell.update_model(self.total_count)


def assign_prior_to_grid(
    grid: Grid, prior: list[Probability], alpha: float
) -> None:
    """Assigns a single prior probability value to all cells in a grid.

    Args:
        grid (Grid): The grid to which the prior will be assigned.
        prior (list[Probability]): The prior probabilities to be assigned to
        each cell.
        alpha (float): Concentration hyperparameter for the Dirichlet prior.
    """
    for cell in grid.cells.values():
        assert isinstance(cell, BayesianDiscreteDirectional)
        cell.update_prior(prior, alpha)


def assign_cell_priors_to_grid(
    grid: Grid,
    priors: dict[RCCoords, list[Probability]],
    alpha: float,
    add_missing_cells: bool = False,
) -> None:
    """Assigns individual cell priors to each cell in the grid.

    Args:
        grid (Grid): The grid to which the prior will be assigned.
        priors (dict[RCCoords, list[Probability]]): Mapping of cell coordinates
        to their corresponding prior probabilities.
        alpha (float): Concentration hyperparameter for the Dirichlet prior.
        add_missing_cells (bool, optional): Set to True to add cells that are
        missing in grid but present in priors. Defaults to False.
    """
    for cell_id in priors:
        cell = grid.cells.get(cell_id)
        if cell:
            assert isinstance(cell, BayesianDiscreteDirectional)
            cell.update_prior(priors[cell_id], alpha)
        elif add_missing_cells:
            cell = grid.model(
                index=cell_id,
                coords=XY_from_RC(cell_id, grid.origin, grid.resolution),
                resolution=grid.resolution,
            )
            assert isinstance(cell, BayesianDiscreteDirectional)
            cell.update_prior(priors[cell_id], alpha)
            grid.cells[cell_id] = cell
        else:
            logger.warning(
                f"Unable to assign prior to non-existing cell {cell_id} and "
                f"{add_missing_cells=}"
            )
