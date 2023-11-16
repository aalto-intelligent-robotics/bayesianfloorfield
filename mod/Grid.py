import collections
import logging
from copy import deepcopy

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
ch.setFormatter(formatter)


class Grid:
    def __init__(self, origin=None, resolution=None, model=None):
        # setting logging
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.DEBUG)
        self.log.addHandler(ch)

        # basic parameters
        self.resolution = resolution
        self.origin = origin
        self.dimensions = [0, 0]
        self.max_dimensions = [0, 0]
        self.model = model

        self.cells = collections.defaultdict(None)
        self.total_count = 0

    def add_data(self, data):
        for _, row in data.iterrows():
            col_index = int((row["x"] - self.origin[0]) / self.resolution)
            row_index = int((row["y"] - self.origin[1]) / self.resolution)
            key = (row_index, col_index)
            if self.dimensions[0] < row_index:
                self.dimensions[0] = row_index
            if self.dimensions[1] < col_index:
                self.dimensions[1] = col_index
            if key not in self.cells:
                self.cells[key] = self.model(
                    index=(row_index, col_index),
                    coords=(
                        row_index * self.resolution + self.origin[1],
                        col_index * self.resolution + self.origin[0],
                    ),
                    resolution=self.resolution,
                )
            self.total_count = self.total_count + 1
            self.cells[key].add_data(row)

    def update_model(self):
        for he in self.cells:
            self.cells[he].update_model(self.total_count)


def move_grid_origin(grid, new_origin):
    new_grid = deepcopy(grid)
    old_origin = new_grid.origin
    delta_origin = (
        old_origin[0] - new_origin[0],
        old_origin[1] - new_origin[1],
    )
    for cell in new_grid.cells.values():
        cell.coords["x"] = cell.coords["x"] - delta_origin[0]
        cell.coords["y"] = cell.coords["y"] - delta_origin[1]
    new_grid.origin = new_origin
    return new_grid


def assign_prior_to_grid(grid, priors, alpha):
    for cell in grid.cells.values():
        cell.assign_prior(priors, alpha)
