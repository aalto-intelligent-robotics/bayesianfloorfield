from typing import List, Sequence, Tuple

import numpy as np
from mod.Grid import Grid
from mod.OccupancyMap import OccupancyMap
from numpy.typing import ArrayLike
from torch.utils.data import Dataset

RowColumnPair = Tuple[int, int]


class Window:
    def __init__(self, size: int) -> None:
        self.size = size

    @property
    def half_size(self) -> int:
        return int(self.size / 2)

    def corners(self, center: RowColumnPair) -> Sequence[int]:
        return (
            center[0] - self.half_size,  # left
            center[1] - self.half_size,  # top
            center[0] + self.half_size,  # right
            center[1] + self.half_size,  # bottom
        )

    def indeces(self, center: RowColumnPair) -> Sequence[RowColumnPair]:
        return [
            (center[0] + row, center[1] + column)
            for row in range(-self.half_size, self.half_size)
            for column in range(-self.half_size, self.half_size)
        ]


class DiscreteDirectionalDataset(Dataset):
    def __init__(
        self,
        occupancy: OccupancyMap,
        dynamics: Grid,
        window_size: int = 32,
        output_channels: int = 8,
    ) -> None:
        self.occupancy = occupancy.binary_map
        self.dynamics = dynamics
        self.map_size = self.occupancy.size
        self.output_channels = output_channels
        self.window = Window(window_size)
        self.indeces = self.get_indeces()

    def __len__(self) -> int:
        return len(self.indeces)

    def __getitem__(
        self, index: int
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        center = self.indeces[index]
        input = np.asarray(
            self.occupancy.crop(self.window.corners(center)),  # type: ignore
            "float",
        )
        assert (
            input.shape[0] == self.window.size
            and input.shape[1] == self.window.size
        )
        output, mask = self._make_dyn_matrix(center)
        return (input, output, mask)

    def _make_dyn_matrix(
        self, center: RowColumnPair
    ) -> Tuple[ArrayLike, ArrayLike]:
        output = np.zeros(
            (self.window.size, self.window.size, self.output_channels), "float"
        )
        mask = np.zeros((self.window.size, self.window.size), "bool")
        for index in self.window.indeces(center):
            if index in self.dynamics.cells:
                cell = self.dynamics.cells[index]  # type: ignore
                prob: Sequence[float] = [
                    b["probability"] for b in cell.bins.values()
                ]
                assert max(prob) != 0
                output[index[0] - center[0], index[1] - center[1], :] = prob
                mask[index[0] - center[0], index[1] - center[1]] = True
        return (output, mask)

    def get_indeces(self) -> Sequence[RowColumnPair]:
        indeces: List[RowColumnPair] = []
        for p in self.dynamics.cells.keys():  # type: ignore
            indeces += self.window.indeces(p)
        return indeces
