from abc import abstractmethod
from typing import List, Sequence, Tuple

import numpy as np
from mod.Grid import Grid
from mod.OccupancyMap import OccupancyMap
from torch.utils.data import Dataset

from deepflow.utils import RowColumnPair, Window


class PeopleFlowDataset(Dataset):
    def __init__(
        self,
        occupancy: OccupancyMap,
        dynamics: Grid,
        window_size: int,
        output_channels: int,
    ) -> None:
        super().__init__()
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = self.indeces[index]
        return self.get_by_center(center)

    def get_by_center(
        self,
        center: RowColumnPair,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        input = np.asarray(
            self.occupancy.crop(self.window.corners(center)),
            "float",
        )
        assert (
            input.shape[0] == self.window.size
            and input.shape[1] == self.window.size
        )
        output, mask = self._make_dyn_matrix(center)
        return (np.expand_dims(input, 0), output, mask)

    @abstractmethod
    def _make_dyn_matrix(
        self, center: RowColumnPair
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def get_indeces(self) -> Sequence[RowColumnPair]:
        indices: List[RowColumnPair] = []
        for p in self.dynamics.cells.keys():
            indices += self.window.indeces(p)
        return indices


class DiscreteDirectionalDataset(PeopleFlowDataset):
    def __init__(
        self, occupancy: OccupancyMap, dynamics: Grid, window_size: int = 32
    ) -> None:
        super().__init__(occupancy, dynamics, window_size, output_channels=8)

    def _make_dyn_matrix(
        self, center: RowColumnPair
    ) -> Tuple[np.ndarray, np.ndarray]:
        output = np.zeros(
            (self.window.size, self.window.size, self.output_channels), "float"
        )
        mask = np.zeros((self.window.size, self.window.size), "bool")
        for index in self.window.indeces(center):
            if index in self.dynamics.cells:
                cell = self.dynamics.cells[index]
                prob: Sequence[float] = [
                    b["probability"] for b in cell.bins.values()
                ]
                output[index[0] - center[0], index[1] - center[1], :] = prob
                mask[index[0] - center[0], index[1] - center[1]] = True
        return (output, mask)


class ConditionalDirectionalDataset(PeopleFlowDataset):
    def __init__(
        self, occupancy: OccupancyMap, dynamics: Grid, window_size: int = 32
    ) -> None:
        super().__init__(occupancy, dynamics, window_size, output_channels=64)
