from abc import abstractmethod
from typing import Sequence, Tuple

import numpy as np
from mod.Grid import Grid
from mod.OccupancyMap import OccupancyMap
from torch.utils.data import Dataset

from deepflow.utils import Direction, RowColumnPair, Window


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

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        center = self.indeces[index]
        return self.get_by_center(center)

    def get_by_center(
        self,
        center: RowColumnPair,
    ) -> Tuple[np.ndarray, np.ndarray]:
        input = np.asarray(
            self.occupancy.crop(self.window.corners(center)),
            "float",
        )
        assert (
            input.shape[0] == self.window.size
            and input.shape[1] == self.window.size
        )
        output = self._dynamics(center)
        return (np.expand_dims(input, 0), output)

    def get_indeces(self) -> Sequence[RowColumnPair]:
        return list(self.dynamics.cells.keys())

    @abstractmethod
    def _dynamics(self, center: RowColumnPair) -> np.ndarray:
        raise NotImplementedError


class DiscreteDirectionalDataset(PeopleFlowDataset):
    def __init__(
        self, occupancy: OccupancyMap, dynamics: Grid, window_size: int = 32
    ) -> None:
        super().__init__(occupancy, dynamics, window_size, output_channels=8)

    def _dynamics(self, center: RowColumnPair) -> np.ndarray:
        bins = self.dynamics.cells[center].bins
        prob: Sequence[float] = [
            bins[d.rad()]["probability"] for d in Direction
        ]
        return np.array(prob)


class ConditionalDirectionalDataset(PeopleFlowDataset):
    def __init__(
        self, occupancy: OccupancyMap, dynamics: Grid, window_size: int = 32
    ) -> None:
        super().__init__(occupancy, dynamics, window_size, output_channels=64)

    def _dynamics(self, center: RowColumnPair) -> np.ndarray:
        bins = self.dynamics.cells[center].model
        prob = [
            bins[(sd.rad(), ed.rad())]["probability"]
            for sd in Direction
            for ed in Direction
        ]
        return np.array(prob)
