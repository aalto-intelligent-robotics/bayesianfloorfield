from abc import abstractmethod
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from mod.Grid import Grid
from mod.OccupancyMap import OccupancyMap
from torch.utils.data import Dataset

from deepflow.utils import Direction, RowColumnPair, Window, flip_directions


class _RandomFlipPeopleFlow(torch.nn.Module):
    def __init__(
        self,
        axis: int,
        directions: list[tuple[Direction, Direction]],
        p: float = 0.5,
    ):
        assert p >= 0 and p <= 1
        super().__init__()
        self.p = p
        self.axis = axis
        self.directions = directions

    def forward(
        self, data: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        if torch.rand(1) < self.p:
            occ, dyn = data
            occ = np.flip(occ, axis=self.axis).copy()
            dyn = self._flip_dynamics(dyn)
            data = (occ, dyn)
        return data

    def _flip_dynamics(self, dynamics: np.ndarray) -> np.ndarray:
        sz = dynamics.shape
        if sz == (8,):
            for dirA, dirB in self.directions:
                dynamics = flip_directions(dynamics, dirA, dirB)
        elif sz == (64,):
            raise NotImplementedError
        else:
            raise NotImplementedError(
                f"Received wrong shape: {sz}.\n"
                "This transformation can only be applied to directional or "
                "conditional directional models"
            )
        return dynamics


class RandomHorizontalFlipPeopleFlow(_RandomFlipPeopleFlow):
    def __init__(self, p: float = 0.5):
        directions = [
            (Direction.NW, Direction.NE),
            (Direction.W, Direction.E),
            (Direction.SW, Direction.SE),
        ]
        super().__init__(axis=2, directions=directions, p=p)


class RandomVerticalFlipPeopleFlow(_RandomFlipPeopleFlow):
    def __init__(self, p: float = 0.5):
        directions = [
            (Direction.NW, Direction.SW),
            (Direction.N, Direction.S),
            (Direction.NE, Direction.SE),
        ]
        super().__init__(axis=1, directions=directions, p=p)


class PeopleFlowDataset(Dataset):
    def __init__(
        self,
        occupancy: OccupancyMap,
        dynamics: Grid,
        window_size: int,
        output_channels: int,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.occupancy = occupancy.binary_map
        self.dynamics = dynamics
        self.map_size = self.occupancy.size
        self.output_channels = output_channels
        self.window = Window(window_size)
        self.indeces = self.get_indeces()
        self.transform = transform

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
        sample = (np.expand_dims(input, 0), output)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_indeces(self) -> Sequence[RowColumnPair]:
        return list(self.dynamics.cells.keys())

    @abstractmethod
    def _dynamics(self, center: RowColumnPair) -> np.ndarray:
        raise NotImplementedError


class DiscreteDirectionalDataset(PeopleFlowDataset):
    def __init__(
        self,
        occupancy: OccupancyMap,
        dynamics: Grid,
        window_size: int = 32,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            occupancy,
            dynamics,
            window_size,
            output_channels=8,
            transform=transform,
        )

    def _dynamics(self, center: RowColumnPair) -> np.ndarray:
        bins = self.dynamics.cells[center].bins
        prob: Sequence[float] = [
            bins[d.rad()]["probability"] for d in Direction
        ]
        return np.array(prob)


class ConditionalDirectionalDataset(PeopleFlowDataset):
    def __init__(
        self,
        occupancy: OccupancyMap,
        dynamics: Grid,
        window_size: int = 32,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            occupancy,
            dynamics,
            window_size,
            output_channels=64,
            transform=transform,
        )

    def _dynamics(self, center: RowColumnPair) -> np.ndarray:
        bins = self.dynamics.cells[center].model
        prob = [
            bins[(sd.rad(), ed.rad())]["probability"]
            for sd in Direction
            for ed in Direction
        ]
        return np.array(prob)
