from abc import abstractmethod
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from mod.Grid import Grid
from mod.OccupancyMap import OccupancyMap
from PIL import Image
from torch.utils.data import Dataset

from deepflow.utils import Direction, RowColumnPair, Window


class _RandomFlipPeopleFlow(torch.nn.Module):
    """Random people flow flipping base class"""

    def __init__(
        self,
        axis: int,
        directions: list[tuple[Direction, Direction]],
        p: float = 0.5,
    ):
        """Init `_RandomFlipPeopleFlow` class

        Args:
            axis (int): axis along which to flip the occupancy image
            directions (list[tuple[Direction, Direction]]): a list of Direction
            pairs. While flipping the dynamics, for each pairs, the first
            Direction will be switched with second.
            p (float, optional): Probability of flipping. Defaults to 0.5.
        """
        assert 0 <= p <= 1
        super().__init__()
        self.p = p
        self.axis = axis
        self.directions = directions
        self._dirs_from = np.concatenate(directions)
        self._dirs_to = np.concatenate([(dB, dA) for dA, dB in directions])

    def forward(
        self, data: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the (occupancy, dynamics) tuple `data` flipped"""
        if torch.rand(1) < self.p:
            occ, dyn = data
            occ = np.flip(occ, axis=self.axis).copy()
            dyn = self._flip_dynamics(dyn)
            data = (occ, dyn)
        return data

    def _flip_dynamics(self, dynamics: np.ndarray) -> np.ndarray:
        """Return `dynamics` with flipped Directions"""
        sz = dynamics.shape
        if sz == (8,):
            dynamics[self._dirs_from] = dynamics[self._dirs_to]
        elif sz == (64,):
            dynamics = dynamics.reshape((8, 8))
            dynamics[self._dirs_from] = dynamics[self._dirs_to]
            dynamics[:, self._dirs_from] = dynamics[:, self._dirs_to]
            dynamics = dynamics.reshape(64)
        else:
            raise NotImplementedError(
                f"Received wrong shape: {sz}.\n"
                "This transformation can only be applied to directional or "
                "conditional directional models"
            )
        return dynamics


class RandomHorizontalFlipPeopleFlow(_RandomFlipPeopleFlow):
    """Performs an horizontal flip on a people flow data sample"""

    def __init__(self, p: float = 0.5):
        """Init `RandomHorizontalFlipPeopleFlow` class

        Args:
            p (float, optional): Probability of flipping. Defaults to 0.5.
        """
        directions = [
            (Direction.NW, Direction.NE),
            (Direction.W, Direction.E),
            (Direction.SW, Direction.SE),
        ]
        super().__init__(axis=2, directions=directions, p=p)


class RandomVerticalFlipPeopleFlow(_RandomFlipPeopleFlow):
    """Performs a vertical flip on a people flow data sample"""

    def __init__(self, p: float = 0.5):
        """Init `RandomVerticalFlipPeopleFlow` class

        Args:
            p (float, optional): Probability of flipping. Defaults to 0.5.
        """
        directions = [
            (Direction.NW, Direction.SW),
            (Direction.N, Direction.S),
            (Direction.NE, Direction.SE),
        ]
        super().__init__(axis=1, directions=directions, p=p)


class RandomRotationPeopleFlow(torch.nn.Module):
    def __init__(
        self, p: float = 0.5, angles_p: Sequence[float] = [1 / 3, 1 / 3, 1 / 3]
    ):
        assert 0 <= p <= 1
        assert len(angles_p) == 3 and np.sum(angles_p) == 1
        super().__init__()
        self.p = p
        self.angles_p = angles_p
        self.axes = (1, 2)

    def forward(
        self, data: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        if torch.rand(1) < self.p:
            num_rotations = np.random.choice([1, 2, 3], p=self.angles_p)
            occ, dyn = data
            occ = np.rot90(occ, k=num_rotations, axes=self.axes).copy()
            dyn = self._rotate_dynamics(dyn, k=num_rotations)
            data = (occ, dyn)
        return data

    def _rotate_dynamics(self, dynamics: np.ndarray, k: int = 1) -> np.ndarray:
        sz = dynamics.shape
        if sz == (8,):
            dynamics = np.take(dynamics, range(-2 * k, 8 - 2 * k), mode="wrap")
        elif sz == (64,):
            dynamics = dynamics.reshape((8, 8))
            dynamics = np.take(
                dynamics, range(-2 * k, 8 - 2 * k), axis=0, mode="wrap"
            )
            dynamics = np.take(
                dynamics, range(-2 * k, 8 - 2 * k), axis=1, mode="wrap"
            )
            dynamics = dynamics.reshape(64)
        else:
            raise NotImplementedError(
                f"Received wrong shape: {sz}.\n"
                "This transformation can only be applied to directional or "
                "conditional directional models"
            )
        return dynamics


class PeopleFlowDataset(Dataset):
    def __init__(
        self,
        occupancy: OccupancyMap,
        dynamics: Grid,
        window_size: int,
        output_channels: int,
        scale: int = 1,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.occupancy = occupancy.binary_map
        self.map_origin = occupancy.origin
        self.map_size = self.occupancy.size
        self.dynamics = dynamics
        self.res_ratio = occupancy.resolution / (dynamics.resolution / 1000)
        self.output_channels = output_channels
        self.window_size = window_size
        self.indeces = self.get_indeces()
        self.scale = scale
        self.transform = transform
        self.window = Window(window_size * scale)

    def __len__(self) -> int:
        return len(self.indeces)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        center = self.indeces[index]
        return self.get_by_center(center)

    def get_by_center(
        self,
        center: RowColumnPair,
    ) -> Tuple[np.ndarray, np.ndarray]:
        center_occupancy = (
            self.map_size[1]
            - int((center[0] + self.map_origin[1]) / self.res_ratio),
            int((center[1] - self.map_origin[0]) / self.res_ratio),
        )
        input = (
            np.asarray(
                self.occupancy.crop(
                    self.window.corners(center_occupancy)
                ).resize(
                    (self.window_size, self.window_size), Image.ANTIALIAS
                ),
                "float",
            )
            / 255.0
        )
        assert (
            input.shape[0] == self.window_size
            and input.shape[1] == self.window_size
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
        scale: int = 1,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            occupancy,
            dynamics,
            window_size,
            output_channels=8,
            scale=scale,
            transform=transform,
        )

    def _dynamics(self, center: RowColumnPair) -> np.ndarray:
        bins = self.dynamics.cells[center].bins
        prob: Sequence[float] = [bins[d.rad]["probability"] for d in Direction]
        return np.array(prob)


class ConditionalDirectionalDataset(PeopleFlowDataset):
    def __init__(
        self,
        occupancy: OccupancyMap,
        dynamics: Grid,
        window_size: int = 32,
        scale: int = 1,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            occupancy,
            dynamics,
            window_size,
            output_channels=64,
            scale=scale,
            transform=transform,
        )

    def _dynamics(self, center: RowColumnPair) -> np.ndarray:
        # Encodes dynamics in order: [(Start_E, End_E), (Start_E, End_NE), ...]
        bins = self.dynamics.cells[center].model
        prob = [
            bins[(sd.rad, ed.rad)]["probability"]
            for sd in Direction
            for ed in Direction
        ]
        return np.array(prob, dtype=np.float32)
