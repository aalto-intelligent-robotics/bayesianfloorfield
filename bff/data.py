from abc import abstractmethod
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from bff.utils import Direction, RowColumnPair, Window
from mod.grid import Grid
from mod.models import DiscreteDirectional
from mod.occupancy import OccupancyMap
from mod.utils import RCCoords


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
        else:
            raise NotImplementedError(
                f"Received wrong shape: {sz}.\n"
                "This transformation can only be applied to directional models"
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
    """Performs a random rotation on a people flow data sample"""

    def __init__(
        self, p: float = 0.5, angles_p: Sequence[float] = [1 / 3, 1 / 3, 1 / 3]
    ):
        """Init `RandomRotationPeopleFlow` class

        Args:
            p (float, optional): Probability of applying a rotation. Defaults
            to 0.5.
            angles_p (Sequence[float], optional): A sequence of three
            probabilities associated with the number of 90° rotations. The sum
            of the elements should equal to 1. Defaults to [1/3, 1/3, 1/3].
        """
        assert 0 <= p <= 1
        assert len(angles_p) == 3 and np.sum(angles_p) == 1
        super().__init__()
        self.p = p
        self.angles_p = angles_p
        self.axes = (1, 2)

    def forward(
        self, data: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the (occupancy, dynamics) tuple `data` randomly rotated"""
        if torch.rand(1) < self.p:
            num_rotations = np.random.choice([1, 2, 3], p=self.angles_p)
            occ, dyn = data
            occ = np.rot90(occ, k=num_rotations, axes=self.axes).copy()
            dyn = self._rotate_dynamics(dyn, k=num_rotations)
            data = (occ, dyn)
        return data

    def _rotate_dynamics(self, dynamics: np.ndarray, k: int = 1) -> np.ndarray:
        """Return `dynamics` with Directions rotated `k` times"""
        sz = dynamics.shape
        if sz == (8,):
            dynamics = np.take(dynamics, range(-2 * k, 8 - 2 * k), mode="wrap")
        else:
            raise NotImplementedError(
                f"Received wrong shape: {sz}.\n"
                "This transformation can only be applied to directional models"
            )
        return dynamics


class PeopleFlowDataset(Dataset):
    """A base Dataset to be used for people flow learning"""

    def __init__(
        self,
        occupancy: OccupancyMap,
        dynamics: Grid,
        window_size: int,
        output_channels: int,
        scale: float = 1,
        transform: Optional[Callable] = None,
    ) -> None:
        """Init `PeopleFlowDataset` class

        Args:
            occupancy (OccupancyMap): The occupancy map of the environment.
            dynamics (Grid): The MoD representing the groundtruth people flow.
            window_size (int): The size of the window over the occupancy to use
            to make input for the network.
            output_channels (int): The number of output channels, equal to the
            number of bins in the `dynamics` MoD.
            scale (float, optional): The scale factor. If > 1, a window of size
            `window_size` x `scale` will actually be cropped and then
            downscaled to `window_size`. Defaults to 1.
            transform (Callable, optional): The function or transform to apply
            to an input occupancy window before returning a transformed
            version. Defaults to None.
        """
        super().__init__()
        self.occupancy = occupancy.binary_map
        self.map_origin = occupancy.origin
        self.map_size = self.occupancy.size
        self.dynamics = dynamics
        self.res_ratio = occupancy.resolution / dynamics.resolution
        self.output_channels = output_channels
        self.window_size = window_size
        self.indeces = self.get_indeces()
        self.scale = scale
        self.transform = transform
        self.window = Window(int(window_size * scale))

    def __len__(self) -> int:
        return len(self.indeces)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        center = self.indeces[index]
        return self.get_by_center(center)

    def get_by_center(
        self,
        center: RowColumnPair,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieves data by the center coordinates.

        Args:
            center (RowColumnPair): The center coordinates for data retrieval.

        Returns:
            tuple: The data as `(occupancy_window, groundtruth_dynamics)`
            tuple. `occupancy_window` is a numpy array of size
            `self.window_size` x `self.window_size` centered on `center`, while
            `groundtruth_dynamics` is the groundtruth people flow for `center`
            as a numpy array of `output_channels` probabilities.
        """
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
        """Returns the list of indices for all data in the dataset"""
        return list(self.dynamics.cells.keys())

    @abstractmethod
    def _dynamics(self, center: RowColumnPair) -> np.ndarray:
        """Returns the dynamics for the cell at index `center`."""
        raise NotImplementedError


class DiscreteDirectionalDataset(PeopleFlowDataset):
    """A Dataset to be used for learning people flow from 8-directional Floor
    Field groundtruth
    """

    def __init__(
        self,
        occupancy: OccupancyMap,
        dynamics: Grid,
        window_size: int = 32,
        scale: float = 1,
        transform: Optional[Callable] = None,
    ) -> None:
        """Init `DiscreteDirectionalDataset` class

        Args:
            occupancy (OccupancyMap): The occupancy map of the environment.
            dynamics (Grid): The MoD representing the groundtruth people flow.
            window_size (int): The size of the window over the occupancy to use
            to make input for the network.
            scale (float, optional): The scale factor. If > 1, a window of size
            `window_size` x `scale` will actually be cropped and then
            downscaled to `window_size`. Defaults to 1.
            transform (Callable, optional): The function or transform to apply
            to an input occupancy window before returning a transformed
            version. Defaults to None.
        """
        super().__init__(
            occupancy,
            dynamics,
            window_size,
            output_channels=8,
            scale=scale,
            transform=transform,
        )

    def _dynamics(self, center: RowColumnPair) -> np.ndarray:
        """Returns the dynamics for the cell at index `center`."""
        cell = self.dynamics.cells[RCCoords(center[0], center[1])]
        assert isinstance(cell, DiscreteDirectional)
        return np.array(cell.bins, dtype=np.float32)
