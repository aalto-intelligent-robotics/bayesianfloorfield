from typing import Optional, Sequence, Set, Tuple

import numpy as np
import torch
from mod.OccupancyMap import OccupancyMap

from deepflow.nets import PeopleFlow

RowColumnPair = Tuple[int, int]


def estimate_dynamics(
    net: PeopleFlow, occupancy: OccupancyMap, window_size: int = 32
) -> np.ndarray:
    bin_map = occupancy.binary_map
    size = bin_map.size
    map = np.zeros((size[0], size[1], net.out_channels), "float")

    window = Window(window_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    for row in range(size[0]):
        for column in range(size[1]):
            center = (row, column)
            crop = np.asarray(
                bin_map.crop(window.corners(center)),
                "float",
            )
            input = torch.from_numpy(np.expand_dims(crop, axis=(0, 1)))
            input = input.to(device, dtype=torch.float)
            output = net(input)

            dynamics = (
                output.cpu()
                .detach()
                .numpy()
                .squeeze()[:, window.half_size, window.half_size]
            )
            map[row, column] = dynamics
    return map


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
            center[0] + self.half_size + self.size % 2,  # right
            center[1] + self.half_size + self.size % 2,  # bottom
        )

    def indeces(
        self, center: RowColumnPair, bounds: Optional[Sequence[int]] = None
    ) -> Set[RowColumnPair]:
        indeces = {
            (center[0] + row, center[1] + column)
            for row in range(-self.half_size, self.half_size + self.size % 2)
            for column in range(
                -self.half_size, self.half_size + self.size % 2
            )
        }
        if bounds:
            assert len(bounds) == 4
            min_row, max_row, min_col, max_col = (
                bounds[0],
                bounds[1],
                bounds[2],
                bounds[3],
            )
            indeces = {
                (
                    min(max(i[0], min_row), max_row),
                    min(max(i[1], min_col), max_col),
                )
                for i in indeces
            }
        return indeces
