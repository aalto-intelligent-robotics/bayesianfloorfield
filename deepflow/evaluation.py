import numpy as np
import torch
from PIL import Image

from deepflow.utils import Direction, OccupancyMap, Window
from deepflow.nets import DiscreteDirectional


def track2pixels(track: np.ndarray, occupancy: OccupancyMap) -> np.ndarray:
    r = occupancy.resolution
    o = occupancy.origin
    s = occupancy.map.size
    pixels = np.empty((2, track.shape[1]))
    pixels[0, :] = s[1] - (track[1, :] - o[1]) / r
    pixels[1, :] = (track[0, :] - o[0]) / r
    return pixels


def pixels2grid(
    pixels: np.ndarray, grid_resolution: float, occupancy_resolution: float
) -> np.ndarray:
    ratio = occupancy_resolution / grid_resolution
    cell_indeces = (pixels * ratio).astype(int)
    grid = []
    last_cell = np.array([np.nan, np.nan])
    pixels_in_cell = []
    for i, cell in enumerate(cell_indeces.T):
        if all(cell == last_cell):
            pixels_in_cell.append((pixels.T)[i])
        else:
            if len(pixels_in_cell):
                grid.append(
                    np.concatenate(
                        (np.mean(pixels_in_cell, axis=0), last_cell)
                    )
                )
            pixels_in_cell = [(pixels.T)[i]]
            last_cell = cell
    if len(pixels_in_cell):
        grid.append(
            np.concatenate((np.mean(pixels_in_cell, axis=0), last_cell))
        )
    return np.array(grid).T


def track_likelihood(
    track: np.ndarray,
    occupancy: OccupancyMap,
    window_size: int,
    scale: int,
    net: DiscreteDirectional,
    device: torch.device = torch.device("cpu"),
) -> float:
    window = Window(window_size * scale)
    net.to(device)
    net.eval()
    like = 0
    for i in range(track.shape[1] - 1):
        row, col = track[0:2, i]
        next_row, next_col = track[0:2, i + 1]
        center = (int(row), int(col))
        dir = Direction.from_points((col, row), (next_col, next_row))
        crop = (
            np.asarray(
                occupancy.binary_map.crop(window.corners(center)).resize(
                    (window_size, window_size), Image.ANTIALIAS
                ),
                "float",
            )
            / 255.0
        )
        inputs = torch.from_numpy(np.expand_dims(crop, axis=(0, 1))).to(
            device, dtype=torch.float
        )
        with torch.no_grad():
            pred = (
                net(inputs)[0, :, window_size // 2, window_size // 2]
                .cpu()
                .numpy()
            )
        like += pred[dir] / (track.shape[1] - 1)
    return like
