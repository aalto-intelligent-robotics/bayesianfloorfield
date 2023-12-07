from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from PIL import Image
from tqdm import tqdm

import mod.models as mod
from directionalflow.nets import DiscreteDirectional
from directionalflow.utils import Direction, OccupancyMap, Window
from mod.grid import Grid
from mod.utils import RCCoords


def convert_matlab(track_path: Path) -> np.ndarray:
    return sio.loadmat(track_path, squeeze_me=True)["dataTrajectoryNoIDCell"]


def convert_grid(grid: Grid) -> np.ndarray:
    columns = ["time", "person_id", "x", "y", "motion_angle"]
    data = pd.DataFrame()
    for cell in tqdm(grid.cells.values()):
        data = pd.concat([data, cell.data[columns]])

    data_grouped = data.groupby("person_id")
    usable_tracks = data_grouped.size().where(data_grouped.size() > 1).dropna()
    data_list = np.empty(len(usable_tracks), dtype=object)
    i = 0
    for _, person_data in tqdm(data_grouped):
        if len(person_data) > 1:
            person_data.sort_values("time", inplace=True)
            person_data[["x", "y"]] /= 1000
            data_list[i] = person_data[["x", "y", "motion_angle"]].to_numpy().T
            i += 1
    return data_list


def track2pixels(track: np.ndarray, occupancy: OccupancyMap) -> np.ndarray:
    r = occupancy.resolution
    o = occupancy.origin
    s = occupancy.map.size
    pixels = np.empty((3, track.shape[1]))
    pixels[0, :] = s[1] - (track[1, :] - o[1]) / r
    pixels[1, :] = (track[0, :] - o[0]) / r
    pixels[2, :] = track[2, :]
    return pixels


def pixels2grid(
    pixels: np.ndarray, grid_resolution: float, occupancy_resolution: float
) -> np.ndarray:
    ratio = occupancy_resolution / grid_resolution
    cell_indeces = (pixels[0:2, :] * ratio).astype(int)
    grid = []
    last_cell = np.array([np.nan, np.nan])
    pixels_in_cell = []
    for i, cell in enumerate(cell_indeces.T):
        if all(cell == last_cell):
            pixels_in_cell.append((pixels[0:2, :].T)[i])
        else:
            if len(pixels_in_cell):
                grid.append(
                    np.concatenate(
                        (np.mean(pixels_in_cell, axis=0), last_cell)
                    )
                )
            pixels_in_cell = [(pixels[0:2, :].T)[i]]
            last_cell = cell
    if len(pixels_in_cell):
        grid.append(
            np.concatenate((np.mean(pixels_in_cell, axis=0), last_cell))
        )
    return np.array(grid).T


def pixels2grid_complete(
    pixels: np.ndarray, grid_resolution: float, occupancy_resolution: float
) -> np.ndarray:
    ratio = occupancy_resolution / grid_resolution
    cell_indeces = (pixels[0:2, :] * ratio).astype(int)
    return np.concatenate([pixels, cell_indeces])


def track_likelihood_net(
    track: np.ndarray,
    occupancy: OccupancyMap,
    window_size: int,
    scale: int,
    net: DiscreteDirectional,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, int]:
    window = Window(window_size * scale)
    net.to(device)
    net.eval()
    like = 0
    matches = 0
    for i in range(track.shape[1]):
        row, col = track[0:2, i]
        center = (int(row), int(col))
        dir = Direction.from_rad(track[2, i])
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
        like += pred[dir]
        matches += 1
    return (like, matches)


def track_likelihood_model(
    track: np.ndarray,
    occupancy: OccupancyMap,
    grid: Grid,
    ignore_missing: bool = False,
) -> tuple[float, int]:
    occupancy_top = int(occupancy.map.size[1] * occupancy.resolution)
    delta_origins = [
        occupancy.origin[1] - grid.origin[1] / grid.resolution,
        occupancy.origin[0] - grid.origin[0] / grid.resolution,
    ]
    like = 0.0
    missing = 0
    matches = 0
    for i in range(track.shape[1]):
        dir = Direction.from_rad(track[2, i])
        grid_row = occupancy_top - 1 - (track[3, i] + delta_origins[0])
        grid_col = track[4, i] + delta_origins[1]
        if (grid_row, grid_col) in grid.cells:
            cell = grid.cells[RCCoords(grid_row, grid_col)]
            assert isinstance(cell, mod.DiscreteDirectional)
            pred = cell.bin_probabilities
            like += pred[dir]
            matches += 1
        elif not ignore_missing:
            like += 1 / 8
            missing += 1
            matches += 1
    # if missing:
    #    print(f"missing: {missing}")
    return (like, matches)
