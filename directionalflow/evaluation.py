from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from PIL import Image
from tqdm import tqdm

from directionalflow.data import get_directional_prob
from directionalflow.nets import DiscreteDirectional
from directionalflow.utils import Direction, OccupancyMap, Window
from mod.Grid import Grid


def convert_matlab(track_path: Path) -> np.ndarray:
    return sio.loadmat(track_path, squeeze_me=True)["dataTrajectoryNoIDCell"]


def convert_grid(grid: Grid) -> np.ndarray:
    columns = ["time", "person_id", "x", "y"]
    data = pd.DataFrame(columns=columns)
    for cell in tqdm(grid.cells.values()):
        data = data.append(cell.data[columns])

    data_grouped = data.groupby("person_id")
    usable_tracks = data_grouped.size().where(data_grouped.size() > 1).dropna()
    data_list = np.empty(len(usable_tracks), dtype=object)
    i = 0
    for _, person_data in tqdm(data_grouped):
        if len(person_data) > 1:
            person_data.sort_values("time", inplace=True)
            data_list[i] = person_data[["x", "y"]].to_numpy().T / 1000
            i += 1
    return data_list


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


def track_likelihood_net(
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
        dir = Direction.from_points((col, -row), (next_col, -next_row))
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


def track_likelihood_model(
    track: np.ndarray,
    occupancy: OccupancyMap,
    grid: Grid,
    ignore_missing: bool = False,
) -> float:
    like = 0.0
    occupancy_top = int(occupancy.map.size[1] * occupancy.resolution)
    delta_origins = [
        occupancy.origin[1] - grid.origin[1] / grid.resolution,
        occupancy.origin[0] - grid.origin[0] / grid.resolution,
    ]
    missing = 0
    matches = 0
    for i in range(track.shape[1] - 1):
        row, col = track[0:2, i]
        next_row, next_col = track[0:2, i + 1]
        dir = Direction.from_points((col, -row), (next_col, -next_row))
        grid_row = occupancy_top - 1 - (track[2, i] + delta_origins[0])
        grid_col = track[3, i] + delta_origins[1]
        if (grid_row, grid_col) in grid.cells:
            cell = grid.cells[(grid_row, grid_col)]
            pred = get_directional_prob(cell.bins)
            like += pred[dir]
            matches += 1
        elif not ignore_missing:
            like += 1 / 8
            missing += 1
            matches += 1
    if missing:
        print(f"missing: {missing}")
    return like / matches if matches else like
