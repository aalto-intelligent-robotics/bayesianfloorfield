from pathlib import Path
from typing import Literal

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
from mod.utils import RC_from_XY, RCCoords, XYCoords


def extract_tracks_from_matlab(track_path: Path) -> np.ndarray:
    """returns [tracks[x, y]]"""
    return sio.loadmat(track_path, squeeze_me=True)["dataTrajectoryNoIDCell"]


def extract_tracks_from_grid(
    grid: Grid, with_grountruth: bool = False
) -> np.ndarray:
    """
    returns [tracks[x, y, motion_angle]] if with_groundtruth = False (default),
    otherwise [tracks[x, y, motion_angle, bin, cell_row, cell_col]]
    """
    columns = ["time", "person_id", "x", "y", "motion_angle", "bin"]
    data = pd.DataFrame()
    for cell in tqdm(grid.cells.values()):
        cell_data = cell.data[columns].assign(
            cell_row=cell.index.row, cell_col=cell.index.column
        )
        data = pd.concat([data, cell_data])

    data_grouped = data.groupby("person_id")
    data_list = np.empty(len(data_grouped), dtype=object)

    columns_to_export = ["x", "y", "motion_angle"]
    if with_grountruth:
        columns_to_export += ["bin", "cell_row", "cell_col"]

    i = 0
    for _, person_data in tqdm(data_grouped):
        person_data.sort_values("time", inplace=True)
        data_list[i] = person_data[columns_to_export].to_numpy().T
        i += 1
    return data_list


def pixels_from_track(
    track: np.ndarray, occupancy: OccupancyMap
) -> np.ndarray:
    r = occupancy.resolution
    o = occupancy.origin
    s = occupancy.map.size
    pixels = np.empty((3, track.shape[1]))
    pixels[0, :] = s[1] - (track[1, :] - o[1]) / r
    pixels[1, :] = (track[0, :] - o[0]) / r
    pixels[2, :] = track[2, :]
    return pixels


def append_cell_indeces_to_track(track: np.ndarray, grid: Grid) -> np.ndarray:
    cell_indeces = np.empty((2, track.shape[1]))
    for i, point in enumerate(track.T):
        cell = RC_from_XY(
            XYCoords(point[0], point[1]), grid.origin, grid.resolution
        )
        cell_indeces[0, i] = cell.row
        cell_indeces[1, i] = cell.column
    return np.concatenate([track, cell_indeces])


def track_likelihood_net(
    pixels: np.ndarray,
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
    for i in range(pixels.shape[1]):
        row, col = pixels[0:2, i]
        center = (int(row), int(col))
        dir = Direction.from_rad(pixels[2, i])
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
            pred = net(inputs)[0, :].cpu().numpy()
        like += pred[dir]
        matches += 1
    return (like, matches)


def track_likelihood_model(
    track: np.ndarray,
    grid: Grid,
    missing_cells: Literal["skip", "uniform", "zero"] = "uniform",
) -> tuple[float, int, int]:
    assert missing_cells in ["skip", "uniform", "zero"]
    like = 0.0
    missing = 0
    matches = 0
    for i in range(track.shape[1]):
        dir = Direction.from_rad(track[2, i])
        grid_row = track[3, i]
        grid_col = track[4, i]
        if (grid_row, grid_col) in grid.cells:
            cell = grid.cells[RCCoords(grid_row, grid_col)]
            assert isinstance(cell, mod.DiscreteDirectional)
            # if not cell.bins[dir]:
            #    print(f"{cell.index=} {cell.bins=} {dir=}: ZERO")
            like += cell.bins[dir]
            matches += 1
        else:
            missing += 1
            if missing_cells != "skip":
                matches += 1
                if missing_cells == "uniform":
                    like += 1 / 8

    return (like, matches, missing)
