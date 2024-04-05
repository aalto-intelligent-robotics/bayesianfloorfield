import pickle
from pathlib import Path
from typing import Literal, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

import mod.models as mod
from bff.nets import DiscreteDirectional
from bff.utils import Direction, OccupancyMap, Window
from mod.grid import (
    Grid,
    PositiveFloat,
    assign_cell_priors_to_grid,
    assign_prior_to_grid,
)
from mod.models import Cell, Probability
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

    def extract_data(cell: Cell) -> pd.DataFrame:
        """Extract data from a single cell"""
        columns = ["time", "person_id", "x", "y", "motion_angle", "bin"]
        cell_data = cell.data[columns].assign(
            cell_row=cell.index.row, cell_col=cell.index.column
        )
        return cell_data

    def process_person(
        person_data: pd.DataFrame, columns_to_export: list[str]
    ) -> np.ndarray:
        """Process data for a single person"""
        person_data.sort_values("time", inplace=True)
        return person_data[columns_to_export].to_numpy().T

    columns_to_export = ["x", "y", "motion_angle"]
    if with_grountruth:
        columns_to_export += ["bin", "cell_row", "cell_col"]

    # Parallel extraction of cell data
    result = Parallel(n_jobs=-1)(
        delayed(extract_data)(cell)
        for cell in tqdm(grid.cells.values(), desc="Extracting data from grid")
    )
    data = pd.concat(result)

    # Group the data by person_id
    data_grouped = data.groupby("person_id")

    # Parallel processing of person data
    data_list = Parallel(n_jobs=-1)(
        delayed(process_person)(person_data, columns_to_export)
        for _, person_data in tqdm(
            data_grouped, desc="Grouping data by trajectory"
        )
    )

    data_array = np.empty(len(data_list), dtype=object)

    for i in range(len(data_list)):
        data_array[i] = data_list[i]

    return data_array


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


def append_cell_indeces_to_track(
    track: np.ndarray, grid_origin: XYCoords, grid_resolution: PositiveFloat
) -> np.ndarray:
    cell_indeces = np.empty((2, track.shape[1]))
    for i, point in enumerate(track.T):
        cell = RC_from_XY(
            XYCoords(point[0], point[1]), grid_origin, grid_resolution
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


def track_likelihood_net_2(
    pixels: np.ndarray, dynamics: np.ndarray
) -> tuple[float, int]:
    like = 0
    matches = 0
    for i in range(pixels.shape[1]):
        row, col = pixels[0:2, i]
        center = (int(row), int(col))
        dir = Direction.from_rad(pixels[2, i])
        like += dynamics[center[0], center[1], dir]
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
            like += cell.bins[dir]
            matches += 1
        else:
            missing += 1
            if missing_cells != "skip":
                matches += 1
                if missing_cells == "uniform":
                    like += 1 / 8

    return (like, matches, missing)


def evaluate_likelihood(
    grid: Union[Path, Grid],
    groundtruth_tracks: Sequence[np.ndarray],
    prior: Optional[
        Union[list[Probability], dict[RCCoords, list[Probability]]]
    ] = None,
    alpha: Optional[float] = None,
) -> dict[str, Union[float, int]]:
    _grid: Grid = (
        pickle.load(open(grid, "rb")) if isinstance(grid, Path) else grid
    )
    if prior:
        assert alpha is not None
        if isinstance(prior, list):
            assign_prior_to_grid(grid=_grid, prior=prior, alpha=alpha)
        elif isinstance(prior, dict):
            assign_cell_priors_to_grid(
                grid=_grid, priors=prior, alpha=alpha, add_missing_cells=True
            )
        else:
            raise ValueError(
                "`prior` should be either a list of probabilities (positive "
                "floats) or a dictionary mapping cell indeces to lists of "
                "probabilities."
            )

    like = 0.0
    matches = 0
    missing = 0

    return_dict = {}

    for track in groundtruth_tracks:
        t = append_cell_indeces_to_track(track, _grid.origin, _grid.resolution)
        t_like, t_matches, t_missing = track_likelihood_model(t, _grid)
        like += t_like
        matches += t_matches
        missing += t_missing

    return_dict = {
        "total_like": like,
        "matches": matches,
        "avg_like": like / matches if matches else 0.0,
        "num_tracks": len(groundtruth_tracks),
        "missing": missing,
    }

    return return_dict


def evaluate_likelihood_iterations(
    grid_iterations: Mapping[int, Union[Path, Grid]],
    groundtruth_tracks: np.ndarray,
    prior: Optional[
        Union[list[Probability], dict[RCCoords, list[Probability]]]
    ] = None,
    alpha: Optional[float] = None,
) -> dict[int, dict]:
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_likelihood)(
            grid_path, groundtruth_tracks, prior, alpha
        )
        for grid_path in tqdm(
            grid_iterations.values(), desc="Evaluating likelihoods"
        )
    )

    # Combining results
    return {k: results[i] for i, k in enumerate(grid_iterations)}
