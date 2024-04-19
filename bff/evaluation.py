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
    """Returns the tracks extracted from the KTH dataset as `[tracks[x, y]]`"""
    return sio.loadmat(track_path, squeeze_me=True)["dataTrajectoryNoIDCell"]


def extract_tracks_from_grid(
    grid: Grid, with_grountruth: bool = False
) -> np.ndarray:
    """Extract the trajectory data from an MoD.

    Args:
        grid (Grid): The MoD to extract trajectory data from.
        with_groundtruth (bool): Whether to append the groundtruth bin, cell
        row, and cell columns to the trajectory data. Defaults to False.

    Returns:
        np.ndarray: `[tracks[x, y, motion_angle]]` if
        `with_groundtruth = False`, otherwise
        `[tracks[x, y, motion_angle, bin, cell_row, cell_col]]`
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
    """Converts track data to pixel coordinates in an occupancy map.

    Args:
        track (np.ndarray): The track data to be converted.
        occupancy (OccupancyMap): The occupancy map used to convert track data.

    Returns:
        np.ndarray: The pixel coordinates corresponding to the input track data
        on the given occupancy map.
    """
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
    """Append cell indices to a track. The cell indices are computed based on
    the grid origin and grid resolution.

    Args:
        track (np.ndarray): The track data to which the cell indices are to be
        appended.
        grid_origin (XYCoords): The origin coordinates of the grid.
        grid_resolution (PositiveFloat): The resolution of the grid. It
        represents the length of one side of a cell in the grid in meters.

    Returns:
        np.ndarray: The track data with appended cell indices as
        `[tracks[x, y, motion_angle, cell_row, cell_col]]`.
    """
    cell_indeces = np.empty((2, track.shape[1]))
    for i, point in enumerate(track.T):
        cell = RC_from_XY(
            XYCoords(point[0], point[1]), grid_origin, grid_resolution
        )
        cell_indeces[0, i] = cell.row
        cell_indeces[1, i] = cell.column
    return np.concatenate([track, cell_indeces])


def track_likelihood_from_net(
    pixels: np.ndarray,
    occupancy: OccupancyMap,
    window_size: int,
    scale: int,
    net: DiscreteDirectional,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, int]:
    """Computes the likelihood of a track according to a model given by a
    DiscreteDirectional net.

    Args:
        pixels (np.ndarray): The pixel coordinates to of the track.
        occupancy (OccupancyMap): The occupancy map used to produce the network
        input.
        window_size (int): The size of the window to use as network input.
        scale (int): The scale factor of the network.
        net (DiscreteDirectional): The network to compute likelihood from.
        device (torch.device, optional): The device to perform computations on.
        Defaults to CPU.

    Returns:
        tuple(float, int): A tuple `(likelihood, num_matches)`, containing the
        sum of likelihoods for each pixel in the track and the total number of
        matches. Average likelihood can be obtained as
        `likelihood / num_matches`.
    """
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


def track_likelihood_from_dynamic_map(
    pixels: np.ndarray, dynamics: np.ndarray
) -> tuple[float, int]:
    """Computes the likelihood of a track according to a model given by a
    Map of Dynamics returned by a DiscreteDirectionalNetwork.

    Args:
        pixels (np.ndarray): The pixel coordinates to of the track.
        dynamics (np.ndarray): The MoD to compute likelihood from.

    Returns:
        tuple(float, int): A tuple `(likelihood, num_matches)`, containing the
        sum of likelihoods for each pixel in the track and the total number of
        matches. Average likelihood can be obtained as
        `likelihood / num_matches`.
    """
    like = 0
    matches = 0
    for i in range(pixels.shape[1]):
        row, col = pixels[0:2, i]
        center = (int(row), int(col))
        dir = Direction.from_rad(pixels[2, i])
        like += dynamics[center[0], center[1], dir]
        matches += 1
    return (like, matches)


def track_likelihood_from_grid(
    track: np.ndarray,
    grid: Grid,
    missing_cells: Literal["skip", "uniform", "zero"] = "uniform",
) -> tuple[float, int, int]:
    """Computes the likelihood of a track according to a model given by a
    Map of Dynamics expressed as a Grid. Considers missing cells in the grid
    according to the provided strategy.

    Args:
        track (np.ndarray): The track to compute likelihood for.
        grid (Grid): The grid containing the cell information.
        missing_cells (Literal["skip", "uniform", "zero"], optional): Strategy
        to handle the missing cells, it can be "skip" to ignore missing cells
        completely in the computation, "uniform" to assume uniform probability
        in the missing cells, or "zero" to assume zero probability in the
        missing cells. Defaults to "uniform".

    Returns:
        tuple(float, int, int): A tuple `(likelihood, num_matches, missing)`,
        containing the sum of likelihoods for each pixel in the track, the
        total number of matches, and the number of missing cells in the grid.
    """
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
    """Evaluates the likelihood of groundtruth tracks based on a grid. The grid
    can have a prior assigned.

    Args:
        grid (Union[Path, Grid]): The grid or the path to the grid file.
        groundtruth_tracks (Sequence[np.ndarray]): The sequences of groundtruth
        tracks.
        prior (Union[list[Probability], dict[RCCoords, list[Probability]]],
        optional): The prior to be used, it can be either a list of
        probabilities representing the prior to use for all cells, or a
        dictionary mapping cell indices to their priors. If not provided, no
        prior is used.
        alpha (float, optional): The hyperconcentration parameter representing
        the trust in the prior. If the prior is provided, this argument must
        also be provided.

    Returns:
        dict: A dictionary with the total likelihood, the number of matches,
        the average likelihood, the number of tracks and the number of missing
        points.
    """
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
        t_like, t_matches, t_missing = track_likelihood_from_grid(t, _grid)
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
    """Evaluates the likelihood over multiple grids.

    Args:
        grid_iterations (Mapping[int, Union[Path, Grid]]): A series of grids or
        paths to the grid files, sorted by iteration.
        groundtruth_tracks (np.ndarray): The groundtruth tracks used for
        likelihood evaluation.
        prior (Union[list[Probability], dict[RCCoords, list[Probability]]],
        optional): The prior to be used, it can be either a list of
        probabilities representing the prior to use for all cells, or a
        dictionary mapping cell indices to their priors. If not provided, no
        prior is used.
        alpha (float, optional): The hyperconcentration parameter representing
        the trust in the prior. If the prior is provided, this argument must
        also be provided.

    Returns:
        dict: A dictionary keyed by iteration number, each containing a
        dictionary with the total likelihood, the number of matches, the
        average likelihood, the number of tracks and the number of missing
        points.
    """
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
