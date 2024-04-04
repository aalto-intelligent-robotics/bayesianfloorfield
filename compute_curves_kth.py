# %% Some magic
# ! %load_ext autoreload
# ! %autoreload 2
# %% Imports

import json
import logging
import pickle
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bff.evaluation import (
    append_cell_indeces_to_track,
    evaluate_likelihood_iterations,
    extract_tracks_from_grid,
)
from mod import grid, models
from mod.occupancy import OccupancyMap
from mod.utils import TDRC_from_XY
from mod.visualisation import show_occupancy

logging.basicConfig(level=logging.INFO)

sys.modules["Grid"] = grid
sys.modules["Models"] = models

# Change BASE_PATH to the folder where data and models are located
BASE_PATH = Path("/mnt/hdd/datasets/KTH_track/")

# Change NET_MAP_PATH to the folder where data and models are located
NET_MAP_PATH = Path("maps")
NET_EPOCHS = 120
NET_WINDOW_SIZE = 64
NET_SCALE_FACTOR = 20

ALPHA = 5
RUN_SUFFIX = ""

KTH_FILES = (BASE_PATH / "models" / "bayes").glob(
    "discrete_directional_kth_*.p"
)
MAP_METADATA = BASE_PATH / "map.yaml"
GRID_BAYES_DATA = {
    int(file.stem.split("_")[-1]): file for file in sorted(KTH_FILES)
}
GRID_TEST_DATA = sorted(GRID_BAYES_DATA.items())[-1][1]

PLOT_DPI = 800

grid_test: grid.Grid = pickle.load(open(GRID_TEST_DATA, "rb"))
occupancy = OccupancyMap.from_yaml(MAP_METADATA)
net_id_string = f"_w{NET_WINDOW_SIZE}_s{NET_SCALE_FACTOR}_t_{NET_EPOCHS}"
net_map = np.load(NET_MAP_PATH / f"map_kth{net_id_string}.npy")
tracks = extract_tracks_from_grid(grid_test)

# %%

plt.figure(dpi=PLOT_DPI)
show_occupancy(occupancy)

# ids = range(len(tracks))
# ids = [5101]
ids = [random.randint(0, len(tracks) - 1) for i in range(10)]

for id in ids:
    t: np.ndarray = tracks[id]

    X = t[0, :]
    Y = t[1, :]

    plt.plot(X, Y, linewidth=0.5)
    # plt.scatter(X, Y, s=0.05)

# %%

plt.figure(dpi=PLOT_DPI)
plt.imshow(
    occupancy.map,
    extent=(
        occupancy.origin[0] - grid_test.origin[0],
        occupancy.origin[0]
        - grid_test.origin[0]
        + occupancy.map.size[0] * occupancy.resolution,
        occupancy.origin[1] - grid_test.origin[1],
        occupancy.origin[1]
        - grid_test.origin[1]
        + occupancy.map.size[1] * occupancy.resolution,
    ),
    cmap="gray",
)
plt.grid(True, linewidth=0.1)
plt.xticks(range(0, grid_test.dimensions.column + 1))
plt.yticks(range(0, grid_test.dimensions.row + 1))
for id in ids:
    t = append_cell_indeces_to_track(
        tracks[id], grid_test.origin, grid_test.resolution
    )
    X = t[0, :] - grid_test.origin[0]
    Y = t[1, :] - grid_test.origin[1]
    U = t[-1, :] + grid_test.resolution / 2
    V = t[-2, :] + grid_test.resolution / 2
    plt.plot(X, Y, "x-", markersize=0.1, linewidth=0.1)
    plt.scatter(U, V, s=0.1)

# %%

evaluation_ids = range(len(tracks))
# evaluation_ids = [5101]
# evaluation_ids = [random.randint(0, len(tracks) - 1) for i in range(10)]
# evaluation_ids = ids

# %%

print("Bayesian model (uniform prior)")

uni_bayes_data = evaluate_likelihood_iterations(
    GRID_BAYES_DATA, tracks[evaluation_ids], [1 / 8] * 8, ALPHA
)

with open(
    "curves/KTH_uniprior" + ("_" + RUN_SUFFIX if RUN_SUFFIX else "") + ".json",
    "w",
) as f:
    json.dump(uni_bayes_data, f)

# %%

print("Bayesian model (network prior)")

net_prior = {}
for cell_id, cell in grid_test.cells.items():
    try:
        net_map_index = TDRC_from_XY(
            cell.center,
            occupancy.origin,
            occupancy.resolution,
            num_rows=net_map.shape[0],
        )
        net_prior[cell_id] = net_map[
            net_map_index.row, net_map_index.column, :
        ]
    except ValueError:
        net_prior[cell_id] = [1 / 8] * 8

net_bayes_data = evaluate_likelihood_iterations(
    GRID_BAYES_DATA, tracks[evaluation_ids], net_prior, ALPHA
)

with open(
    f"curves/KTH{net_id_string}"
    + ("_" + RUN_SUFFIX if RUN_SUFFIX else "")
    + ".json",
    "w",
) as f:
    json.dump(net_bayes_data, f)

# %%

print("Traditional model (no prior)")

trad_data = evaluate_likelihood_iterations(
    GRID_BAYES_DATA, tracks[evaluation_ids]
)

with open(
    "curves/KTH_trad" + ("_" + RUN_SUFFIX if RUN_SUFFIX else "") + ".json",
    "w",
) as f:
    json.dump(trad_data, f)
