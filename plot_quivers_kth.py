# %% Some magic
# ! %load_ext autoreload
# ! %autoreload 2
# %% Imports

import pickle
import sys
from pathlib import Path

import numpy as np

from mod import grid, models
from mod.occupancy import OccupancyMap
from mod.utils import TDRC_from_XY, XYCoords
from mod.visualisation import show_discrete_directional

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

occupancy.origin = XYCoords(0, 0)

grid_bayes = pickle.load(open(GRID_BAYES_DATA[0], "rb"))
grid.assign_cell_priors_to_grid(
    grid=grid_bayes, priors=net_prior, alpha=ALPHA, add_missing_cells=True
)
show_discrete_directional(grid_bayes, occupancy, dpi=3000, save_name="kth_0")

grid_bayes = pickle.load(open(GRID_BAYES_DATA[10000], "rb"))
show_discrete_directional(
    grid_bayes,
    occupancy,
    dpi=3000,
    save_name="kth_10k_noprior",
)

grid.assign_cell_priors_to_grid(
    grid=grid_bayes, priors=net_prior, alpha=ALPHA, add_missing_cells=True
)
show_discrete_directional(grid_bayes, occupancy, dpi=3000, save_name="kth_10k")

grid_bayes = pickle.load(open(GRID_TEST_DATA, "rb"))
show_discrete_directional(grid_bayes, occupancy, dpi=3000, save_name="kth_gt")
