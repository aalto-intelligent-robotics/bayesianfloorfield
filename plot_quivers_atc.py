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
BASE_PATH = Path("/mnt/hdd/datasets/ATC/")
ATC_DAYS = {1: "20121114", 2: "20121118"}
TRAIN_DAY = 1
TEST_DAY = 2

# Change NET_MAP_PATH to the folder where data and models are located
NET_MAP_PATH = Path("maps")
NET_EPOCHS = 120
NET_WINDOW_SIZE = 64
NET_SCALE_FACTOR = 20

ALPHA = 5
RUN_SUFFIX = ""

ATC_TRAIN_DAY = ATC_DAYS[TRAIN_DAY]
ATC_TEST_DAY = ATC_DAYS[TEST_DAY]
ATC_TRAIN_FILES = (BASE_PATH / "models" / "bayes" / ATC_TRAIN_DAY).glob(
    f"discrete_directional_{ATC_TRAIN_DAY}_*.p"
)
ATC_TEST_FILES = (BASE_PATH / "models" / "bayes" / ATC_TEST_DAY).glob(
    f"discrete_directional_{ATC_TEST_DAY}_*.p"
)
MAP_METADATA = BASE_PATH / "localization_grid.yaml"
GRID_BAYES_DATA = {
    int(file.stem.split("_")[-1]): file for file in sorted(ATC_TRAIN_FILES)
}
GRID_TEST_DATA = sorted(ATC_TEST_FILES)[-1]

PLOT_DPI = 800

grid_test: grid.Grid = pickle.load(open(GRID_TEST_DATA, "rb"))
occupancy = OccupancyMap.from_yaml(MAP_METADATA)
occupancy.origin = XYCoords(-60, -40)
net_id_string = f"_w{NET_WINDOW_SIZE}_s{NET_SCALE_FACTOR}_t_{NET_EPOCHS}"
net_map = np.load(NET_MAP_PATH / f"map_atc{net_id_string}.npy")

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

occupancy.origin = XYCoords(-18, 0)

grid_bayes = pickle.load(open(GRID_BAYES_DATA[0], "rb"))
grid.assign_cell_priors_to_grid(
    grid=grid_bayes, priors=net_prior, alpha=ALPHA, add_missing_cells=True
)
show_discrete_directional(grid_bayes, occupancy, dpi=3000, save_name="atc_0")

grid_bayes = pickle.load(open(GRID_BAYES_DATA[10000], "rb"))
show_discrete_directional(
    grid_bayes,
    occupancy,
    dpi=3000,
    save_name="atc_10k_noprior",
)

grid.assign_cell_priors_to_grid(
    grid=grid_bayes, priors=net_prior, alpha=ALPHA, add_missing_cells=True
)
show_discrete_directional(grid_bayes, occupancy, dpi=3000, save_name="atc_10k")

grid_bayes = pickle.load(open(GRID_TEST_DATA, "rb"))
show_discrete_directional(grid_bayes, occupancy, dpi=3000, save_name="atc_gt")
