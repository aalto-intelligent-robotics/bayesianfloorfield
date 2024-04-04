# %% Some magic
# ! %load_ext autoreload
# ! %autoreload 2
# %% Imports

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from directionalflow.evaluation import extract_tracks_from_grid
from directionalflow.nets import DiscreteDirectional
from directionalflow.utils import Window, estimate_dynamics
from mod import grid, models
from mod.occupancy import OccupancyMap
from mod.utils import XYCoords

sys.modules["Grid"] = grid
sys.modules["Models"] = models

# Change BASE_PATH to the folder where data and models are located
BASE_PATH = Path("/mnt/hdd/datasets/ATC/")

MAP_METADATA = BASE_PATH / "localization_grid.yaml"
GRID_TRAIN_DATA = (
    BASE_PATH
    / "models"
    / "bayes"
    / "20121114"
    / "discrete_directional_20121114_3121209.p"
)
GRID_TEST_DATA = (
    BASE_PATH
    / "models"
    / "bayes"
    / "20121118"
    / "discrete_directional_20121118_8533469.p"
)

NET_EPOCHS = 120
NET_WINDOW_SIZE = 64
NET_SCALE_FACTOR = 20

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PLOT_DPI = 800

grid_test: grid.Grid = pickle.load(open(GRID_TEST_DATA, "rb"))
grid_train: grid.Grid = pickle.load(open(GRID_TRAIN_DATA, "rb"))
occupancy = OccupancyMap.from_yaml(MAP_METADATA)
occupancy.origin = XYCoords(-60, -40)
tracks = extract_tracks_from_grid(grid_test)

net_id_string = f"_w{NET_WINDOW_SIZE}_s{NET_SCALE_FACTOR}_t_{NET_EPOCHS}"
net = DiscreteDirectional(NET_WINDOW_SIZE)
net.load_weights(f"models/people_net{net_id_string}.pth")
window = Window(NET_WINDOW_SIZE * NET_SCALE_FACTOR)

# %%

evaluation_ids = range(len(tracks))
# evaluation_ids = [5101]
# evaluation_ids = [random.randint(0, len(tracks) - 1) for i in range(10)]
# evaluation_ids = ids

like = 0.0
matches = 0
dynamics = estimate_dynamics(
    net,
    occupancy,
    scale=1,
    net_scale=NET_SCALE_FACTOR,
    device=DEVICE,
    batch_size=5,
)

np.save(f"maps/map_atc_{net_id_string}.npy", dynamics)
