# %% Some magic
# ! %load_ext autoreload
# ! %autoreload 2
# %% Imports

from pathlib import Path

import numpy as np
import torch

from bff.nets import DiscreteDirectional
from bff.utils import estimate_dynamics, plot_dir
from mod.occupancy import OccupancyMap
from mod.utils import Direction

# Change BASE_PATH to the folder where data and models are located
BASE_PATH = Path("/mnt/hdd/datasets/ATC/")

MAP_METADATA = BASE_PATH / "localization_grid.yaml"

NET_EPOCHS = 120
NET_WINDOW_SIZE = 64
NET_SCALE_FACTOR = 20

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PLOT_DPI = 300

occupancy = OccupancyMap.from_yaml(MAP_METADATA)

net_id_string = f"_w{NET_WINDOW_SIZE}_s{NET_SCALE_FACTOR}_t_{NET_EPOCHS}"
net = DiscreteDirectional(NET_WINDOW_SIZE)
net.load_weights(f"models/people_net{net_id_string}.pth")

# %% Build the deep prior map

prior = estimate_dynamics(
    net,
    occupancy,
    scale=1,
    net_scale=NET_SCALE_FACTOR,
    device=DEVICE,
    batch_size=5,
)

# %% Save deep prior map

np.save(f"maps/map_atc{net_id_string}.npy", prior)

# %% Load deep prior map

prior = np.load(f"maps/map_atc{net_id_string}.npy")

# %% Visualize

plot_dir(occupancy, prior, Direction.NW, dpi=PLOT_DPI)
plot_dir(occupancy, prior, Direction.NE, dpi=PLOT_DPI)
plot_dir(occupancy, prior, Direction.SW, dpi=PLOT_DPI)
plot_dir(occupancy, prior, Direction.SE, dpi=PLOT_DPI)
