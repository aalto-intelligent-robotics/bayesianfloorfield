# %% Some magic
# ! %load_ext autoreload
# ! %autoreload 2
# %% Imports

import pickle
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from directionalflow.evaluation import (
    append_cell_indeces_to_track,
    extract_tracks_from_grid,
    pixels_from_track,
    track_likelihood_model,
    track_likelihood_net,
)
from directionalflow.nets import DiscreteDirectional
from directionalflow.utils import Window, estimate_dynamics, plot_quivers
from mod import grid, models
from mod.occupancy import OccupancyMap
from mod.visualisation import show_occupancy

sys.modules["Grid"] = grid
sys.modules["Models"] = models

# Change BASE_PATH to the folder where data and models are located
BASE_PATH = Path("/mnt/hdd/datasets/KTH_track/")

MAP_METADATA = BASE_PATH / "map.yaml"
GRID_DATA = (
    BASE_PATH / "models" / "bayes" / "discrete_directional_kth_0421111.p"
)

NET_EPOCHS = 120
NET_WINDOW_SIZE = 64
NET_SCALE_FACTOR = 16

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PLOT_DPI = 800

grid_test: grid.Grid = pickle.load(open(GRID_DATA, "rb"))
occupancy = OccupancyMap.from_yaml(MAP_METADATA)
tracks = extract_tracks_from_grid(grid_test)

net_id_string = f"_w{NET_WINDOW_SIZE}_s{NET_SCALE_FACTOR}_t_{NET_EPOCHS}"
net = DiscreteDirectional(NET_WINDOW_SIZE)
net.load_weights(f"models/people_net{net_id_string}.pth")
window = Window(NET_WINDOW_SIZE * NET_SCALE_FACTOR)

# %%

plt.figure(dpi=PLOT_DPI)
show_occupancy(occupancy)

# ids = range(len(tracks))
# ids = [5101]  # straight track
# ids = [4110]  # track with corners
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
    t = append_cell_indeces_to_track(tracks[id], grid_test)
    X = t[0, :] - grid_test.origin[0]
    Y = t[1, :] - grid_test.origin[1]
    U = t[-1, :] + grid_test.resolution / 2
    V = t[-2, :] + grid_test.resolution / 2
    plt.plot(X, Y, "x-", markersize=0.1, linewidth=0.1)
    plt.scatter(U, V, s=0.1)

# %%

center = (691, 925)
inputs = (
    np.asarray(
        occupancy.binary_map.crop(window.corners(center)).resize(
            (NET_WINDOW_SIZE, NET_WINDOW_SIZE), Image.ANTIALIAS
        ),
        "float",
    )
    / 255.0
)

outputs = estimate_dynamics(net, inputs, device=DEVICE, batch_size=32)

plot_quivers(inputs, outputs, dpi=PLOT_DPI)
plt.plot(NET_WINDOW_SIZE // 2, NET_WINDOW_SIZE // 2, "o", markersize=0.5)

# %%

print(f"Deep model: people_net{net_id_string}")

evaluation_ids = range(len(tracks))
# evaluation_ids = [5101]  # straight track
# evaluation_ids = [4110]  # track with corners
# evaluation_ids = [random.randint(0, len(tracks) - 1) for i in range(10)]
# evaluation_ids = ids

like = 0.0
matches = 0
pbar = tqdm(evaluation_ids, postfix={"avg likelihood": 0})
for id in pbar:
    p = pixels_from_track(tracks[id], occupancy)
    like_t, matches_t = track_likelihood_net(
        p, occupancy, NET_WINDOW_SIZE, NET_SCALE_FACTOR, net, DEVICE
    )
    like += like_t
    matches += matches_t
    pbar.set_postfix({"avg likelihood": like / matches})  # type: ignore
print(
    f"chance: {1 / 8}, total like: {like:.3f}, avg like: "
    f"{like / matches:.3f} "
    f"(on {len(evaluation_ids)} tracks)"
)

# %%

print("Optimal model")

evaluation_ids = range(len(tracks))
# evaluation_ids = [5101]  # straight track
# evaluation_ids = [4110]  # track with corners
# evaluation_ids = [random.randint(0, len(tracks) - 1) for i in range(10)]
# evaluation_ids = ids

like = 0.0
matches = 0
missing = 0
pbar = tqdm(evaluation_ids, postfix={"avg likelihood": 0, "missing": 0})
for id in pbar:
    t = append_cell_indeces_to_track(tracks[id], grid_test)
    like_t, matches_t, missing_t = track_likelihood_model(t, grid_test)
    like += like_t
    matches += matches_t
    missing += missing_t
    pbar.set_postfix(  # type: ignore
        {"avg likelihood": like / matches, "missing": missing}
    )
print(
    f"chance: {1 / 8}, total like: {like:.3f}, avg like: "
    f"{like / matches:.3f} (on {(len(evaluation_ids))} tracks)"
)
