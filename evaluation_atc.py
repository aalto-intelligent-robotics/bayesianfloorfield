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

from deepflow.evaluation import (
    convert_grid,
    pixels2grid,
    track2pixels,
    track_likelihood_model,
    track_likelihood_net,
)
from deepflow.nets import DiscreteDirectional
from deepflow.utils import Window, estimate_dynamics, plot_quivers
from mod import Grid, Models
from mod.OccupancyMap import OccupancyMap

sys.modules["Grid"] = Grid
sys.modules["Models"] = Models

BASE_PATH = Path("/mnt/hdd/datasets/ATC/")
MAP_METADATA = BASE_PATH / "localization_grid.yaml"
MAP_PGM = BASE_PATH / "localization_grid.pgm"
GRID_DATA = BASE_PATH / "models" / "discrete_directional.p"
GRID_TRAIN_DATA = BASE_PATH / "models" / "discrete_directional_2.p"

EPOCHS = 100
WINDOW_SIZE = 64
SCALE = 8
# GRID_SCALE = SCALE
GRID_SCALE = 20

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
PLOT_DPI = 800

grid: Grid.Grid = pickle.load(open(GRID_DATA, "rb"))
grid_train: Grid.Grid = pickle.load(open(GRID_TRAIN_DATA, "rb"))
occupancy = OccupancyMap.from_yaml(MAP_METADATA)
occupancy.origin = [-60.0, -40.0, 0.0]
tracks: np.ndarray = convert_grid(grid)

id_string = f"_w{WINDOW_SIZE}_s{SCALE}_t_{EPOCHS}"

net = DiscreteDirectional(WINDOW_SIZE)
window = Window(WINDOW_SIZE * SCALE)

path = f"./people_net{id_string}.pth"
checkpoint = torch.load(path)["model_state_dict"]
net.load_state_dict(checkpoint)


def show_occupancy(occupancy: OccupancyMap) -> None:
    r = occupancy.resolution
    o = occupancy.origin
    sz = occupancy.map.size
    extent = (o[0], o[0] + sz[0] * r, o[1], o[1] + sz[1] * r)
    plt.imshow(occupancy.map, extent=extent, cmap="gray")


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
    cmap="gray",
)
plt.grid(True, linewidth=0.1)
plt.xticks(range(0, occupancy.map.size[0], GRID_SCALE))
plt.yticks(range(0, occupancy.map.size[1], GRID_SCALE))
for id in ids:
    p = track2pixels(tracks[id], occupancy)
    t = pixels2grid(p, occupancy.resolution * GRID_SCALE, occupancy.resolution)
    X = t[1, :]
    Y = t[0, :]
    U = t[3, :] * GRID_SCALE + GRID_SCALE / 2
    V = t[2, :] * GRID_SCALE + GRID_SCALE / 2
    plt.plot(p[1, :], p[0, :], "x-", markersize=0.1, linewidth=0.1)
    plt.plot(X, Y, "o", markersize=0.1, linewidth=0.1)
    plt.scatter(U, V, s=0.1)

# %%

center = (691, 925)
inputs = (
    np.asarray(
        occupancy.binary_map.crop(window.corners(center)).resize(
            (WINDOW_SIZE, WINDOW_SIZE), Image.ANTIALIAS
        ),
        "float",
    )
    / 255.0
)

outputs = estimate_dynamics(net, inputs, device=DEVICE, batch_size=32)

plot_quivers(inputs, outputs, dpi=PLOT_DPI)
plt.plot(WINDOW_SIZE // 2, WINDOW_SIZE // 2, "o", markersize=0.5)

# %%

print(f"Deep model: people_net{id_string}")

evaluation_ids = range(len(tracks))
# evaluation_ids = [5101]
# evaluation_ids = [random.randint(0, len(tracks) - 1) for i in range(10)]
# evaluation_ids = ids

like = 0.0
skipped = 0
for id in tqdm(evaluation_ids):
    p = track2pixels(tracks[id], occupancy)
    t = pixels2grid(p, occupancy.resolution * GRID_SCALE, occupancy.resolution)
    if t.shape[1] > 1:
        like += track_likelihood_net(
            t, occupancy, WINDOW_SIZE, SCALE, net, DEVICE
        )
    else:
        skipped += 1
print(
    f"chance: {1 / 8}, total like: {like:.3f}, avg like: "
    f"{like / (len(evaluation_ids)-skipped):.3f} "
    f"(on {(len(evaluation_ids)-skipped)} tracks, {skipped} skipped)"
)

# %%

print("Traditional model")

evaluation_ids = range(len(tracks))
# evaluation_ids = [5101]
# evaluation_ids = [random.randint(0, len(tracks) - 1) for i in range(10)]
# evaluation_ids = ids

like = 0.0
skipped = 0
for id in tqdm(evaluation_ids):
    p = track2pixels(tracks[id], occupancy)
    t = pixels2grid(p, occupancy.resolution * GRID_SCALE, occupancy.resolution)
    if t.shape[1] > 1:
        like += track_likelihood_model(t, occupancy, grid_train)
    else:
        skipped += 1
print(
    f"chance: {1 / 8}, total like: {like:.3f}, avg like: "
    f"{like / (len(evaluation_ids)-skipped):.3f} "
    f"(on {(len(evaluation_ids)-skipped)} tracks, {skipped} skipped)"
)

# %%

print("Optimal model")

evaluation_ids = range(len(tracks))
# evaluation_ids = [5101]
# evaluation_ids = [random.randint(0, len(tracks) - 1) for i in range(10)]
# evaluation_ids = ids

like = 0.0
skipped = 0
for id in tqdm(evaluation_ids):
    p = track2pixels(tracks[id], occupancy)
    t = pixels2grid(p, occupancy.resolution * GRID_SCALE, occupancy.resolution)
    if t.shape[1] > 1:
        like += track_likelihood_model(t, occupancy, grid)
    else:
        skipped += 1
print(
    f"chance: {1 / 8}, total like: {like:.3f}, avg like: "
    f"{like / (len(evaluation_ids)-skipped):.3f} "
    f"(on {(len(evaluation_ids)-skipped)} tracks, {skipped} skipped)"
)
