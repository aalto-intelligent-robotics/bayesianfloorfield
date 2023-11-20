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
from tqdm import tqdm

from directionalflow.evaluation import (
    convert_grid,
    pixels2grid,
    track2pixels,
    track_likelihood_model,
)
from mod import Grid, Models
from mod.OccupancyMap import OccupancyMap

sys.modules["Grid"] = Grid
sys.modules["Models"] = Models

# Change BASE_PATH to the folder where data and models are located
BASE_PATH = Path("/mnt/hdd/datasets/ATC/")

MAP_METADATA = BASE_PATH / "localization_grid.yaml"
GRID_BAYES_DATA = {
    i: BASE_PATH / "models" / "bayes" / f"discrete_directional_{i:07d}.p"
    for i in range(0, 3100001, 100000)
}
GRID_FULL_DATA = BASE_PATH / "models" / "discrete_directional.p"
GRID_DATA = BASE_PATH / "models" / "discrete_directional.p"

GRID_SCALE = 20
PLOT_DPI = 800

grid: Grid.Grid = pickle.load(open(GRID_DATA, "rb"))
grid_full: Grid.Grid = pickle.load(open(GRID_FULL_DATA, "rb"))
occupancy = OccupancyMap.from_yaml(MAP_METADATA)
occupancy.origin = [-60.0, -40.0, 0.0]
tracks = convert_grid(grid_full)


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

print("Bayesian model")

evaluation_ids = range(len(tracks))
# evaluation_ids = [5101]
# evaluation_ids = [random.randint(0, len(tracks) - 1) for i in range(10)]
# evaluation_ids = ids

bayes_data = {}
for iterations, grid_bayes_path_data in GRID_BAYES_DATA.items():
    print(f"Iterations: {iterations} - file {grid_bayes_path_data.name}")
    grid_bayes: Grid.Grid = pickle.load(open(grid_bayes_path_data, "rb"))
    like = 0.0
    skipped = 0
    for id in tqdm(evaluation_ids):
        p = track2pixels(tracks[id], occupancy)
        t = pixels2grid(
            p, occupancy.resolution * GRID_SCALE, occupancy.resolution
        )
        if t.shape[1] > 1:
            track_like = track_likelihood_model(t, occupancy, grid_bayes)
            like += track_like
        else:
            skipped += 1
    bayes_data[iterations] = {
        "total_like": like,
        "avg_like": like / (len(evaluation_ids) - skipped),
        "num_tracks": len(evaluation_ids) - skipped,
        "skipped": skipped,
    }
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
        like += track_likelihood_model(t, occupancy, grid_full)
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
