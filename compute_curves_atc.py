# %% Some magic
# ! %load_ext autoreload
# ! %autoreload 2
# %% Imports

import json
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
    pixels2grid_complete,
    track2pixels,
    track_likelihood_model,
)
from mod import grid, models
from mod.occupancy import OccupancyMap
from mod.utils import XYCoords

sys.modules["Grid"] = grid
sys.modules["Models"] = models

# Change BASE_PATH to the folder where data and models are located
BASE_PATH = Path("/mnt/hdd/datasets/ATC/")
ATC_DAYS = {1: ("20121114", 3121209), 2: ("20121118", 8533469)}
TRAIN_DAY = 1
TEST_DAY = 2
BAYES_INCREMENT = 50000
USE_PRIOR = False
RUN_SUFFIX = ""

ATC_TRAIN_DAY = ATC_DAYS[TRAIN_DAY][0]
BAYES_MAX_DATA_TRAIN = ATC_DAYS[TRAIN_DAY][1]
ATC_TEST_DAY = ATC_DAYS[TEST_DAY][0]
BAYES_MAX_DATA_TEST = ATC_DAYS[TEST_DAY][1]
MAP_METADATA = BASE_PATH / "localization_grid.yaml"
GRID_BAYES_DATA = {
    i: BASE_PATH
    / "models"
    / "bayes"
    / ATC_TRAIN_DAY
    / f"discrete_directional_{ATC_TRAIN_DAY}_{i:07d}.p"
    for i in range(0, BAYES_MAX_DATA_TRAIN, BAYES_INCREMENT)
}
GRID_BAYES_DATA[BAYES_MAX_DATA_TRAIN] = (
    BASE_PATH
    / "models"
    / "bayes"
    / ATC_TRAIN_DAY
    / f"discrete_directional_{ATC_TRAIN_DAY}_{BAYES_MAX_DATA_TRAIN:07d}.p"
)
GRID_FULL_DATA = GRID_BAYES_DATA[BAYES_MAX_DATA_TRAIN]
GRID_DATA = (
    BASE_PATH
    / "models"
    / "bayes"
    / ATC_TEST_DAY
    / f"discrete_directional_{ATC_TEST_DAY}_{BAYES_MAX_DATA_TEST:07d}.p"
)

GRID_SCALE = 20
PLOT_DPI = 800

grid_test: grid.Grid = pickle.load(open(GRID_DATA, "rb"))
grid_full: grid.Grid = pickle.load(open(GRID_FULL_DATA, "rb"))
occupancy = OccupancyMap.from_yaml(MAP_METADATA)
occupancy.origin = XYCoords(-60, -40)
tracks = convert_grid(grid_test)


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
    grid_bayes: grid.Grid = pickle.load(open(grid_bayes_path_data, "rb"))
    if USE_PRIOR:
        print("Assigning prior")
        grid.assign_prior_to_grid(
            grid=grid_bayes, prior=[1 / 8] * 8, alpha=100
        )
    like = 0.0
    matches = 0
    missing = 0
    for id in tqdm(evaluation_ids):
        p = track2pixels(tracks[id], occupancy)
        t = pixels2grid_complete(
            p, occupancy.resolution * GRID_SCALE, occupancy.resolution
        )
        t_like, t_matches, t_missing = track_likelihood_model(
            t, occupancy, grid_bayes
        )
        like += t_like
        matches += t_matches
        missing += t_missing

    bayes_data[iterations] = {
        "total_like": like,
        "matches": matches,
        "avg_like": like / matches if matches else 0.0,
        "num_tracks": len(evaluation_ids),
        "missing": missing,
    }
    print(
        f"chance: {1 / 8}, total like: {like:.3f}, avg like: "
        f"{like / matches if matches else 0.0:.3f} "
        f"(on {len(evaluation_ids)} tracks, {missing}/{matches} cell missed)"
    )

with open(
    (
        f"curves/ATC_train{ATC_TRAIN_DAY}_test{ATC_TEST_DAY}"
        + ("_prior" if USE_PRIOR else "")
        + ("_" + RUN_SUFFIX if RUN_SUFFIX else "")
        + ".json"
    ),
    "w",
) as f:
    json.dump(bayes_data, f)

# %%

print("Traditional model")

evaluation_ids = range(len(tracks))
# evaluation_ids = [5101]
# evaluation_ids = [random.randint(0, len(tracks) - 1) for i in range(10)]
# evaluation_ids = ids

like = 0.0
matches = 0
missing = 0
for id in tqdm(evaluation_ids):
    p = track2pixels(tracks[id], occupancy)
    t = pixels2grid_complete(
        p, occupancy.resolution * GRID_SCALE, occupancy.resolution
    )

    t_like, t_matches, t_missing = track_likelihood_model(
        t, occupancy, grid_full
    )
    like += t_like
    matches += t_matches
    missing += t_missing
print(
    f"chance: {1 / 8}, total like: {like:.3f}, avg like: "
    f"{like / matches if matches else 0.0:.3f} "
    f"(on {len(evaluation_ids)} tracks, {missing}/{matches} cell missed)"
)

# %%

print("Optimal model")

evaluation_ids = range(len(tracks))
# evaluation_ids = [5101]
# evaluation_ids = [random.randint(0, len(tracks) - 1) for i in range(10)]
# evaluation_ids = ids

like = 0.0
matches = 0
missing = 0
for id in tqdm(evaluation_ids):
    p = track2pixels(tracks[id], occupancy)
    t = pixels2grid_complete(
        p, occupancy.resolution * GRID_SCALE, occupancy.resolution
    )
    t_like, t_matches, t_missing = track_likelihood_model(
        t, occupancy, grid_test
    )
    like += t_like
    matches += t_matches
    missing += t_missing
print(
    f"chance: {1 / 8}, total like: {like:.3f}, avg like: "
    f"{like / matches if matches else 0.0:.3f} "
    f"(on {len(evaluation_ids)} tracks, {missing}/{matches} cell missed)"
)
