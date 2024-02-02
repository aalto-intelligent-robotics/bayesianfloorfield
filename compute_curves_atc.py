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
    append_cell_indeces_to_track,
    extract_tracks_from_grid,
    track_likelihood_model,
)
from mod import grid, models
from mod.occupancy import OccupancyMap
from mod.utils import XYCoords
from mod.visualisation import show_occupancy

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
GRID_TEST_DATA = (
    BASE_PATH
    / "models"
    / "bayes"
    / ATC_TEST_DAY
    / f"discrete_directional_{ATC_TEST_DAY}_{BAYES_MAX_DATA_TEST:07d}.p"
)

PLOT_DPI = 800

grid_test: grid.Grid = pickle.load(open(GRID_TEST_DATA, "rb"))
grid_full: grid.Grid = pickle.load(open(GRID_FULL_DATA, "rb"))
occupancy = OccupancyMap.from_yaml(MAP_METADATA)
occupancy.origin = XYCoords(-60, -40)
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
    t = append_cell_indeces_to_track(tracks[id], grid_test)
    X = t[0, :] - grid_test.origin[0]
    Y = t[1, :] - grid_test.origin[1]
    U = t[-1, :] + grid_test.resolution / 2
    V = t[-2, :] + grid_test.resolution / 2
    plt.plot(X, Y, "x-", markersize=0.1, linewidth=0.1)
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
    pbar = tqdm(evaluation_ids, postfix={"avg likelihood": 0, "missing": 0})
    for id in pbar:
        t = append_cell_indeces_to_track(tracks[id], grid_test)
        t_like, t_matches, t_missing = track_likelihood_model(t, grid_bayes)
        like += t_like
        matches += t_matches
        missing += t_missing
        pbar.set_postfix(  # type: ignore
            {"avg likelihood": like / matches, "missing": missing}
        )
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
pbar = tqdm(evaluation_ids, postfix={"avg likelihood": 0, "missing": 0})
for id in pbar:
    t = append_cell_indeces_to_track(tracks[id], grid_test)
    like_t, matches_t, missing_t = track_likelihood_model(t, grid_full)
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

# %%

print("Optimal model")

evaluation_ids = range(len(tracks))
# evaluation_ids = [5101]
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
