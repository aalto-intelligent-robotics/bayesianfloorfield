import pickle

import pandas as pd

from mod.grid import Grid
from mod.models import BayesianDiscreteDirectional
from mod.occupancy import OccupancyMap
from mod.utils import XYCoords, get_local_settings
from mod.visualisation import show_all

ATC_DAY = "20121114"
# ATC_DAY = "20121118"

USE_PICKLE = False

_, local = get_local_settings("config/local_settings_atc.json")
csv_path = local["dataset_folder"] + f"subsampled/atc-{ATC_DAY}_5.csv"
pickle_path = (
    local["pickle_folder"] + f"bayes/{ATC_DAY}/discrete_directional_{ATC_DAY}"
)

if USE_PICKLE:
    g = pickle.load(open(pickle_path, "rb"))
else:
    input_file = pd.read_csv(csv_path, chunksize=50000)
    g = Grid(
        origin=XYCoords(-42, -40),
        resolution=1,
        model=BayesianDiscreteDirectional,
    )
    total_observations = 0

    print("Processing prior")
    g.update_model()
    filename = f"{pickle_path}_{total_observations:07d}.p"
    pickle.dump(g, open(filename, "wb"))
    print(f"** Saved {filename}")

    for chunk in input_file:
        print(
            f"Processing chunk [{total_observations}-"
            f"{total_observations + len(chunk.index)}]"
        )
        chunk[["x", "y"]] /= 1000  # convert from mm to m
        g.add_data(chunk)
        total_observations = total_observations + len(chunk.index)
        print("** Chunk processed, updating model...")
        g.update_model()
        filename = f"{pickle_path}_{total_observations:07d}.p"
        pickle.dump(g, open(filename, "wb"))
        print(f"** Saved {filename}")

occupancy = OccupancyMap.from_yaml(
    local["dataset_folder"] + "localization_grid.yaml"
)

show_all(grid=g, occ=occupancy, occ_overlay=True)
