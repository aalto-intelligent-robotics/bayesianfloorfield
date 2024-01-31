import pickle

import pandas as pd

from mod.grid import Grid
from mod.models import DiscreteDirectional
from mod.occupancy import OccupancyMap
from mod.utils import XYCoords, get_local_settings
from mod.visualisation import show_all

ATC_DAY = "20121114"
# ATC_DAY = "20121118"

USE_PICKLE = False

_, local = get_local_settings("config/local_settings_atc.json")
csv_path = local["dataset_folder"] + f"subsampled/atc-{ATC_DAY}_5.csv"
pickle_path = local["pickle_folder"] + f"discrete_directional_{ATC_DAY}.p"

if USE_PICKLE:
    g = pickle.load(open(pickle_path, "rb"))
else:
    input_file = pd.read_csv(csv_path, chunksize=100000)
    g = Grid(
        origin=XYCoords(-42, -40),
        resolution=1,
        model=DiscreteDirectional,
    )
    total_observations = 0
    for chunk in input_file:
        chunk[["x", "y"]] /= 1000  # convert from mm to m
        g.add_data(chunk)
        total_observations = total_observations + len(chunk.index)
    g.update_model()
    pickle.dump(g, open(pickle_path, "wb"))

occupancy = OccupancyMap.from_yaml(
    local["dataset_folder"] + "localization_grid.yaml"
)

show_all(grid=g, occ=occupancy, occ_overlay=True)
