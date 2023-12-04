import pickle

import pandas as pd

from mod.grid import Grid
from mod.models import DiscreteDirectional, XYCoords
from mod.occupancy import OccupancyMap
from mod.utils import get_local_settings
from mod.visualisation import show_all

use_pickle = False

_, local = get_local_settings("config/local_settings_kth.json")
csv_path = local["dataset_folder"] + "kth_trajectory_data.csv"
pickle_path = local["pickle_folder"] + "discrete_directional_kth.p"

if use_pickle:
    g = pickle.load(open(pickle_path, "rb"))
else:
    input_file = pd.read_csv(csv_path, chunksize=100000)
    g = Grid(
        origin=XYCoords(-58.9, -30.75), resolution=1, model=DiscreteDirectional
    )
    total_observations = 0
    for chunk in input_file:
        g.add_data(chunk)
        total_observations = total_observations + len(chunk.index)
    g.update_model()
    pickle.dump(g, open(pickle_path, "wb"))

occupancy = OccupancyMap.from_yaml(local["dataset_folder"] + "map.yaml")
occupancy.origin = [0, 0, 0]

show_all(grid=g, occ=occupancy, occ_overlay=True)
