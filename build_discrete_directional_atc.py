import pickle

import pandas as pd

from mod.grid import Grid
from mod.models import DiscreteDirectional, XYCoords
from mod.occupancy import OccupancyMap
from mod.utils import get_local_settings
from mod.visualisation import MapVisualisation

use_pickle = False

_, local = get_local_settings("config/local_settings_atc.json")
csv_path = local["dataset_folder"] + "subsampled/atc-20121114_5.csv"
pickle_path = local["pickle_folder"] + "discrete_directional.p"

if use_pickle:
    g = pickle.load(open(pickle_path, "rb"))
else:
    input_file = pd.read_csv(csv_path, chunksize=100000)
    g = Grid(
        origin=XYCoords(-40000, -40000),
        resolution=1000,
        model=DiscreteDirectional,
    )
    total_observations = 0
    for chunk in input_file:
        g.add_data(chunk)
        total_observations = total_observations + len(chunk.index)
    g.update_model()
    pickle.dump(g, open(pickle_path, "wb"))

occupancy = OccupancyMap.from_yaml(
    local["dataset_folder"] + "localization_grid.yaml"
)

v = MapVisualisation(mod=g, occ=occupancy)
v.show(occ_overlay=True)
