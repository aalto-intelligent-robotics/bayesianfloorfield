import pickle

import pandas as pd

from mod.grid import Grid
from mod.models import BayesianDiscreteDirectional
from mod.utils import XYCoords, get_local_settings

CHUNK_SIZE = 2000
MAX_OBSERVATIONS = None
ATC_DAY = "20121114"
# ATC_DAY = "20121118"

_, local = get_local_settings("config/local_settings_atc.json")
csv_path = local["dataset_folder"] + f"subsampled/atc-{ATC_DAY}_5.csv"
pickle_path = (
    local["pickle_folder"] + f"bayes/{ATC_DAY}/discrete_directional_{ATC_DAY}"
)

input_file = pd.read_csv(csv_path, chunksize=CHUNK_SIZE)
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
    if MAX_OBSERVATIONS is not None and total_observations >= MAX_OBSERVATIONS:
        print(f"** Stopping, max observation ({MAX_OBSERVATIONS}) reached")
        break
