from pathlib import Path

import pandas as pd
import scipy.io as sio

BASE_PATH = Path("/mnt/hdd/datasets/KTH_track/")
MAP_METADATA = BASE_PATH / "map.yaml"
MAP_PGM = BASE_PATH / "map.pgm"
TRACKS_DATA = BASE_PATH / "dataTrajectoryNoIDCell6251.mat"
CSV_FILE = BASE_PATH / "kth_trajectory_data.csv"

tracks = sio.loadmat(TRACKS_DATA, squeeze_me=True)["dataTrajectoryNoIDCell"]
tracks_list = []
for i, t in enumerate(tracks):
    t_df = pd.DataFrame(t.T, columns=["x", "y", "sec", "nsec"])
    t_df["person_id"] = i
    tracks_list.append(t_df)
tracks_df = pd.concat(tracks_list)
tracks_df["time"] = tracks_df.sec + tracks_df.nsec * 1e-09
tracks_df.sort_values("time", inplace=True)
tracks_df.to_csv(
    CSV_FILE, index=False, columns=["time", "person_id", "x", "y"]
)
