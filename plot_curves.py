import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

PLOT_KTH = True
PLOT_ATC = True
ATC_DAYS = {1: ("20121114", 3121209), 2: ("20121118", 8533469)}
NUM_POINTS = 71

files_kth = Path("curves/IROS").glob("KTH_*.json")
files_day1 = Path("curves/IROS").glob(f"ATC_train{ATC_DAYS[1][0]}*.json")

SOURCES_KTH = {file.stem: file for file in files_kth}
SOURCES_DAY1 = {file.stem: file for file in files_day1}


def curve_plot(
    curves: dict[str, Path], num_points: Optional[int] = None
) -> None:
    plt.figure(dpi=300)
    for name, source_file in curves.items():
        with open(source_file) as json_file:
            data: dict[str, dict] = json.load(json_file)
        X = []
        Y = []
        for x, y in data.items():
            X.append(int(x))
            Y.append(float(y["avg_like"]))
        plt.plot(
            X[: num_points if num_points else -1],
            Y[: num_points if num_points else -1],
            label=name,
        )
    plt.legend(loc="lower right")
    plt.xlabel("number of observations")
    plt.ylabel(r"$\mathcal{L}$")


if PLOT_KTH:
    curve_plot(curves=SOURCES_KTH, num_points=NUM_POINTS)
if PLOT_ATC:
    curve_plot(curves=SOURCES_DAY1, num_points=NUM_POINTS)
