import json
from pathlib import Path

import matplotlib.pyplot as plt

ATC_DAYS = {1: ("20121114", 3121209), 2: ("20121118", 8533469)}

files_day1 = Path("curves").glob(f"ATC_train{ATC_DAYS[1][0]}*.json")
files_day2 = Path("curves").glob(f"ATC_train{ATC_DAYS[2][0]}*.json")

SOURCES_DAY1 = {file.stem: file for file in files_day1}
SOURCES_DAY2 = {file.stem: file for file in files_day2}
print(SOURCES_DAY1)
print(SOURCES_DAY2)

fig = plt.figure(dpi=300)
for name, source_file in SOURCES_DAY1.items():
    with open(source_file) as json_file:
        data: dict[str, dict] = json.load(json_file)
    X = []
    Y = []
    for x, y in data.items():
        X.append(int(x))
        Y.append(float(y["avg_like"]))
    plt.plot(X, Y, linestyle="dotted", label=name)
plt.legend(loc="lower right")
# plt.show()

for name, source_file in SOURCES_DAY2.items():
    with open(source_file) as json_file:
        data = json.load(json_file)
    X = []
    Y = []
    for x, y in data.items():
        X.append(int(x))
        Y.append(float(y["avg_like"]))
    plt.plot(X, Y, linestyle="dotted", label=name)
plt.legend(loc="lower right")
