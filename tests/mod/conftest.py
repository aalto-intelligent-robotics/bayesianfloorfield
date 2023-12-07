from io import StringIO
from typing import NamedTuple

import pandas as pd
import pytest
import yaml
from PIL import Image


class OccupancyMapPaths(NamedTuple):
    image: str
    metadata: str


@pytest.fixture
def sample_data() -> pd.DataFrame:
    data_string = (
        "time,person_id,x,y,velocity,motion_angle\n"
        "0.708,1,39830,-22779,0.823082,5.784\n"
        "0.742,2,-9671,7410,0.903133,2.269\n"
        "0.813,3,34735,-17600,1.331884,2.817\n"
        "0.882,1,39793,-22710,0.532843,5.631\n"
        "0.915,2,-9853,7596,1.199536,2.451\n"
        "0.983,3,34479,-17615,1.338209,2.908\n"
        "0.051,1,39747,-22581,0.226711,2.899\n"
        "0.085,2,-10114,7718,1.112047,2.457\n"
        "0.152,3,34193,-17656,1.448267,3.025"
    )
    return pd.read_csv(StringIO(data_string), skipinitialspace=True)


@pytest.fixture(scope="session")
def sample_occupancy_map_and_yaml_paths(
    tmp_path_factory: pytest.TempPathFactory,
) -> OccupancyMapPaths:
    tmp_path = tmp_path_factory.mktemp("data")
    map = Image.new("L", (2, 2))
    map.putpixel((1, 0), 128)
    fn = tmp_path / "map.png"
    map.save(fn)

    fn_yaml = tmp_path / "map.yaml"
    metadata = {
        "image": "map.png",
        "resolution": 1,
        "origin": [7, 10, 0],
        "negate": True,
        "occupied_thresh": 0.5,
        "free_thresh": 0.1,
    }
    with open(fn_yaml, "w") as yaml_file:
        yaml.dump(metadata, yaml_file)
    return OccupancyMapPaths(fn.as_posix(), fn_yaml.as_posix())
