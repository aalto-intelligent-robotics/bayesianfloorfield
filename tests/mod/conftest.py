from io import StringIO
from typing import NamedTuple

import pandas as pd
import pytest
import yaml
from PIL import Image

import mod.models as mod
from mod.grid import Grid
from mod.utils import RCCoords, XYCoords


class OccupancyMapPaths(NamedTuple):
    image: str
    metadata: str


@pytest.fixture
def sample_data() -> pd.DataFrame:
    data_string = (
        "time,person_id,x,y,velocity,motion_angle\n"
        "0.708,1,39.830,-22.779,0.823082,5.784\n"
        "0.742,2,-9.671,7.410,0.903133,2.269\n"
        "0.813,3,34.735,-17.600,1.331884,2.817\n"
        "0.882,1,39.793,-22.710,0.532843,5.631\n"
        "0.915,2,-9.853,7.596,1.199536,2.451\n"
        "0.983,3,34.479,-17.615,1.338209,2.908\n"
        "0.051,1,39.747,-22.581,0.226711,2.899\n"
        "0.085,2,-10.114,7.718,1.112047,2.457\n"
        "0.152,3,34.193,-17.656,1.448267,3.025"
    )
    return pd.read_csv(StringIO(data_string), skipinitialspace=True)


@pytest.fixture
def bayesian_grid() -> Grid:
    cell1 = mod.BayesianDiscreteDirectional(
        coords=XYCoords(0, 0), index=RCCoords(0, 0), resolution=1
    )
    cell2 = mod.BayesianDiscreteDirectional(
        coords=XYCoords(0, 1), index=RCCoords(0, 1), resolution=1
    )
    cell1.bins = [0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
    cell2.bins = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    columns = [
        "time",
        "person_id",
        "x",
        "y",
        "motion_angle",
        "row",
        "col",
        "bin",
    ]
    cell1.data = pd.DataFrame(
        [
            [0, 0, 0.4, 2.5, 0, 0, 0, 0],
            [0.1, 0, 0.6, 2.2, 3.14159 * 5 / 4, 0, 0, 5],
        ],
        columns=columns,
    )
    cell2.data = pd.DataFrame(
        [[0.2, 1, 1.5, 2.8, 3.14159 / 4, 0, 1, 1]], columns=columns
    )
    return Grid(
        resolution=1,
        origin=XYCoords(0, 2),
        model=mod.BayesianDiscreteDirectional,
        cells={RCCoords(0, 0): cell1, RCCoords(0, 1): cell2},
        total_count=3,
    )


@pytest.fixture(scope="session")
def sample_occupancy_map_and_yaml_paths(
    tmp_path_factory: pytest.TempPathFactory,
) -> OccupancyMapPaths:
    tmp_path = tmp_path_factory.mktemp("data")
    map = Image.new("L", (3, 2))
    map.putpixel((1, 0), 128)
    fn = tmp_path / "map.png"
    map.save(fn)

    fn_yaml = tmp_path / "map.yaml"
    metadata = {
        "image": "map.png",
        "resolution": 0.5,
        "origin": [1, 2, 0],
        "negate": True,
        "occupied_thresh": 0.5,
        "free_thresh": 0.1,
    }
    with open(fn_yaml, "w") as yaml_file:
        yaml.dump(metadata, yaml_file)
    return OccupancyMapPaths(fn.as_posix(), fn_yaml.as_posix())
