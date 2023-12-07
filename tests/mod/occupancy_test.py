import numpy as np
import pytest

from mod.occupancy import OccupancyMap
from mod.utils import RCCoords, XYCoords

from .conftest import OccupancyMapPaths


def test_occupancy_init_from_yaml(
    sample_occupancy_map_and_yaml_paths: OccupancyMapPaths,
) -> None:
    occ = OccupancyMap.from_yaml(sample_occupancy_map_and_yaml_paths.metadata)
    assert occ.resolution == 0.5
    assert occ.origin.x == 1
    assert occ.origin.y == 2
    assert occ.negate
    assert occ.occupied_thresh == 128
    assert occ.free_thresh == 26
    assert np.array_equal(occ.map, [[0, 128, 0], [0, 0, 0]])


@pytest.mark.parametrize(
    ["thresh", "expected_thresh"],
    [
        (0.0, 0),
        (0.2, 51),
        (0.4, 102),
        (0.8, 204),
        (1.0, 1),
        (100.0, 100),
        (255.0, 255),
    ],
)
def test_occupancy_init_thresholds(
    sample_occupancy_map_and_yaml_paths: OccupancyMapPaths,
    thresh: float,
    expected_thresh: int,
) -> None:
    image_path = sample_occupancy_map_and_yaml_paths.image
    occ = OccupancyMap(
        image_file=image_path,
        resolution=1,
        origin=[0, 0, 0],
        negate=False,
        free_thresh=thresh,
        occupied_thresh=thresh,
    )
    assert occ.occupied_thresh == expected_thresh
    assert occ.free_thresh == expected_thresh


@pytest.mark.parametrize(
    ["thresh", "negate", "expected"],
    [
        (0.0, False, [[255, 255, 255], [255, 255, 255]]),
        (0.2, False, [[255, 255, 255], [255, 255, 255]]),
        (0.5, False, [[255, 0, 255], [255, 255, 255]]),
        (208.0, False, [[255, 0, 255], [255, 255, 255]]),
        (255.0, False, [[0, 0, 0], [0, 0, 0]]),
        (0.0, True, [[0, 255, 0], [0, 0, 0]]),
        (0.2, True, [[0, 255, 0], [0, 0, 0]]),
        (0.5, True, [[0, 0, 0], [0, 0, 0]]),
        (208.0, True, [[0, 0, 0], [0, 0, 0]]),
        (255.0, True, [[0, 0, 0], [0, 0, 0]]),
    ],
)
def test_occupancy_binary_map(
    sample_occupancy_map_and_yaml_paths: OccupancyMapPaths,
    thresh: float,
    negate: bool,
    expected: list,
) -> None:
    image_path = sample_occupancy_map_and_yaml_paths.image
    occ = OccupancyMap(
        image_file=image_path,
        resolution=1,
        origin=[0, 0, 0],
        negate=negate,
        free_thresh=thresh,
        occupied_thresh=thresh,
    )
    assert np.array_equal(occ.binary_map, expected)


@pytest.mark.parametrize(
    ["pixel", "xy"],
    [
        (RCCoords(0, 0), XYCoords(1.25, 2.75)),
        (RCCoords(1, 0), XYCoords(1, 2)),
        (RCCoords(0, 1), XYCoords(1.5, 2.5)),
        (RCCoords(1, 1), XYCoords(1.75, 2.25)),
        (RCCoords(0, 2), XYCoords(2, 2.5)),
        (RCCoords(1, 2), XYCoords(2.25, 2.25)),
    ],
)
def test_pixel_from_XY(
    sample_occupancy_map_and_yaml_paths: OccupancyMapPaths,
    pixel: RCCoords,
    xy: XYCoords,
) -> None:
    occ = OccupancyMap.from_yaml(sample_occupancy_map_and_yaml_paths.metadata)
    assert occ.pixel_from_XY(xy) == pixel


@pytest.mark.parametrize(
    "xy",
    [
        XYCoords(-1, -2),
        XYCoords(0.9, 1.9),
        XYCoords(0.5, 2.5),
        XYCoords(1.5, 1.5),
        XYCoords(1.5, 3),
        XYCoords(2.5, 2.5),
    ],
)
def test_pixel_from_XY_out_of_bound(
    sample_occupancy_map_and_yaml_paths: OccupancyMapPaths, xy: XYCoords
) -> None:
    occ = OccupancyMap.from_yaml(sample_occupancy_map_and_yaml_paths.metadata)
    with pytest.raises(ValueError):
        occ.pixel_from_XY(xy)


@pytest.mark.parametrize(
    ["pixel", "xy"],
    [
        (RCCoords(0, 0), XYCoords(1, 2.5)),
        (RCCoords(1, 0), XYCoords(1, 2)),
        (RCCoords(0, 1), XYCoords(1.5, 2.5)),
        (RCCoords(1, 1), XYCoords(1.5, 2)),
        (RCCoords(0, 2), XYCoords(2, 2.5)),
        (RCCoords(1, 2), XYCoords(2, 2)),
    ],
)
def test_XY_from_pixel(
    sample_occupancy_map_and_yaml_paths: OccupancyMapPaths,
    pixel: RCCoords,
    xy: XYCoords,
) -> None:
    occ = OccupancyMap.from_yaml(sample_occupancy_map_and_yaml_paths.metadata)
    assert occ.XY_from_pixel(pixel) == xy


@pytest.mark.parametrize(
    "pixel",
    [
        RCCoords(-1, 0),
        RCCoords(0, -1),
        RCCoords(2, 2),
        RCCoords(1, 3),
    ],
)
def test_XY_from_pixel_out_of_bound(
    sample_occupancy_map_and_yaml_paths: OccupancyMapPaths, pixel: RCCoords
) -> None:
    occ = OccupancyMap.from_yaml(sample_occupancy_map_and_yaml_paths.metadata)
    with pytest.raises(ValueError):
        occ.XY_from_pixel(pixel)
