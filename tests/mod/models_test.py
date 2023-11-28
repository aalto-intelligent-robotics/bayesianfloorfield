import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mod.models import (
    BayesianDiscreteDirectional,
    Cell,
    DiscreteDirectional,
    RCCoords,
    XYCoords,
)


def test_xy_coord() -> None:
    c = XYCoords(1.23, 2.11)
    assert c.x == 1.23
    assert c.y == 2.11


def test_rc_coord() -> None:
    c = RCCoords(1, 2)
    assert c.row == 1
    assert c.column == 2


def test_cell_init() -> None:
    cell = Cell(coords=XYCoords(-1, 1), index=RCCoords(10, 20), resolution=2)
    assert cell.coords.x == -1 and cell.coords.y == 1
    assert cell.index.row == 10 and cell.index.column == 20
    assert cell.center.x == 0 and cell.center.y == 2
    assert cell.observation_count == 0
    assert cell.probability == 0


def test_cell_frozen_fields() -> None:
    cell = Cell(coords=XYCoords(-1, 1), index=RCCoords(10, 20), resolution=2)
    with pytest.raises(AttributeError):
        cell.coords.x = 1  # type: ignore
    with pytest.raises(AttributeError):
        cell.coords.y = 1  # type: ignore
    with pytest.raises(AttributeError):
        cell.index.row = 1  # type: ignore
    with pytest.raises(AttributeError):
        cell.index.column = 1  # type: ignore
    with pytest.raises(ValidationError):
        cell.coords = (2.3, 1.2)  # type: ignore
    with pytest.raises(ValidationError):
        cell.index = (3, 1)  # type: ignore
    with pytest.raises(ValidationError):
        cell.resolution = 4


def test_cell_validation() -> None:
    with pytest.raises(ValidationError):
        # negative resolution
        cell = Cell(
            coords=XYCoords(-1, 1), index=RCCoords(10, 20), resolution=-1
        )
    with pytest.raises(ValidationError):
        # negative index
        cell = Cell(
            coords=XYCoords(-1, 1), index=RCCoords(10, -20), resolution=1
        )

    cell = Cell(coords=XYCoords(-1, 1), index=RCCoords(10, 20), resolution=2)
    with pytest.raises(ValidationError):
        cell.probability = -0.4  # negative probability
    with pytest.raises(ValidationError):
        cell.probability = 1.2  # probability > 1


def test_cell_add_data(sample_data: pd.DataFrame) -> None:
    cell = Cell(coords=XYCoords(-1, 1), index=RCCoords(10, 20), resolution=2)

    cell.add_data(sample_data)
    assert cell.observation_count == 9
    cell.add_data(sample_data)
    assert cell.observation_count == 18


def test_cell_update_model(sample_data: pd.DataFrame) -> None:
    cell = Cell(coords=XYCoords(-1, 1), index=RCCoords(10, 20), resolution=2)

    cell.update_model(total_observations=0)
    assert cell.probability == 0

    cell.update_model(total_observations=10)
    assert cell.probability == 0

    cell.add_data(sample_data)
    with pytest.raises(ValueError):
        cell.update_model(total_observations=0)

    cell.update_model(total_observations=9)
    assert cell.probability == 1

    cell.update_model(total_observations=18)
    assert cell.probability == 0.5

    with pytest.raises(ValidationError):
        cell.update_model(total_observations=len(sample_data.index) - 1)


def test_discr_dir_init() -> None:
    cell = DiscreteDirectional(
        coords=XYCoords(-1, 1), index=RCCoords(10, 20), resolution=2
    )
    assert cell.bin_count == 8
    assert cell.half_split == pytest.approx(0.39269908)
    assert len(cell.directions) == 8
    assert len(cell.bins) == 8
    assert sum(cell.bin_probabilities) == pytest.approx(1)

    cell = DiscreteDirectional(
        coords=XYCoords(-1, 1),
        index=RCCoords(10, 20),
        resolution=2,
        bin_count=4,
    )
    assert cell.bin_count == 4
    assert cell.half_split == pytest.approx(0.39269908 * 2)
    assert len(cell.directions) == 4
    assert len(cell.bins) == 4
    assert sum(cell.bin_probabilities) == pytest.approx(1)


def test_discr_dir_bin_from_angle() -> None:
    cell_8 = DiscreteDirectional(
        coords=XYCoords(-1, 1), index=RCCoords(10, 20), resolution=2
    )
    cell_3 = DiscreteDirectional(
        coords=XYCoords(-1, 1),
        index=RCCoords(10, 20),
        resolution=2,
        bin_count=3,
    )

    rads = np.arange(0, 2 * np.pi, np.pi / 24)
    negative_rads = np.arange(-2 * np.pi, 0, np.pi / 24)
    double_rads = np.arange(2 * np.pi, 4 * np.pi, np.pi / 24)
    expected_8 = (
        [0] * 4
        + [1] * 6
        + [2] * 6
        + [3] * 6
        + [4] * 6
        + [5] * 6
        + [6] * 6
        + [7] * 6
        + [0] * 2
    )
    expected_3 = [0] * 9 + [1] * 16 + [2] * 16 + [0] * 7

    assert [cell_8.bin_from_angle(rad) for rad in rads] == expected_8
    assert [cell_3.bin_from_angle(rad) for rad in rads] == expected_3
    assert [cell_8.bin_from_angle(rad) for rad in negative_rads] == expected_8
    assert [cell_3.bin_from_angle(rad) for rad in negative_rads] == expected_3
    assert [cell_8.bin_from_angle(rad) for rad in double_rads] == expected_8
    assert [cell_3.bin_from_angle(rad) for rad in double_rads] == expected_3


def test_discr_dir_update_model(sample_data: pd.DataFrame) -> None:
    cell = DiscreteDirectional(
        coords=XYCoords(-1, 1), index=RCCoords(10, 20), resolution=2
    )
    cell.update_model(total_observations=0)
    assert cell.probability == 0
    assert all([bin.probability == 1 / 8 for bin in cell.bins])

    cell.update_model(total_observations=10)
    assert cell.probability == 0
    assert all([bin.probability == 1 / 8 for bin in cell.bins])

    cell.add_data(sample_data)
    with pytest.raises(ValueError):
        cell.update_model(total_observations=0)

    cell.update_model(total_observations=9)
    assert cell.probability == 1
    assert cell.bin_probabilities == [0, 0, 0, 3 / 9, 4 / 9, 0, 0, 2 / 9]

    with pytest.raises(ValidationError):
        cell.update_model(total_observations=len(sample_data.index) - 1)


def test_bayes_discr_dir_init() -> None:
    cell = BayesianDiscreteDirectional(
        coords=XYCoords(-1, 1), index=RCCoords(10, 20), resolution=2
    )
    assert cell.bin_count == 8
    assert cell.half_split == pytest.approx(0.39269908)
    assert len(cell.directions) == 8
    assert len(cell.bins) == 8
    assert sum(cell.bin_probabilities) == pytest.approx(1)
    assert cell.alpha == 0
    assert all([prior == 1 / 8 for prior in cell.priors])

    cell = BayesianDiscreteDirectional(
        coords=XYCoords(-1, 1),
        index=RCCoords(10, 20),
        resolution=2,
        bin_count=4,
    )
    assert cell.bin_count == 4
    assert cell.half_split == pytest.approx(0.39269908 * 2)
    assert len(cell.directions) == 4
    assert len(cell.bins) == 4
    assert sum(cell.bin_probabilities) == pytest.approx(1)
    assert cell.alpha == 0
    assert all([prior == 1 / 4 for prior in cell.priors])


def test_bayes_discr_dir_update_model(sample_data: pd.DataFrame) -> None:
    cell = BayesianDiscreteDirectional(
        coords=XYCoords(-1, 1), index=RCCoords(10, 20), resolution=2
    )
    cell.update_model(total_observations=0)
    assert cell.probability == 0
    assert all([bin.probability == 1 / 8 for bin in cell.bins])

    cell.update_model(total_observations=10)
    assert cell.probability == 0
    assert all([bin.probability == 1 / 8 for bin in cell.bins])

    cell.add_data(sample_data)
    with pytest.raises(ValueError):
        cell.update_model(total_observations=0)

    cell.update_model(total_observations=9)
    assert cell.probability == 1
    assert cell.bin_probabilities == [0, 0, 0, 3 / 9, 4 / 9, 0, 0, 2 / 9]

    with pytest.raises(ValidationError):
        cell.update_model(total_observations=len(sample_data.index) - 1)
