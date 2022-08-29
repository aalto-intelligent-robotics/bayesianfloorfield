from typing import Callable, Sequence

import numpy as np
import pytest
import torchvision.transforms as transforms
from directionalflow.data import (
    ConditionalDirectionalDataset,
    DiscreteDirectionalDataset,
    RandomHorizontalFlipPeopleFlow,
    RandomRotationPeopleFlow,
    RandomVerticalFlipPeopleFlow,
    get_conditional_prob,
    get_directional_prob,
)
from mod.Grid import Grid


@pytest.mark.parametrize(
    ["cell", "d_expected"],
    [
        ((0, 0), [0.5, 0, 0, 0, 0, 0.5, 0, 0]),
        ((0, 1), [0, 1.0, 0, 0, 0, 0, 0, 0]),
    ],
)
def test_get_directional_prob(
    cell: tuple, d_expected: Sequence, grid: Grid
) -> None:
    assert get_directional_prob(grid.cells[cell].bins) == d_expected


@pytest.mark.parametrize(
    ["cell", "d_expected"],
    [
        ((0, 0), [0.5] + [0.0] * 4 + [0.5] + [0.0] * 58),
        ((0, 1), [0.0] * 25 + [1.0] + [0.0] * 38),
    ],
)
def test_get_conditional_prob(
    cell: tuple, d_expected: Sequence, grid_conditional: Grid
) -> None:
    assert (
        get_conditional_prob(grid_conditional.cells[cell].model) == d_expected
    )


@pytest.mark.parametrize(
    ["d", "d_expected"],
    [
        (
            np.linspace(1.0, 8.0, 8),
            [1.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
        ),
        (
            np.linspace(1.0, 64.0, 64),
            [1.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]
            + [57.0, 64.0, 63.0, 62.0, 61.0, 60.0, 59.0, 58.0]
            + [49.0, 56.0, 55.0, 54.0, 53.0, 52.0, 51.0, 50.0]
            + [41.0, 48.0, 47.0, 46.0, 45.0, 44.0, 43.0, 42.0]
            + [33.0, 40.0, 39.0, 38.0, 37.0, 36.0, 35.0, 34.0]
            + [25.0, 32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0]
            + [17.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0]
            + [9.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0],
        ),
    ],
)
def test_vertical_flip(d: np.ndarray, d_expected: Sequence) -> None:
    flip = RandomVerticalFlipPeopleFlow(p=1)
    o = np.array([[[1, 1, 0], [0, 1, 0], [0, 0, 1]]])
    o_expected = np.array([[[0, 0, 1], [0, 1, 0], [1, 1, 0]]])
    o_flipped, d_flipped = flip((o, d))
    assert (o_flipped == o_expected).all()
    assert (d_flipped == d_expected).all()


@pytest.mark.parametrize(
    ["d", "d_expected"],
    [
        (
            np.linspace(1.0, 8.0, 8),
            [5.0, 4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0],
        ),
        (
            np.linspace(1.0, 64.0, 64),
            [37.0, 36.0, 35.0, 34.0, 33.0, 40.0, 39.0, 38.0]
            + [29.0, 28.0, 27.0, 26.0, 25.0, 32.0, 31.0, 30.0]
            + [21.0, 20.0, 19.0, 18.0, 17.0, 24.0, 23.0, 22.0]
            + [13.0, 12.0, 11.0, 10.0, 9.0, 16.0, 15.0, 14.0]
            + [5.0, 4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0]
            + [61.0, 60.0, 59.0, 58.0, 57.0, 64.0, 63.0, 62.0]
            + [53.0, 52.0, 51.0, 50.0, 49.0, 56.0, 55.0, 54.0]
            + [45.0, 44.0, 43.0, 42.0, 41.0, 48.0, 47.0, 46.0],
        ),
    ],
)
def test_horizontal_flip(d: np.ndarray, d_expected: Sequence) -> None:
    flip = RandomHorizontalFlipPeopleFlow(p=1)
    o = np.array([[[1, 1, 0], [0, 1, 0], [0, 0, 1]]])
    o_expected = np.array([[[0, 1, 1], [0, 1, 0], [1, 0, 0]]])
    o_flipped, d_flipped = flip((o, d))
    assert (o_flipped == o_expected).all()
    assert (d_flipped == d_expected).all()


@pytest.mark.parametrize(
    "d",
    [np.linspace(1.0, 8.0, 8), np.linspace(1.0, 64.0, 64)],
)
def test_p_zero_flip(d: np.ndarray) -> None:
    flip = RandomHorizontalFlipPeopleFlow(p=0)
    o = np.array([[[1, 1, 0], [0, 1, 0], [0, 0, 1]]])
    o_flipped, d_flipped = flip((o, d))
    assert (o_flipped == o).all()
    assert (d_flipped == d).all()


def test_data(dataset: DiscreteDirectionalDataset) -> None:
    assert len(dataset) == 2
    image, groundtruth = dataset[1]
    assert image.shape == (1, 2, 2)
    assert groundtruth.shape == (8,)
    assert (image == np.array([[[0, 0], [0, 1]]])).all()
    assert (groundtruth == np.array([0, 1, 0, 0, 0, 0, 0, 0])).all()


@pytest.mark.parametrize(
    [
        "transform",
        "image_expected",
        "groundtruth_expected_0",
        "groundtruth_expected_1",
    ],
    [
        (
            RandomRotationPeopleFlow(p=1, angles_p=[1, 0, 0]),
            np.array([[[0.0, 1.0], [0.0, 0.0]]]),
            np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5]),
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ),
        (
            RandomRotationPeopleFlow(p=1, angles_p=[0, 1, 0]),
            np.array([[[1.0, 0.0], [0.0, 0.0]]]),
            np.array([0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        ),
        (
            RandomRotationPeopleFlow(p=1, angles_p=[0, 0, 1]),
            np.array([[[0.0, 0.0], [1.0, 0.0]]]),
            np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        (
            RandomHorizontalFlipPeopleFlow(p=1),
            np.array([[[0.0, 0.0], [1.0, 0.0]]]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5]),
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ),
        (
            RandomVerticalFlipPeopleFlow(p=1),
            np.array([[[0.0, 1.0], [0.0, 0.0]]]),
            np.array([0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        (
            transforms.Compose(
                [
                    RandomHorizontalFlipPeopleFlow(p=1),
                    RandomVerticalFlipPeopleFlow(p=1),
                ]
            ),
            np.array([[[1.0, 0.0], [0.0, 0.0]]]),
            np.array([0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        ),
    ],
)
def test_data_transformed(
    dataset: DiscreteDirectionalDataset,
    transform: Callable,
    image_expected: np.ndarray,
    groundtruth_expected_0: np.ndarray,
    groundtruth_expected_1: np.ndarray,
) -> None:
    dataset.transform = transform
    assert len(dataset) == 2
    _, groundtruth_0 = dataset[0]
    image_1, groundtruth_1 = dataset[1]
    assert (image_1 == image_expected).all()
    assert (groundtruth_0 == groundtruth_expected_0).all()
    assert (groundtruth_1 == groundtruth_expected_1).all()


@pytest.mark.parametrize(
    [
        "transform",
        "image_expected",
        "st_0",
        "st_1",
        "groundtruth_expected_0",
        "groundtruth_expected_1",
    ],
    [
        (
            RandomRotationPeopleFlow(p=1, angles_p=[1, 0, 0]),
            np.array([[[0.0, 1.0], [0.0, 0.0]]]),
            2,
            5,
            np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5]),
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ),
        (
            RandomRotationPeopleFlow(p=1, angles_p=[0, 1, 0]),
            np.array([[[1.0, 0.0], [0.0, 0.0]]]),
            4,
            7,
            np.array([0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        ),
        (
            RandomRotationPeopleFlow(p=1, angles_p=[0, 0, 1]),
            np.array([[[0.0, 0.0], [1.0, 0.0]]]),
            6,
            1,
            np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        (
            RandomHorizontalFlipPeopleFlow(p=1),
            np.array([[[0.0, 0.0], [1.0, 0.0]]]),
            4,
            1,
            np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5]),
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ),
        (
            RandomVerticalFlipPeopleFlow(p=1),
            np.array([[[0.0, 1.0], [0.0, 0.0]]]),
            0,
            5,
            np.array([0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        (
            transforms.Compose(
                [
                    RandomHorizontalFlipPeopleFlow(p=1),
                    RandomVerticalFlipPeopleFlow(p=1),
                ]
            ),
            np.array([[[1.0, 0.0], [0.0, 0.0]]]),
            4,
            7,
            np.array([0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        ),
    ],
)
def test_data_transformed_conditional(
    dataset_conditional: ConditionalDirectionalDataset,
    transform: Callable,
    image_expected: np.ndarray,
    st_0: int,
    st_1: int,
    groundtruth_expected_0: np.ndarray,
    groundtruth_expected_1: np.ndarray,
) -> None:
    dataset_conditional.transform = transform
    assert len(dataset_conditional) == 2
    _, groundtruth_0 = dataset_conditional[0]
    image_1, groundtruth_1 = dataset_conditional[1]
    assert (image_1 == image_expected).all()
    assert (
        groundtruth_0[st_0 * 8 : st_0 * 8 + 8] == groundtruth_expected_0
    ).all()
    assert (
        groundtruth_1[st_1 * 8 : st_1 * 8 + 8] == groundtruth_expected_1
    ).all()
