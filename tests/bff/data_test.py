from typing import Callable, Sequence

import numpy as np
import pytest
import torchvision.transforms as transforms

from bff.data import (
    DiscreteDirectionalDataset,
    RandomHorizontalFlipPeopleFlow,
    RandomRotationPeopleFlow,
    RandomVerticalFlipPeopleFlow,
)


@pytest.mark.parametrize(
    ["d", "d_expected"],
    [
        (
            np.linspace(1.0, 8.0, 8),
            [1.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
        )
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
        )
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
