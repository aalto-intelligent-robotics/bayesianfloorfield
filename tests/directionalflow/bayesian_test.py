import numpy as np
import pytest

from directionalflow.bayesian import BayesianMoD


def test_init() -> None:
    mod = BayesianMoD(
        resolution=1, origin=[0, 0], alpha=1, prior=np.ones([2, 2, 8]) * 1 / 8
    )
    assert (mod.posterior == mod.prior).all()


@pytest.mark.parametrize(
    ["resolution", "alpha", "prior"],
    [
        (-1, 1, np.ones([2, 2, 8]) * 1 / 8),
        (0, 1, np.ones([2, 2, 8]) * 1 / 8),
        (1, -1, np.ones([2, 2, 8]) * 1 / 8),
        (1, 0, np.ones([2, 2, 8]) * 1 / 8),
        (1, 1, np.ones([2, 2, 8])),
        (1, 0, -np.ones([2, 2, 8]) * 1 / 8),
    ],
)
def test_exceptions(
    resolution: float, alpha: float, prior: np.ndarray
) -> None:
    with pytest.raises(AssertionError):
        BayesianMoD(
            resolution=resolution, origin=[0, 0], alpha=alpha, prior=prior
        )


def test_posterior() -> None:
    mod = BayesianMoD(
        resolution=1, origin=[0, 0], alpha=4, prior=np.ones([1, 1, 4]) * 1 / 4
    )
    mod.add_observations((0, 0, 0), 5)
    mod.add_observations((0, 0, 1), 2)
    mod.add_observations((0, 0, 2))
    assert np.isclose(mod.posterior, [1 / 2, 1 / 4, 1 / 6, 1 / 12]).all()


def test_observations_exception() -> None:
    mod = BayesianMoD(
        resolution=1, origin=[0, 0], alpha=4, prior=np.ones([1, 1, 4]) * 1 / 4
    )
    with pytest.raises(AssertionError):
        mod.add_observations((0, 0, 0), -2)
