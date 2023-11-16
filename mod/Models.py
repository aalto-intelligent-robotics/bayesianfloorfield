import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("Model Logger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
ch.setFormatter(formatter)
logger.addHandler(ch)

_2PI = np.pi * 2


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x) % _2PI
    return (rho, phi)


class Cell:
    # TODO Description
    def __init__(self, coords, index, resolution):
        # logging
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.DEBUG)
        self.log.addHandler(ch)

        self.resolution = resolution
        self.coords = {"x": coords[1], "y": coords[0]}
        self.index = {"row": index[0], "column": index[1]}
        self.data = None
        self.observation_count = 0
        self.probability = 0
        # logging
        self.log.info("At creation parameters are {}".format(vars(self)))

    @property
    def center(self):
        return {
            "x": self.coords["x"] + self.resolution / 2,
            "y": self.coords["y"] + self.resolution / 2,
        }

    def add_data(self, data):
        # TODO Description
        if self.data is None:
            self.data = pd.DataFrame()

        self.data = self.data.append(pd.DataFrame(data).transpose())
        self.observation_count = self.observation_count + 1
        # logging
        self.log.info(
            "Cell {} now have {}({}) data points".format(
                self.index, len(self.data.index), self.observation_count
            )
        )


class DiscreteDirectional(Cell):
    """
    Floor Filed
    """

    def __init__(self, coords, index, resolution, bin_count=8):
        super().__init__(coords, index, resolution)
        self.half_split = np.pi / bin_count
        self.directions = np.arange(0, _2PI, _2PI / bin_count)

        self.bins = {
            d: {"data": pd.DataFrame(), "probability": 0, "mean": 0}
            for d in self.directions
        }

    def __get_bin(self, orientation):
        for i, d in enumerate(self.directions):
            if np.abs(d - (orientation % _2PI)) <= self.half_split:
                return i
        return 0

    def update_model(self, total_observations):
        # TODO implement computation of mean
        self.probability = self.observation_count / total_observations
        bins = self.data["motion_angle"].apply(self.__get_bin).to_numpy()
        for i, b in enumerate(bins):
            direction = self.directions[b]
            self.bins[direction]["data"] = self.bins[direction]["data"].append(
                self.data.iloc[i]
            )
        for key in self.bins:
            self.bins[key]["probability"] = float(
                len(self.bins[key]["data"].index)
            ) / float(len(self.data.index))


class BayesianDiscreteDirectional(Cell):
    """
    Bayesian Floor Filed
    """

    def __init__(self, coords, index, resolution, bin_count=8):
        super().__init__(coords, index, resolution)
        self.half_split = np.pi / bin_count
        self.directions = np.arange(0, _2PI, _2PI / bin_count)
        self.priors = np.ones_like(self.directions) / bin_count
        self.alpha = 0.0

        self.bins = {
            d: {"data": pd.DataFrame(), "probability": 0, "mean": 0}
            for d in self.directions
        }

    def assign_prior(self, priors: np.ndarray, alpha: float):
        assert self.priors.shape == priors.shape
        assert (priors >= 0).all(), "Prior contains negative probabilities"
        assert np.isclose(
            np.sum(priors), 1
        ), "Not all probability distributions in the prior sum to 1"
        assert alpha > 0, "Alpha should be positive"
        self.priors = priors
        self.alpha = alpha

    def __get_bin(self, orientation):
        for i, d in enumerate(self.directions):
            if np.abs(d - (orientation % _2PI)) <= self.half_split:
                return i
        return 0

    def update_model(self, total_observations):
        # TODO implement computation of mean
        self.probability = self.observation_count / total_observations
        bins = self.data["motion_angle"].apply(self.__get_bin).to_numpy()
        for i, b in enumerate(bins):
            direction = self.directions[b]
            self.bins[direction]["data"] = self.bins[direction]["data"].append(
                self.data.iloc[i]
            )
        for i, key in enumerate(self.bins):
            posterior = self.priors[i] * self.alpha + len(
                self.bins[key]["data"].index
            )
            self.bins[key]["probability"] = posterior / (
                np.sum(self.priors) * self.alpha + len(self.data.index)
            )
