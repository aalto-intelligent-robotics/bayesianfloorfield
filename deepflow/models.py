import numpy as np
import torch
from mod.OccupancyMap import OccupancyMap

from deepflow.data import Window
from deepflow.nets import PeopleFlow


class DynamicMap:
    def __init__(self, net: PeopleFlow, occupancy: OccupancyMap) -> None:
        self.net = net
        self.occupancy = occupancy.binary_map
        self.size = self.occupancy.size
        self._estimate_model()

    def _estimate_model(self) -> None:
        map = np.zeros((self.size[0], self.size[1], 8), "float")

        window = Window(32)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for row in range(self.size[0]):
            for column in range(self.size[1]):
                center = (row, column)
                crop = np.asarray(
                    self.occupancy.crop(window.corners(center)),
                    "float",
                )
                input = torch.Tensor(crop, device)
                output = self.net.forward(input)

                map[row, column] = output[row, column, :]
        self.map = map
