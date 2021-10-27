from functools import partial
from typing import Type, Union

import mod.Models as mod
import numpy as np
import octopytorch as octo
import torch
from mod.Grid import Grid
from mod.OccupancyMap import OccupancyMap
from torch import nn

from deepflow.data import Window

DynModel = Union[
    Type[mod.DiscreteDirectional], Type[mod.DiscreteConditionalDirectional]
]


class PeopleFlow(octo.models.Tiramisu):
    def __init__(self, model: DynModel, out_channels: int) -> None:
        self.model = model

        module_bank = octo.DEFAULT_MODULE_BANK.copy()

        # Dropout
        module_bank[octo.ModuleType.DROPOUT] = partial(
            nn.Dropout2d, p=0.2, inplace=True
        )
        # Every activation in the model is going to be a GELU (Gaussian Error
        # Linear Units function). GELU(x) = x * Φ(x) See:
        # https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
        module_bank[octo.ModuleType.ACTIVATION] = nn.ReLU
        # Example for segmentation:
        # module_bank[octo.ModuleType.ACTIVATION_FINAL] =
        # partial(nn.LogSoftmax, dim=1) Example for regression (default):
        module_bank[octo.ModuleType.ACTIVATION_FINAL] = nn.Identity

        super().__init__(
            in_channels=1,  # RGB images
            out_channels=out_channels,  # 5-channel output (5 classes)
            init_conv_filters=48,  # Channels outputted by the 1st convolution
            structure=(
                [4, 4, 4, 4, 4],  # Down blocks
                4,  # bottleneck layers
                [4, 4, 4, 4, 4],  # Up blocks
            ),
            growth_rate=12,  # Growth rate of the DenseLayers
            compression=1.0,  # No compression
            early_transition=False,  # No early transition
            include_top=True,  # Includes last layer and activation
            checkpoint=False,  # No memory checkpointing
            module_bank=module_bank,  # Modules to use
        )

        # Initializes all the convolutional kernel weights.
        self.initialize_kernels(nn.init.kaiming_uniform_, conv=True)

    def estimate_model(self, map: OccupancyMap) -> Grid:
        grid = Grid(
            origin=map.origin, resolution=map.resolution, model=self.model
        )
        binary_map = map.binary_map
        sz = binary_map.size
        window = Window(32)
        for row in range(sz[0]):
            for column in range(sz[1]):
                center = (row, column)
                input = torch.tensor(
                    np.asarray(
                        binary_map.crop(window.corners(center)),
                        "float",
                    )
                )
                output = self.forward(input)
                for key in window.indeces(
                    center,
                    bounds=[0, sz[0], 0, sz[1]],
                ):
                    value = output[key[0], key[1], :]
                    if key not in grid.cells:
                        grid.cells[key] = self.model(
                            index=(key[0], key[1]),
                            coords=(
                                key[0] * map.resolution + map.origin[0],
                                key[1] * map.resolution + map.origin[1],
                            ),
                        )
                    grid.total_count = grid.total_count + sum(value)
                    grid.cells[key].add_data(value)
        grid.update_model()
        return grid


class DiscreteDirectional(PeopleFlow):
    def __init__(self) -> None:
        super().__init__(model=mod.DiscreteDirectional, out_channels=8)


class ConditionalDiscreteDirectional(PeopleFlow):
    def __init__(self) -> None:
        super().__init__(
            model=mod.DiscreteConditionalDirectional, out_channels=64
        )
