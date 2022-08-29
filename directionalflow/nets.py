from functools import partial
from typing import Type

import mod.Models as mod
import octopytorch as octo
from torch import nn

DynModel = Type[mod.DiscreteDirectional]


class PeopleFlow(octo.models.Tiramisu):
    def __init__(
        self, model: DynModel, window_size: int, out_channels: int
    ) -> None:
        self.model = model
        self.window_size = window_size

        module_bank = octo.DEFAULT_MODULE_BANK.copy()

        # Dropout
        module_bank[octo.ModuleType.DROPOUT] = partial(
            nn.Dropout2d, p=0.2, inplace=True
        )

        # Activations
        module_bank[octo.ModuleType.ACTIVATION] = nn.ReLU
        module_bank[octo.ModuleType.ACTIVATION_FINAL] = partial(
            nn.Softmax, dim=1
        )

        super().__init__(
            in_channels=1,  # Binary image
            out_channels=out_channels,  # N-channel output
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


class DiscreteDirectional(PeopleFlow):
    def __init__(self, window_size: int) -> None:
        super().__init__(
            model=mod.DiscreteDirectional,
            window_size=window_size,
            out_channels=8,
        )
