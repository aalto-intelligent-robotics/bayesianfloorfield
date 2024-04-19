from functools import partial
from typing import Type

from torch import Tensor, load, nn

import mod.models as mod
import octopytorch as octo

DynModel = Type[mod.DiscreteDirectional]


class PeopleFlow(octo.models.Tiramisu):
    """The base class for a network learning people flow from occupancy"""

    def __init__(
        self, model: DynModel, window_size: int, out_channels: int
    ) -> None:
        """Init `PeopleFlow` class

        Args:
            model (DynModel): The dynamic model this network is learning from.
            window_size (int): The size of the window over the occupancy used
            as network input.
            out_channels (int): The number of output channels, matching the
            number of discrete directions in the dynamic model.
        """
        self.model = model
        self.window_size = window_size
        self.window_center = window_size // 2

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

    def forward(self, inp: Tensor) -> Tensor:
        y_pred = super().forward(inp)
        return y_pred[..., self.window_center, self.window_center]

    def load_weights(self, checkpoint_path: str) -> None:
        """Loads pre-computed weights from `checkpoint_path`"""
        checkpoint = load(checkpoint_path)["model_state_dict"]
        self.load_state_dict(checkpoint)


class DiscreteDirectional(PeopleFlow):
    """A network learning an 8-directional Floor Field from occupancy"""

    def __init__(self, window_size: int) -> None:
        """Init `DiscreteDirectional` class

        Args:
            window_size (int): The size of the window over the occupancy used
            as network input.
        """
        super().__init__(
            model=mod.DiscreteDirectional,
            window_size=window_size,
            out_channels=8,
        )
