from functools import partial

import octopytorch as octo
from torch import nn


class PeopleFlow(octo.models.Tiramisu):
    def __init__(self, out_channels: int) -> None:
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


class DiscreteDirectional(PeopleFlow):
    def __init__(self) -> None:
        super().__init__(out_channels=8)


class ConditionalDiscreteDirectional(PeopleFlow):
    def __init__(self) -> None:
        super().__init__(out_channels=32)
