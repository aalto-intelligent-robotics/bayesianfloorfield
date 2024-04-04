import pytest
import torch

import mod.models as mod
from bff.nets import DiscreteDirectional


def test_discretedirectional_model() -> None:
    net = DiscreteDirectional(window_size=32)
    assert net.model == mod.DiscreteDirectional
    assert net.out_channels == 8


def test_network_ouput() -> None:
    input = torch.zeros((2, 1, 32, 32))
    net = DiscreteDirectional(window_size=32)
    with torch.no_grad():
        output = net(input)
    assert output.shape == (2, 8)
    assert torch.sum(output[0, :]) == pytest.approx(1)
    assert torch.sum(output[1, :]) == pytest.approx(1)
