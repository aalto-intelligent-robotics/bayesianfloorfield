import mod.Models as mod
import pytest
import torch
from directionalflow.nets import (
    ConditionalDiscreteDirectional,
    DiscreteDirectional,
)


def test_discretedirectional_model() -> None:
    net = DiscreteDirectional(window_size=32)
    assert net.model == mod.DiscreteDirectional
    assert net.out_channels == 8


def test_conditionaldirectional_model() -> None:
    net = ConditionalDiscreteDirectional(window_size=32)
    assert net.model == mod.DiscreteConditionalDirectional
    assert net.out_channels == 64


def test_network_ouput() -> None:
    input = torch.zeros((2, 1, 32, 32))
    net = DiscreteDirectional(window_size=32)
    with torch.no_grad():
        output = net(input)
    assert output.shape == (2, 8, 32, 32)
    assert torch.sum(output[0, :, 16, 16]) == pytest.approx(1)
