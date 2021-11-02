from deepflow.nets import DiscreteDirectional, ConditionalDiscreteDirectional
import mod.Models as mod


def test_discretedirectional_model():
    net = DiscreteDirectional()
    assert net.model == mod.DiscreteDirectional
    assert net.out_channels == 8


def test_conditionaldirectional_model():
    net = ConditionalDiscreteDirectional()
    assert net.model == mod.DiscreteConditionalDirectional
    assert net.out_channels == 64
