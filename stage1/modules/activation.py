import torch
from spikingjelly.activation_based.neuron import LIFNode
from spikingjelly.activation_based import surrogate
from typing import Union

surrogate_dict = {
    'sigmoid': surrogate.Sigmoid,
    'atan': surrogate.ATan,
    'leaky_relu': surrogate.LeakyKReLU,
}

class LIF(LIFNode):
    def __init__(self, tau=2.0, surrogate_function='sigmoid',
                 step_mode='m', decay_input=False, 
                 v_threshold=1.0, v_reset=None, detach_reset=True
                 ):

        super().__init__(float(tau), decay_input, v_threshold, v_reset,   # type: ignore
                         surrogate_function=surrogate_dict[surrogate_function](),
                         detach_reset=detach_reset, step_mode=step_mode)
        
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, None]:
        spike = super().forward(x)
        return spike
        