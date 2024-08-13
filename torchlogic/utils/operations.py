import math
from typing import List
from typing import Optional as _Optional

import torch
from torch import nn
from torch import Tensor

import numpy as np


def val_clamp(x, _min: float = 0.0, _max: float = 1.0) -> torch.Tensor:
    """gradient-transparent clamping to clamp values between [min, max]"""
    clamp_min = (x.detach() - _min).clamp(max=0)
    clamp_max = (x.detach() - _max).clamp(min=0)
    return x - clamp_max - clamp_min


def soft_val_clamp(x):
    m = torch.nn.Sigmoid()
    return m(x)


def alpha_clamp(x: torch.Tensor, alpha: float):
    """
    Clamps values lte to alph to zero and gte to alpha to one.
    """
    assert 1.0 >= alpha >= 0.0, "alpha must be between 0 and 1."

    with torch.no_grad():
        true_mask = torch.Tensor(x > (1.0 - alpha))
        false_mask = torch.Tensor(x <= alpha)
        true_delta = (1.0 - x) * true_mask.to(torch.float)
        false_delta = -1.0 * x * false_mask.to(torch.float)
    return x + true_delta + false_delta


def minmax_scale(input_tensor: torch.Tensor, _min: float = 0, _max: float = 1) -> torch.Tensor:
    """scale a tensor between _min and _max"""
    v_min, v_max = input_tensor.min(), input_tensor.max()
    return (input_tensor - v_min) / (v_max - v_min) * (_max - _min) + _min


def get_outputs(name, outputs):
    def hook(model, input, output):
        outputs[name] = output.detach()

    return hook


def register_hooks(model, outputs):
    for x in model.named_children():
        x[1].register_forward_hook(get_outputs(x[0], outputs))
        if not isinstance(x[1], nn.ModuleList):
            register_hooks(x[1], outputs)



def adjust_alpha(alpha: float, eps: float = 1e-8) -> float:
    """
    Adjust the alpha value such that it is in (0.5, 1]

    Returns:
        Dict: relations as groundings
    """
    if alpha == 0.5:
        alpha += eps
        return alpha
    else:
        return alpha


def chunk_inputs(
        inputs: List,
        fan_in: int = 4,
        inputs_probs: List[float] = None
) -> List:
    """
    Chunk the inputs according to the selection criteria mode

    Args:
        inputs (List): List of inputs
        fan_in (int): number of inputs to group together
        inputs_probs (List[float]): list of probabilities to use for sampling inputs

    Returns:
        List: list of chunked inputs
    """
    inputs = np.random.choice(inputs, size=len(inputs), replace=False, p=inputs_probs)
    chunked_inputs = [inputs[i:i + fan_in] for i in range(0, len(inputs), fan_in)]

    if len(chunked_inputs) == 0:
        raise AssertionError("Chunking inputs resulted in list of zero length.  Try adjusting drop rates.")

    return chunked_inputs


def exu(x, weight, bias):
    """ExU hidden unit modification."""
    return torch.exp(weight) * (x - bias)


class EXU(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((1, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((1, out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.trunc_normal_(self.weight, mean=4, std=0.5, a=3.0, b=5.0)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.5, a=-1.0, b=1.0)

    @staticmethod
    def exu(input, weight, bias):
        """ExU hidden unit modification."""
        return torch.exp(weight) * (input - bias)

    def forward(self, input: Tensor) -> Tensor:
        return self.exu(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


def _no_grad_normal_(tensor, mean, std, generator=None):
    with torch.no_grad():
        return tensor.normal_(mean, std, generator=generator)


def _no_grad_uniform_(tensor, a, b, generator=None):
    with torch.no_grad():
        return tensor.uniform_(a, b, generator=generator)


def lukasiewicz_channel_block_xavier_uniform_(
    tensor: Tensor, gain: float = 1.0, generator: _Optional[torch.Generator] = None
) -> Tensor:
    r"""Fill the input `Tensor` with values using a Xavier uniform distribution.

    The method is described in `Understanding the difficulty of training
    deep feedforward neural networks` - Glorot, X. & Bengio, Y. (2010).
    The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """
    fan_in = tensor.size(1)
    fan_out = 1

    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a, generator)


def lukasiewicz_channel_block_xavier_normal_(
    tensor: Tensor,
    gain: float = 1.0,
    generator: _Optional[torch.Generator] = None,
) -> Tensor:
    r"""Fill the input `Tensor` with values using a Xavier normal distribution.

    The method is described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010). The resulting tensor
    will have values sampled from :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    """
    fan_in = tensor.size(1)
    fan_out = 1

    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    return _no_grad_normal_(tensor, 0., std, generator)


__all__ = [soft_val_clamp, minmax_scale, adjust_alpha, chunk_inputs]
