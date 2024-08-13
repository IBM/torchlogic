import torch
from .base import BaseConcatenateBlocksLogic


class ConcatenateBlocksLogic(BaseConcatenateBlocksLogic):

    def __init__(self, modules, outputs_key):
        super(ConcatenateBlocksLogic, self).__init__(modules, outputs_key)

    def forward(self, *inputs):
        """
        A channel logical xor.

        Args:
            *inputs: comma separated tensors of [BATCH_SIZE, CHANNELS, 1, 1]

        Returns:
            out: output tensor [BATCH_SIZE, OUT_CHANNELS, 1, # of input tensors]
        """
        return torch.cat(inputs, dim=-1)
