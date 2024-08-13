from typing import List
import logging

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel

from torchlogic.modules import AttentionNRNModule
from .rn import ReasoningNetworkModel


class BaseAttnNRNModel(ReasoningNetworkModel):

    def __init__(
            self,
            target_names: List[str],
            feature_names: List[str],
            input_size: int,
            output_size: int,
            layer_sizes: List[int],
            swa: bool = False,
            add_negations: bool = False,
            weight_init: float = 0.2,
            attn_emb_dim: int = 32,
            attn_n_layers: int = 2,
            normal_form: str = 'dnf',
            logits: bool = False
    ):
        """
        Initialize a Attention Neural Reasoning Network model.

        Example:
            model = BaseAttnNRNModel(
                target_names=['class1', 'class2'],
                feature_names=['feature1', 'feature2', 'feature3'],
                input_size=3,
                output_size=2,
                layer_sizes=[3, 3]
                out_type='And',
                swa=False,
                add_negations=True,
                weight_init=0.2,
                attn_emb_dim=32,
                attn_n_layer=2
            )

        Args:
            target_names (list): A list of the target names.
            feature_names (list): A list of feature names.
            input_size (int): number of features from input.
            output_size (int): number of outputs.
            layer_sizes (list): A list containing the number of output logics for each layer.
            out_type (str): 'And' or 'Or'.  The logical type of the output layer.
            swa (bool): Use stochastic weight averaging
            add_negations (bool): add negations of logic.
            weight_init (float): Upper bound of uniform weight initialization.  Lower bound is negated value.
            attn_emb_dim (int): Hidden layer size for attention model.
            attn_n_layers (int): Number of layers for attention model.
        """
        ReasoningNetworkModel.__init__(self)

        self.target_names = target_names
        self.feature_names = feature_names
        self.input_size = input_size
        self.output_size = output_size
        self.swa = swa
        self.add_negations = add_negations
        self.weight_init = weight_init
        self.attn_emb_dim = attn_emb_dim
        self.attn_n_layers = attn_n_layers
        self.normal_form = normal_form
        self.logits = logits

        self.rn = AttentionNRNModule(
            input_size=input_size,
            output_size=output_size,
            layer_sizes=layer_sizes,
            feature_names=feature_names,
            add_negations=add_negations,
            weight_init=weight_init,
            attn_emb_dim=attn_emb_dim,
            attn_n_layers=attn_n_layers,
            normal_form=normal_form,
            logits=logits
        )

        self.averaged_rn = None
        if self.swa:
            self.averaged_rn = AveragedModel(self.rn)

        self.target_names = target_names

        if torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs!")
            self.rn = nn.DataParallel(self.rn)
        if self.USE_CUDA:
            self.logger.info(f"Using GPU")
            self.rn = self.rn.cuda()
        elif self.USE_MPS:
            self.logger.info(f"Using MPS")
            self.rn = self.rn.to('mps')

        self.USE_DATA_PARALLEL = isinstance(self.rn, torch.nn.DataParallel)

        self.logger = logging.getLogger(self.__class__.__name__)


__all__ = [BaseAttnNRNModel]
