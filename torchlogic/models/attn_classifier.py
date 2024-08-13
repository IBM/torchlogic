import logging
from typing import List, Union

import numpy as np

from ..models.base import BaseAttnNRNModel
from .mixins import ReasoningNetworkClassifierMixin


class AttnNRNClassifier(BaseAttnNRNModel, ReasoningNetworkClassifierMixin):

    def __init__(
            self,
            target_names: List[str],
            feature_names: List[str],
            input_size: int,
            output_size: int,
            layer_sizes: List[int],
            swa: bool = True,
            add_negations: bool = False,
            weight_init: float = 0.2,
            attn_emb_dim: int = 32,
            attn_n_layers: int = 2,
            normal_form: str = 'dnf',
            logits: bool = True
    ):
        """
        Initialize a AttnNRNClassifier model.

        Example:
            model = AttnNRNClassifier(
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
                attn_n_layers=2
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
        ReasoningNetworkClassifierMixin.__init__(self, output_size=output_size, logits=logits)
        BaseAttnNRNModel.__init__(
            self,
            target_names=target_names,
            input_size=input_size,
            output_size=output_size,
            layer_sizes=layer_sizes,
            feature_names=feature_names,
            swa=swa,
            weight_init=weight_init,
            add_negations=add_negations,
            attn_emb_dim=attn_emb_dim,
            attn_n_layers=attn_n_layers,
            normal_form=normal_form,
            logits=logits
        )
        self.set_modules(self.rn)
        self.logger = logging.getLogger(self.__class__.__name__)

    def explain(
            self,
            quantile: float = 0.5,
            required_output_thresholds: Union[float, np.float64] = 0.9,
            threshold: float = None,
            explain_type: str = 'both',
            print_type: str = 'logical',
            target_names: list = ['positive'],
            explanation_prefix: str = "A sample is in the",
            ignore_uninformative: bool = False,
            rounding_precision: int = 3,
            inverse_transform=None,
            decision_boundary: float = 0.5,
            show_bounds: bool = True,
            simplify: bool = False
    ) -> str:
        raise UserWarning("Global explanations are not available for AttentionNRNClassifier.")

    def print(
            self,
            quantile=0.5,
            required_output_thresholds=None,
            threshold=None,
            explain_type='both',
            print_type='logical',
            target_names=['positive'],
            ignore_uninformative: bool = False,
            rounding_precision: int = 3,
            inverse_transform=None,
            decision_boundary: float = 0.5,
            show_bounds: bool = True
    ):
        raise UserWarning("Global explanations are not available for AttentionNRNClassifier.")


__all__ = [AttnNRNClassifier]
