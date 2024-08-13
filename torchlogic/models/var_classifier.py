import logging
from typing import List

from ..models.base import BaseVarNRNModel
from .mixins import ReasoningNetworkClassifierMixin


class VarNRNClassifier(BaseVarNRNModel, ReasoningNetworkClassifierMixin):

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
            var_emb_dim: int = 50,
            var_n_layers: int = 2,
            normal_form: str = 'dnf',
            logits: bool = False
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
                tau_min=0.2,
                tau_warmup=50,
                attn_emb_dim=50
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
            var_emb_dim (int): Embedding dimension for latent variational space.
        """
        ReasoningNetworkClassifierMixin.__init__(self, output_size=output_size)
        BaseVarNRNModel.__init__(
            self,
            target_names=target_names,
            input_size=input_size,
            output_size=output_size,
            layer_sizes=layer_sizes,
            feature_names=feature_names,
            swa=swa,
            weight_init=weight_init,
            add_negations=add_negations,
            var_emb_dim=var_emb_dim,
            var_n_layers=var_n_layers,
            normal_form=normal_form,
            logits=logits
        )
        self.set_modules(self.rn)
        self.logger = logging.getLogger(self.__class__.__name__)


__all__ = [VarNRNClassifier]
