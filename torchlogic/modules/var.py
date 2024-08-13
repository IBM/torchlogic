import logging

import torch
from torch import nn

from ..nn import (VariationalLukasiewiczChannelOrBlock, VariationalLukasiewiczChannelAndBlock,
                  Predicates)


class VarNRNModule(nn.Module):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            layer_sizes: list,
            feature_names: list,
            add_negations: bool = False,
            weight_init: float = 0.2,
            var_emb_dim: int = 50,
            var_n_layers: int = 2,
            normal_form: str = 'dnf',
            logits: bool = False
    ):
        """
        Initialize a Attention Neural Reasoning Network module.

        Args:
            input_size (int): number of features from input.
            output_size (int): number of outputs.
            layer_sizes (list): A list containing the number of output logics for each layer.
            feature_names (list): A list of feature names.
            out_type (str): 'And' or 'Or'.  The logical type of the output layer.
            add_negations (bool): add negations of logic.
            weight_init (float): Upper bound of uniform weight initialization.  Lower bound is negated value.
            tau (float): Tau value for gumbel softmax
            var_emb_dim (int): Embedding dimension for latent variational space.
        """
        super(VarNRNModule, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer_sizes = layer_sizes
        self.feature_names = feature_names
        self.add_negations = add_negations
        self.weight_init = weight_init
        self.var_emb_dim = var_emb_dim
        self.var_n_layers = var_n_layers
        self.normal_form = normal_form
        self.logits = logits
        self.logger = logging.getLogger(self.__class__.__name__)

        assert self.normal_form in ['cnf', 'dnf'], "'normal_form' must be one of 'dnf', 'cnf'."

        # input layer
        if self.normal_form == 'dnf':
            input_layer = VariationalLukasiewiczChannelAndBlock(
                channels=output_size,
                in_features=input_size,
                out_features=layer_sizes[0],
                n_selected_features=input_size,
                parent_weights_dimension='out_features',
                operands=Predicates(feature_names=feature_names),
                outputs_key='0_and',
                add_negations=add_negations,
                weight_init=weight_init,
                var_emb_dim=var_emb_dim,
                var_n_layers=var_n_layers
            )
            setattr(self, '0_and', input_layer)
        else:
            input_layer = VariationalLukasiewiczChannelOrBlock(
                channels=output_size,
                in_features=input_size,
                out_features=layer_sizes[0],
                n_selected_features=input_size,
                parent_weights_dimension='out_features',
                operands=Predicates(feature_names=feature_names),
                outputs_key='0_or',
                add_negations=add_negations,
                weight_init=weight_init,
                var_emb_dim=var_emb_dim,
                var_n_layers=var_n_layers
            )
            setattr(self, '0_or', input_layer)

        model_layers = [input_layer]

        # add alternating internal layers
        for i in range(1, len(layer_sizes)):

            operands = model_layers[-1]

            if i % 2 == 0:
                if self.normal_form == 'dnf':
                    internal_layer = VariationalLukasiewiczChannelAndBlock(
                        channels=output_size,
                        in_features=layer_sizes[i - 1],
                        out_features=layer_sizes[i],
                        n_selected_features=layer_sizes[i - 1],
                        parent_weights_dimension='out_features',
                        operands=operands,
                        outputs_key=str(i) + '_and',
                        weight_init=weight_init,
                        var_emb_dim=var_emb_dim,
                        var_n_layers=var_n_layers
                    )
                    setattr(self, str(i) + '_and', internal_layer)
                else:
                    internal_layer = VariationalLukasiewiczChannelOrBlock(
                        channels=output_size,
                        in_features=layer_sizes[i],
                        out_features=layer_sizes[i],
                        n_selected_features=layer_sizes[i - 1],
                        parent_weights_dimension='out_features',
                        operands=operands,
                        outputs_key=str(i) + '_or',
                        weight_init=weight_init,
                        var_emb_dim=var_emb_dim,
                        var_n_layers=var_n_layers
                    )
                    setattr(self, str(i) + '_or', internal_layer)
            else:
                if self.normal_form == 'dnf':
                    internal_layer = VariationalLukasiewiczChannelOrBlock(
                        channels=output_size,
                        in_features=layer_sizes[i],
                        out_features=layer_sizes[i],
                        n_selected_features=layer_sizes[i],
                        parent_weights_dimension='out_features',
                        operands=operands,
                        outputs_key=str(i) + '_or',
                        weight_init=weight_init,
                        var_emb_dim=var_emb_dim,
                        var_n_layers=var_n_layers
                    )
                    setattr(self, str(i) + '_or', internal_layer)
                else:
                    internal_layer = VariationalLukasiewiczChannelAndBlock(
                        channels=output_size,
                        in_features=layer_sizes[i - 1],
                        out_features=layer_sizes[i],
                        n_selected_features=layer_sizes[i - 1],
                        parent_weights_dimension='out_features',
                        operands=operands,
                        outputs_key=str(i) + '_and',
                        weight_init=weight_init,
                        var_emb_dim=var_emb_dim,
                        var_n_layers=var_n_layers
                    )
                    setattr(self, str(i) + '_and', internal_layer)

            model_layers += [internal_layer]

        # OR LAYER
        if len(layer_sizes) % 2 == 0:
            if self.normal_form == 'dnf':
                self.output_layer = VariationalLukasiewiczChannelAndBlock(
                    channels=output_size,
                    in_features=layer_sizes[-1],
                    out_features=1,
                    n_selected_features=layer_sizes[-1],
                    parent_weights_dimension='out_features',
                    operands=model_layers[-1],
                    outputs_key='output_layer',
                    weight_init=weight_init,
                    var_emb_dim=var_emb_dim,
                    var_n_layers=var_n_layers
                )
            else:
                self.output_layer = VariationalLukasiewiczChannelOrBlock(
                    channels=output_size,
                    in_features=layer_sizes[-1],
                    out_features=1,
                    n_selected_features=layer_sizes[-1],
                    parent_weights_dimension='out_features',
                    operands=model_layers[-1],
                    outputs_key='output_layer',
                    weight_init=weight_init,
                    var_emb_dim=var_emb_dim,
                    var_n_layers=var_n_layers
                )
        else:
            if self.normal_form == 'dnf':
                self.output_layer = VariationalLukasiewiczChannelOrBlock(
                    channels=output_size,
                    in_features=layer_sizes[-1],
                    out_features=1,
                    n_selected_features=layer_sizes[-1],
                    parent_weights_dimension='out_features',
                    operands=model_layers[-1],
                    outputs_key='output_layer',
                    weight_init=weight_init,
                    var_emb_dim=var_emb_dim,
                    var_n_layers=var_n_layers
                )
            else:
                self.output_layer = VariationalLukasiewiczChannelAndBlock(
                    channels=output_size,
                    in_features=layer_sizes[-1],
                    out_features=1,
                    n_selected_features=layer_sizes[-1],
                    parent_weights_dimension='out_features',
                    operands=model_layers[-1],
                    outputs_key='output_layer',
                    weight_init=weight_init,
                    var_emb_dim=var_emb_dim,
                    var_n_layers=var_n_layers
                )

        self.model = torch.nn.Sequential(*model_layers)

        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        if self.logits:
            return torch.special.logit(self.output_layer(x).squeeze(-1).squeeze(-1), eps=1e-6)
        else:
            return self.output_layer(x).squeeze(-1).squeeze(-1)
