import logging

import torch
from torch import nn

from ..nn import LukasiewiczChannelOrBlock, LukasiewiczChannelAndBlock, Predicates


class BanditNRNModule(nn.Module):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            layer_sizes: list,
            feature_names: list,
            n_selected_features_input: int,
            n_selected_features_internal: int,
            n_selected_features_output: int,
            perform_prune_quantile: float,
            ucb_scale: float,
            normal_form: str = 'dnf',
            add_negations: bool = False,
            weight_init: float = 0.2,
            logits: bool = False
    ):
        """
        Initialize a Bandit Reinforced Neural Reasoning Network module.

        Args:
            input_size (int): number of features from input.
            output_size (int): number of outputs.
            layer_sizes (list): A list containing the number of output logics for each layer.
            feature_names (list): A list of feature names.
            n_selected_features_input (int): The number of features to include in each logic in the input layer.
            n_selected_features_internal (int): The number of logics to include in each logic in the internal layers.
            n_selected_features_output (int): The number of logics to include in each logic in the output layer.
            perform_prune_quantile (float): The quantile to use for pruning randomized RN.
            ucb_scale (float): The scale of the confidence interval in the multi-armed bandit policy.
                               c = 1.96 is a 95% confidence interval.
            normal_form (str): 'dnf' for disjunctive normal form network; 'cnf' for conjunctive normal form network.
            add_negations (bool): add negations of logic.
            weight_init (float): Upper bound of uniform weight initialization.  Lower bound is negated value.
        """
        super(BanditNRNModule, self).__init__()
        self.ucb_scale = ucb_scale
        self.input_size = input_size
        self.output_size = output_size
        self.layer_sizes = layer_sizes
        self.feature_names = feature_names
        self.n_selected_features_input = n_selected_features_input
        self.n_selected_features_internal = n_selected_features_internal
        self.n_selected_features_output = n_selected_features_output
        self.perform_prune_quantile = perform_prune_quantile
        self.normal_form = normal_form
        self.add_negations = add_negations
        self.weight_init = weight_init
        self.logits = logits
        self.logger = logging.getLogger(self.__class__.__name__)

        assert self.normal_form in ['cnf', 'dnf'], "'normal_form' must be one of 'dnf', 'cnf'."

        # input layer
        if self.normal_form == 'dnf':
            input_layer = LukasiewiczChannelAndBlock(
                channels=output_size,
                in_features=input_size,
                out_features=layer_sizes[0],
                n_selected_features=n_selected_features_input,
                parent_weights_dimension='out_features',
                operands=Predicates(feature_names=feature_names),
                outputs_key='0',
                add_negations=add_negations,
                weight_init=weight_init
            )
        else:
            input_layer = LukasiewiczChannelOrBlock(
                channels=output_size,
                in_features=input_size,
                out_features=layer_sizes[0],
                n_selected_features=n_selected_features_input,
                parent_weights_dimension='out_features',
                operands=Predicates(feature_names=feature_names),
                outputs_key='0',
                add_negations=add_negations,
                weight_init=weight_init
            )

        model_layers = [input_layer]

        # add alternating internal layers
        for i in range(1, len(layer_sizes)):

            operands = model_layers[-1]

            if i % 2 == 0:
                if self.normal_form == 'dnf':
                    internal_layer = LukasiewiczChannelAndBlock(
                        channels=output_size,
                        in_features=layer_sizes[i - 1],
                        out_features=layer_sizes[i],
                        n_selected_features=n_selected_features_internal,
                        parent_weights_dimension='out_features',
                        operands=operands,
                        outputs_key=str(i),
                        weight_init=weight_init
                    )
                else:
                    internal_layer = LukasiewiczChannelOrBlock(
                        channels=output_size,
                        in_features=layer_sizes[i],
                        out_features=layer_sizes[i],
                        n_selected_features=n_selected_features_internal,
                        parent_weights_dimension='out_features',
                        operands=operands,
                        outputs_key=str(i),
                        weight_init=weight_init
                    )

            else:
                if self.normal_form == 'dnf':
                    internal_layer = LukasiewiczChannelOrBlock(
                        channels=output_size,
                        in_features=layer_sizes[i],
                        out_features=layer_sizes[i],
                        n_selected_features=n_selected_features_internal,
                        parent_weights_dimension='out_features',
                        operands=operands,
                        outputs_key=str(i),
                        weight_init=weight_init
                    )
                else:
                    internal_layer = LukasiewiczChannelAndBlock(
                        channels=output_size,
                        in_features=layer_sizes[i - 1],
                        out_features=layer_sizes[i],
                        n_selected_features=n_selected_features_internal,
                        parent_weights_dimension='out_features',
                        operands=operands,
                        outputs_key=str(i),
                        weight_init=weight_init
                    )

            model_layers += [internal_layer]

        # OR LAYER
        if len(layer_sizes) % 2 == 0:
            if self.normal_form == 'dnf':
                self.output_layer = LukasiewiczChannelAndBlock(
                    channels=output_size,
                    in_features=layer_sizes[-1],
                    out_features=1,
                    n_selected_features=n_selected_features_output,
                    parent_weights_dimension='out_features',
                    operands=model_layers[-1],
                    outputs_key='output_layer',
                    weight_init=weight_init
                )
            else:
                self.output_layer = LukasiewiczChannelOrBlock(
                    channels=output_size,
                    in_features=layer_sizes[-1],
                    out_features=1,
                    n_selected_features=n_selected_features_output,
                    parent_weights_dimension='out_features',
                    operands=model_layers[-1],
                    outputs_key='output_layer',
                    weight_init=weight_init
                )
        else:
            if self.normal_form == 'dnf':
                self.output_layer = LukasiewiczChannelOrBlock(
                    channels=output_size,
                    in_features=layer_sizes[-1],
                    out_features=1,
                    n_selected_features=n_selected_features_output,
                    parent_weights_dimension='out_features',
                    operands=model_layers[-1],
                    outputs_key='output_layer',
                    weight_init=weight_init
                )
            else:
                self.output_layer = LukasiewiczChannelAndBlock(
                    channels=output_size,
                    in_features=layer_sizes[-1],
                    out_features=1,
                    n_selected_features=n_selected_features_output,
                    parent_weights_dimension='out_features',
                    operands=model_layers[-1],
                    outputs_key='output_layer',
                    weight_init=weight_init
                )

        self.model = torch.nn.Sequential(*model_layers)

        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        if self.logits:
            return torch.special.logit(self.output_layer(x).squeeze(-1).squeeze(-1), eps=1e-6)
        else:
            return self.output_layer(x).squeeze(-1).squeeze(-1)
