import logging
from copy import deepcopy
from typing import Union

import numpy as np

import torch
from torch.nn import Parameter, functional

from .predicates import BasePredicates
from ._core import LukasiewiczCore
from torchlogic.utils.operations import val_clamp, EXU


class LukasiewiczChannelBlock(LukasiewiczCore):

    EPS = 1e-3

    def __init__(
            self,
            channels: int,
            in_features: int,
            out_features: int,
            n_selected_features: int,
            parent_weights_dimension: str,
            operands,
            logic_type: str,
            outputs_key: str,
            add_negations: bool = False,
            weight_init: float = 0.2
    ):
        """
        A logical and channel block.

        Args:
            channels (int): The number of versions of logic.
            in_features (int): The number of features in total from the previous layer or input.
            out_features (int): The number of output features, corresponding to the number of AND nodes for output.
            n_selected_features (int): The number of features to use for input to logic from in_features.
            parent_weights_dimension (str): One of 'out_features', 'channels'.
            operands (Union[LukasiewiczChannelBlock, BasePredicates]): The child logic.
            logic_type (str): type of logic for self.
            outputs_key (str): name of outputs used during explanation generation.
            add_negations (bool): add negated logic matching original logics.
            weight_init (float): Upper bound of uniform weight initialization.  Lower bound is negated value.
        """
        super(LukasiewiczChannelBlock, self).__init__()
        size = (channels, n_selected_features, out_features)
        weights = torch.ones(size).uniform_(-1 * weight_init, weight_init)
        if add_negations:
            weights = torch.cat([weights, weights.clone() * -1], dim=1)

        self.weights = Parameter(weights)
        self.bias = Parameter(torch.tensor(1.0), requires_grad=False)

        self.channels = channels
        self.in_features = in_features
        self.out_features = out_features
        self.operands = operands
        self.outputs_key = outputs_key
        self.parent_weights_dimension = parent_weights_dimension
        self.n_selected_features = n_selected_features
        self.logic_type = logic_type
        self.knowledge_added = False
        self.add_negations = add_negations
        self.weight_init = weight_init

        # MASK IS SIZE: [CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES] and values range from 0 to IN_FEATURES - 1
        mask = torch.empty(
            (channels, out_features, in_features)).uniform_(0, 1).topk(n_selected_features, dim=-1).indices
        if add_negations:
            mask = torch.cat([mask, mask.clone()], dim=-1)
        self.register_buffer('mask', mask)
        self.logger = logging.getLogger(self.__class__.__name__)

    def explain_sample(
            self,
            required_output_thresholds: torch.Tensor,
            outputs_dict: dict = None,
            quantile: float = 0.5,
            threshold: float = None,
            parent_weights: torch.Tensor = None,
            parent_mask: torch.Tensor = None,
            negate: bool = False,
            depth: int = 0,
            explain_type: str = 'both',
            print_type: str = 'logical',
            parent_logic_type: str = None,
            input_features: torch.Tensor = None,
            force_negate: bool = False,
            channel: int = 0,
            global_explain: bool = False,
            print_explanation: bool = False,
            ignore_uninformative: bool = False,
            rounding_precision: int = 3,
            inverse_transform=None,
            show_bounds: bool = True,
            feature_importances: bool = False,
            feature_importances_type: str = 'weight',
            **kwargs
    ) -> list:
        """
        Produce a sample explanation.

        Args:
            required_output_thresholds (float): threshold below which inner logic will be masked and excluded.
            outputs_dict (dict): Dictionary of outputs from forward pass.
            quantile (float): Quantile of logic to produce.
            threshold (float): Weight threshold to use.  If not None, then overrides quantile
            parent_weights (np.array): Array of parent weights.
            parent_mask (np.array): Array of parent mask.
            negate (bool): If True, flip operation for negation.
            depth (int): Depth of the current logic in the network.
            explain_type (str): One of `both`, `positive`, `negative`. Control types of logic to produce.
            print_type (str): one of 'logical' or 'natural'.  If logical, prints logic, otherwise, prints natural text.
            parent_logic_type (str): one of 'Or', 'And'.
            input_features (torch.Tensor): tensor of input features.
            force_negate (bool): If True, extract the negation of logic.
            channel (int): channel to perform traversal over.
            global_explain (bool): If True, perform a global explanation.
            print_explanation (bool): If True, print the explanation.
            ignore_uninformative (bool): If True, ignore uninformative logic that has threshold at 0 or 1.
            rounding_precision (int): precision for rounding percentiles during explanation.
            inverse_transform (sklearn transform): inverse of in the inputs transform
            show_bounds (bool): include numeric boundary used in logic

        Returns:
            list: explanation texts.
        """
        if isinstance(parent_weights, float) or parent_weights.ndim == 0:
            parent_weights = torch.tensor([parent_weights])

        if not global_explain:
            current_outputs = outputs_dict[self.outputs_key]
            od = deepcopy(outputs_dict)
        else:
            od = None

        if hasattr(self, 'var_emb_dim') and hasattr(self, 'var_n_layers'):
            # variational
            mask = self.sample_mask()
            weights_to_use, mask_to_use = self.weights * mask, self.mask
        elif hasattr(self, 'attn_emb_dim') and hasattr(self, 'attn_n_layers'):
            if not isinstance(self.operands, BasePredicates):
                current_inputs = outputs_dict[self.operands.outputs_key]
            else:
                current_inputs = input_features.to(self.weights.device)
                if current_inputs.ndim == 1:
                    current_inputs = current_inputs.unsqueeze(0)
            weights_to_use, mask_to_use = self.produce_explanation_weights(current_inputs), self.mask
        else:
            weights_to_use, mask_to_use = self.weights, self.mask

        # select ranges that need to be processed
        out_feature_range = np.arange(self.out_features)
        parent_weights_mask = self._produce_weights_mask(parent_weights, quantile, threshold)

        if parent_mask is not None:
            out_feature_range = out_feature_range[parent_mask]
        if isinstance(out_feature_range, np.int64):
            out_feature_range = torch.tensor([out_feature_range])

        # process ranges
        explanation = []
        for i, (out_feature, parent_weight_mask) in enumerate(zip(out_feature_range, parent_weights_mask)):
            if parent_weight_mask:  # if the logic is above the quantile threshold then produce the explanation
                if global_explain:
                    oo = self._compute_required_inputs(
                        parent_weights=parent_weights,
                        required_output_threshold=min(
                            required_output_thresholds * (1 - self.EPS if force_negate else 1 + self.EPS), 1.0),
                        # required_output_threshold=required_output_thresholds,
                        parent_logic_type=parent_logic_type,
                        negate=negate,
                        rounding_precision=rounding_precision
                    )
                    co = oo[i].squeeze()
                    oo = oo.squeeze()
                else:
                    co = current_outputs[0, channel, 0, out_feature].cpu()
                    oo = current_outputs[0, channel, 0, parent_mask].cpu().squeeze()

                # current_outputs_mask, next_required_output_thresholds = self._produce_current_outputs_mask(
                current_outputs_mask, next_required_output_thresholds, current_outputs_negated, other_outputs = self._produce_current_outputs_mask(
                    co,
                    oo,
                    parent_weights,
                    parent_weights[i],
                    parent_logic_type,
                    required_output_thresholds,
                    negate,
                    force_negate,
                    rounding_precision
                )

                if feature_importances and feature_importances_type in ['weight', 'weight_proportion']:
                    current_outputs_mask = True

                # if current_outputs_mask or print_explanation:
                if current_outputs_mask:

                    negation = negate if parent_weights[i] >= 0. else not negate
                    op_explain = self.operands.explain_sample(
                        outputs_dict=od,
                        required_output_thresholds=next_required_output_thresholds,
                        quantile=quantile,
                        threshold=threshold,
                        parent_weights=weights_to_use[channel, :, out_feature].detach().cpu().squeeze(),
                        parent_mask=mask_to_use[channel, out_feature, :].detach().cpu(),
                        parent_logic_type=self.logic_type,
                        negate=negation,
                        depth=depth + 1,
                        explain_type=explain_type,
                        print_type=print_type,
                        input_features=input_features,
                        channel=channel,
                        global_explain=global_explain,
                        print_explanation=print_explanation,
                        ignore_uninformative=ignore_uninformative,
                        rounding_precision=rounding_precision,
                        inverse_transform=inverse_transform,
                        force_negate=force_negate,
                        show_bounds=show_bounds,
                        feature_importances=feature_importances,
                        feature_importances_type=feature_importances_type,
                        **kwargs
                    )

                    if op_explain:
                        # TODO find a better name for tmp
                        tmp = self._produce_negation_string(
                                parent_weights[i],
                                f'{self._produce_logic_string(op_explain, print_type, depth)}',
                                explain_type=explain_type,
                                print_type=print_type,
                                depth=depth,
                            )

                        if feature_importances:
                            importance_value = self._generate_feature_importance(
                                    current_outputs_negated=current_outputs_negated,
                                    parent_weight=parent_weights[i],
                                    feature_importances_type=feature_importances_type,
                                    parent_logic_type=parent_logic_type,
                                    required_output_thresholds=required_output_thresholds,
                                    other_outputs=other_outputs
                            )

                            explanation_str = \
                                self._produce_negation_string(
                                    parent_weights[i],
                                    f'{self._produce_logic_string(op_explain, print_type, depth)}',
                                    explain_type=explain_type,
                                    print_type=print_type,
                                    depth=depth,
                                )
                            # TODO could there be multiple consecutive NOT we may need to remove?
                            if explanation_str.startswith('NOT(NOT(') and (explanation_str.endswith('))')):
                                explanation_str = explanation_str[8:-2]
                                # print("removed NOT NOT:", explanation_str)
                            if (explanation_str.startswith('NOT(AND')) and (explanation_str.endswith('))')):
                                # print("NOT AND")
                                if feature_importances_type in ['weight_proportion']:
                                    tmp = 'NOT(AND[' + str(importance_value) + ']' + explanation_str[7:]
                                    # tmp = 'NOT(AND[1.0]' + explanation_str[7:]
                                else:
                                    tmp = 'NOT(AND[' + str(importance_value) + ']' + explanation_str[7:]
                            elif (explanation_str.startswith('NOT(OR')) and (explanation_str.endswith('))')):
                                # print("NOT OR")
                                if feature_importances_type in ['weight_proportion']:
                                    tmp = 'NOT(OR[' + str(importance_value) + ']' + explanation_str[6:]
                                    # tmp = 'NOT(AND[1.0]' + explanation_str[7:]
                                else:
                                    tmp = 'NOT(OR[' + str(importance_value) + ']' + explanation_str[6:]
                            elif (explanation_str.startswith('AND')):
                                tmp = 'AND[' + str(importance_value) + ']' + explanation_str[3:]
                            elif (explanation_str.startswith('OR')):
                                tmp = 'OR[' + str(importance_value) + ']' + explanation_str[2:]
                            explanation += [tmp]
                        else:
                            explanation_str = [
                                self._produce_negation_string(
                                    parent_weights[i],
                                    f'{self._produce_logic_string(op_explain, print_type, depth)}',
                                    explain_type=explain_type,
                                    print_type=print_type,
                                    depth=depth
                                )]
                            explanation += explanation_str

                        if print_explanation:
                            print(f"Logic at depth {depth}: {explanation_str}"
                                  f"\nweights: {weights_to_use[channel, :, out_feature].detach().cpu().squeeze()}"
                                  f"\noutput: {co if parent_weights[i] >= 0 else 1. - co}"
                                  f"\nrequired_threshold: {next_required_output_thresholds}\n")

        explanation = np.unique([x for x in explanation if
                                 x != "" and x.replace("\t", "").replace("\n", "") != 'NOT()']).tolist()
        if feature_importances:
            explanation = [x for x in explanation if
                                 x != "" and x.replace("\t", "").replace("\n", "") != 'NOT()']

        if explanation:
            return explanation
        return [""]

    def explain(
            self,
            required_output_thresholds: torch.Tensor,
            quantile: float = 0.5,
            threshold: float = None,
            parent_weights: torch.Tensor = None,
            parent_mask: torch.Tensor = None,
            negate: bool = False,
            depth: int = 0,
            explain_type: str = 'both',
            print_type: str = 'logical',
            parent_logic_type: str = None,
            force_negate: bool = False,
            channel: int = 0,
            ignore_uninformative: bool = False,
            rounding_precision: int = 3,
            inverse_transform=None,
            show_bounds: bool = True,
            **kwargs
    ) -> list:
        """
        Produce a global explanation.

        Args:
            required_output_thresholds (float): threshold below which inner logic will be masked and excluded.
            quantile (float): Quantile of logic to produce.
            threshold (float): Weight threshold to use.  If not None, then overrides quantile
            parent_weights (np.array): Array of parent weights.
            parent_mask (np.array): Array of parent mask.
            negate (bool): If True, flip operation for negation.
            depth (int): Depth of the current logic in the network.
            explain_type (str): One of `both`, `positive`, `negative`. Control types of logic to produce.
            print_type (str): one of 'logical' or 'natural'.  If logical, prints logic, otherwise, prints natural text.
            parent_logic_type (str): one of 'Or', 'And'.
            force_negate (bool): If True, extract the negation of logic.
            channel (int): channel to perform traversal over.
            ignore_uninformative (bool): If True, ignore uninformative logic that has threshold at 0 or 1.
            rounding_precision (int): precision for rounding percentiles during explanation.
            show_bounds (bool): include numeric boundary used in logic

        Returns:
            list: explanation texts.
        """
        return self.explain_sample(
            required_output_thresholds=required_output_thresholds,
            quantile=quantile,
            threshold=threshold,
            parent_weights=parent_weights,
            parent_mask=parent_mask,
            negate=negate,
            depth=depth,
            explain_type=explain_type,
            print_type=print_type,
            parent_logic_type=parent_logic_type,
            force_negate=force_negate,
            channel=channel,
            global_explain=True,
            ignore_uninformative=ignore_uninformative,
            rounding_precision=rounding_precision,
            inverse_transform=inverse_transform,
            show_bounds=show_bounds,
            **kwargs
        )

    def print_sample(
            self,
            outputs_dict: dict,
            required_output_thresholds: torch.Tensor,
            quantile: float = 0.5,
            threshold: float = None,
            parent_weights: torch.Tensor = None,
            parent_mask: torch.Tensor = None,
            negate: bool = False,
            depth: int = 0,
            explain_type: str = 'both',
            print_type: str = 'logical',
            parent_logic_type: str = None,
            input_features: torch.Tensor = None,
            force_negate: bool = False,
            channel: int = 0,
            ignore_uninformative: bool = False,
            rounding_precision: int = 3,
            inverse_transform=None,
            show_bounds: bool = True,
            **kwargs
    ) -> list:
        """
        Print a sample explanation view of the model.

        Args:
            required_output_thresholds (float): threshold below which inner logic will be masked and excluded.
            outputs_dict (dict): Dictionary of outputs from forward pass.
            quantile (float): Quantile of logic to produce.
            threshold (float): Weight threshold to use.  If not None, then overrides quantile
            parent_weights (np.array): Array of parent weights.
            parent_mask (np.array): Array of parent mask.
            negate (bool): If True, flip operation for negation.
            depth (int): Depth of the current logic in the network.
            explain_type (str): One of `both`, `positive`, `negative`. Control types of logic to produce.
            print_type (str): one of 'logical' or 'natural'.  If logical, prints logic, otherwise, prints natural text.
            parent_logic_type (str): one of 'Or', 'And'.
            input_features (torch.Tensor): tensor of input features.
            force_negate (bool): If True, extract the negation of logic.
            channel (int): channel to perform traversal over.
            ignore_uninformative (bool): If True, ignore uninformative logic that has threshold at 0 or 1.
            rounding_precision (int): precision for rounding percentiles during explanation.
            show_bounds (bool): include numeric boundary used in logic
        """
        return self.explain_sample(
            outputs_dict=outputs_dict,
            required_output_thresholds=required_output_thresholds,
            quantile=quantile,
            threshold=threshold,
            parent_weights=parent_weights,
            parent_mask=parent_mask,
            negate=negate,
            depth=depth,
            explain_type=explain_type,
            print_type=print_type,
            parent_logic_type=parent_logic_type,
            input_features=input_features,
            force_negate=force_negate,
            channel=channel,
            print_explanation=True,
            ignore_uninformative=ignore_uninformative,
            rounding_precision=rounding_precision,
            inverse_transform=inverse_transform,
            show_bounds=show_bounds,
            **kwargs
        )

    def print(
            self,
            required_output_thresholds: torch.Tensor,
            quantile: float = 0.5,
            threshold: float = None,
            parent_weights: torch.Tensor = None,
            parent_mask: torch.Tensor = None,
            negate: bool = False,
            depth: int = 0,
            explain_type: str = 'both',
            print_type: str = 'logical',
            parent_logic_type: str = None,
            force_negate: bool = False,
            channel: int = 0,
            ignore_uninformative: bool = False,
            rounding_precision: int = 3,
            inverse_transform=None,
            show_bounds: bool = True,
            **kwargs
    ) -> list:
        """
        Produce a global explanation view of the model.

        Args:
            required_output_thresholds (float): threshold below which inner logic will be masked and excluded.
            quantile (float): Quantile of logic to produce.
            threshold (float): Weight threshold to use.  If not None, then overrides quantile
            parent_weights (np.array): Array of parent weights.
            parent_mask (np.array): Array of parent mask.
            negate (bool): If True, flip operation for negation.
            depth (int): Depth of the current logic in the network.
            explain_type (str): One of `both`, `positive`, `negative`. Control types of logic to produce.
            print_type (str): one of 'logical' or 'natural'.  If logical, prints logic, otherwise, prints natural text.
            parent_logic_type (str): one of 'Or', 'And'.
            force_negate (bool): If True, extract the negation of logic.
            channel (int): channel to perform traversal over.
            ignore_uninformative (bool): If True, ignore uninformative logic that has threshold at 0 or 1.
            rounding_precision (int): precision for rounding percentiles during explanation.
            show_bounds (bool): include numeric boundary used in logic
        """
        return self.explain_sample(
            required_output_thresholds=required_output_thresholds,
            quantile=quantile,
            threshold=threshold,
            parent_weights=parent_weights,
            parent_mask=parent_mask,
            negate=negate,
            depth=depth,
            explain_type=explain_type,
            print_type=print_type,
            parent_logic_type=parent_logic_type,
            force_negate=force_negate,
            channel=channel,
            global_explain=True,
            print_explanation=True,
            ignore_uninformative=ignore_uninformative,
            rounding_precision=rounding_precision,
            inverse_transform=inverse_transform,
            show_bounds=show_bounds,
            **kwargs
        )

    def add_knowledge(
            self,
            channel,
            out_feature,
            input_indices,
            required_thresholds,
            required_negations,
            freeze_knowledge=True
    ):
        # if knowledge has not been added yet then zero the weights
        if not self.knowledge_added:
            self.weights.data.copy_(torch.zeros_like(self.weights.data))

        # update the mask to use the given predicates
        # [CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES]
        self.mask.data[channel, out_feature, :] = torch.tensor(input_indices)

        # update the weights to use the given required thresholds
        # [CHANNELS, N_SELECTED_FEATURES, OUT_FEATURES]

        # flip negated thresholds
        required_thresholds = torch.tensor(required_thresholds)
        required_negations = torch.tensor(required_negations)
        required_thresholds = (torch.relu(required_negations) * required_thresholds
                               + (1 - torch.relu(required_negations)) * (1 - required_thresholds))

        # determine weights
        non_zero_weight_count = (required_thresholds >= 0).sum()
        if self.logic_type == 'Or':
            required_weights = (1.0/non_zero_weight_count) / required_thresholds
        elif self.logic_type == 'And':
            required_weights = (1.0/non_zero_weight_count) / (1.0 - required_thresholds)
        else:
            raise AssertionError("logic_type must be 'And', 'Or'")
        required_weights = required_weights * required_negations
        required_weights = required_weights * (required_thresholds > 0.0)
        required_weights[torch.logical_and(required_weights >= -1e-8, required_weights <= 1e-8)] = 0.0

        self.weights.data[channel, :, out_feature] = required_weights
        self.weights.requires_grad_(not freeze_knowledge)

        self.knowledge_added = True


class VariationalLukasiewiczChannelBlock(LukasiewiczChannelBlock):

    def __init__(
            self,
            channels: int,
            in_features: int,
            out_features: int,
            n_selected_features: int,
            parent_weights_dimension: str,
            operands,
            logic_type: str,
            outputs_key: str,
            add_negations: bool = False,
            weight_init: float = 0.2,
            var_emb_dim: int = 50,
            var_n_layers: int = 2
    ):
        """
        A logical and channel block.

        Args:
            channels (int): The number of versions of logic.
            in_features (int): The number of features in total from the previous layer or input.
            out_features (int): The number of output features, corresponding to the number of AND nodes for output.
            n_selected_features (int): The number of features to use for input to logic from in_features.
            parent_weights_dimension (str): One of 'out_features', 'channels'.
            operands (Union[LukasiewiczChannelBlock, BasePredicates]): The child logic.
            logic_type (str): type of logic for self.
            outputs_key (str): name of outputs used during explanation generation.
            add_negations (bool): add negated logic matching original logics.
            weight_init (float): Upper bound of uniform weight initialization.  Lower bound is negated value.
        """
        super(VariationalLukasiewiczChannelBlock, self).__init__(
            channels=channels,
            in_features=in_features,
            out_features=out_features,
            n_selected_features=n_selected_features,
            parent_weights_dimension=parent_weights_dimension,
            operands=operands,
            logic_type=logic_type,
            outputs_key=outputs_key,
            add_negations=add_negations,
            weight_init=weight_init
        )
        self.var_emb_dim = var_emb_dim
        self.var_n_layers = var_n_layers
        assert self.var_n_layers >= 2, "`attn_n_layers` must be at least 1."

        self.var_emb = torch.randn(size=(channels * out_features, var_emb_dim))

        # construct variational mean network
        var_mean = [torch.nn.Linear(self.var_emb_dim, self.var_emb_dim), torch.nn.ReLU()]
        for _ in range(self.var_n_layers - 2):
            var_mean += [torch.nn.Linear(self.var_emb_dim, self.var_emb_dim), torch.nn.ReLU()]
        var_mean += [torch.nn.Linear(self.var_emb_dim, self.weights.size(1))]
        self.var_mean = torch.nn.Sequential(*var_mean)

        # construct variational log std network
        var_std = [torch.nn.Linear(self.var_emb_dim, self.var_emb_dim), torch.nn.ReLU()]
        for _ in range(self.var_n_layers - 2):
            var_std += [torch.nn.Linear(self.var_emb_dim, self.var_emb_dim), torch.nn.ReLU()]
        var_std += [torch.nn.Linear(self.var_emb_dim, self.weights.size(1))]
        self.var_std = torch.nn.Sequential(*var_std)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc
        self.N.scale = self.N.scale

        self.tau = Parameter(torch.ones(size=(self.weights.size(0), 1, self.weights.size(2)))).float()

        self.seed_count = 0

    def sample_mask(self):
        # variational sparsity
        if not self.training:
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)  # if you are using multi-GPU.
        else:
            self.seed_count += 1
            torch.manual_seed(self.seed_count)
            torch.cuda.manual_seed(self.seed_count)
            torch.cuda.manual_seed_all(self.seed_count)  # if you are using multi-GPU.

        # If the distributions diverge then reset to draw from a normal dist with mean 1. std 1.
        mu = torch.nan_to_num(self.var_mean(self.var_emb.to(self.var_mean[0].weight.device)), 0., 0., 0.)
        sigma = torch.nan_to_num(
            torch.exp(self.var_std(self.var_emb.to(self.var_mean[0].weight.device))), 1., 1., 1.)
        z = mu + sigma * self.N.sample(mu.shape).to(self.var_mean[0].weight.device)
        z = z.reshape(self.weights.size(0), self.weights.size(2), self.weights.size(1)).transpose(2, 1)
        mask = functional.gumbel_softmax(
            torch.logit(z, eps=1e-6),  # how else can we get tau not to diverge?
            dim=1,
            tau=val_clamp(self.tau, _min=0.1, _max=1.0).float()
        )
        mask = torch.nan_to_num(mask, 0., 0., 0.)  # how do we avoid divergence?
        return mask


class AttentionLukasiewiczChannelBlock(LukasiewiczChannelBlock):

    def __init__(
            self,
            channels: int,
            in_features: int,
            out_features: int,
            n_selected_features: int,
            parent_weights_dimension: str,
            operands,
            logic_type: str,
            outputs_key: str,
            add_negations: bool = False,
            weight_init: float = None,
            attn_emb_dim: int = 32,
            attn_n_layers: int = 2
    ):
        """
        A logical and channel block.

        Args:
            channels (int): The number of versions of logic.
            in_features (int): The number of features in total from the previous layer or input.
            out_features (int): The number of output features, corresponding to the number of AND nodes for output.
            n_selected_features (int): The number of features to use for input to logic from in_features.
            parent_weights_dimension (str): One of 'out_features', 'channels'.
            operands (Union[LukasiewiczChannelBlock, BasePredicates]): The child logic.
            logic_type (str): type of logic for self.
            outputs_key (str): name of outputs used during explanation generation.
            add_negations (bool): add negated logic matching original logics.
            weight_init (float): Upper bound of uniform weight initialization.  Lower bound is negated value.
        """
        super(AttentionLukasiewiczChannelBlock, self).__init__(
            channels=channels,
            in_features=in_features,
            out_features=out_features,
            n_selected_features=n_selected_features,
            parent_weights_dimension=parent_weights_dimension,
            operands=operands,
            logic_type=logic_type,
            outputs_key=outputs_key,
            add_negations=add_negations,
            weight_init=weight_init
        )

        self.attn_emb_dim = attn_emb_dim
        self.attn_n_layers = attn_n_layers
        assert self.attn_n_layers >= 2, "`attn_n_layers` must be at least 1."

        # construct attention network (original)
        attn = [torch.nn.Linear(self.weights.size(1) * 2, self.attn_emb_dim), torch.nn.ReLU()]
        for _ in range(self.attn_n_layers - 2):
            attn += [torch.nn.Linear(self.attn_emb_dim, self.attn_emb_dim), torch.nn.ReLU()]
        attn += [torch.nn.Linear(self.attn_emb_dim, self.weights.size(1))]
        self.attn = torch.nn.Sequential(*attn)
        # end construct attention network (original)

        # # construct attention network (nam like)
        # attn = []
        # for i in range(self.attn_n_layers - 2):
        #     if i == self.attn_n_layers - 3:
        #         attn += [torch.nn.Linear(self.attn_emb_dim, self.attn_emb_dim // 2), torch.nn.ReLU()]
        #     else:
        #         attn += [torch.nn.Linear(self.attn_emb_dim, self.attn_emb_dim), torch.nn.ReLU()]
        # if attn:
        #     attn += [torch.nn.Linear(self.attn_emb_dim // 2, 1, bias=False)]
        # else:
        #     attn = [torch.nn.Linear(self.attn_emb_dim, 1, bias=False)]
        #
        # for layer in attn:
        #     if isinstance(layer, torch.nn.Linear):
        #         torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
        #
        # self.attn = torch.nn.Sequential(EXU(1, self.attn_emb_dim), *attn)
        # # end construct attention network (nam like)

        self.tau = Parameter(torch.ones(size=(1, self.weights.size(0), self.weights.size(2), 1, 1))).float()

        self.seed_count = 0

    def sample_mask(self, x):

        # variational sparsity
        if not self.training:
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)  # if you are using multi-GPU.
        else:
            self.seed_count += 1
            torch.manual_seed(self.seed_count)
            torch.cuda.manual_seed(self.seed_count)
            torch.cuda.manual_seed_all(self.seed_count)  # if you are using multi-GPU.

        # apply attention layers (Original)
        h = self.attn(x)
        # end apply attention layers (Original)

        # # apply attention layers NAM like
        # # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
        # x = x.unsqueeze(-1).unsqueeze(-1)
        # # residual connections
        # for i, layer in enumerate(self.attn):
        #     if i == 0:
        #         h = val_clamp(layer(x))
        #     elif i < len(self.attn) - 1:
        #         h = layer(h)
        #     else:
        #         h = layer(h)
        # h = h.squeeze(-1).squeeze(-1)
        # # end apply attention layers NAM like

        attn = functional.gumbel_softmax(
            torch.logit(h, eps=1e-6),  # how else can we get tau not to diverge?
            dim=-1,
            tau=val_clamp(self.tau, _min=0.1, _max=1.0).float()
        )
        attn = torch.nan_to_num(attn, 0., 0., 0.)  # how do we avoid divergence?

        return attn

    def produce_explanation_weights(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim == 3:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x = x.unsqueeze(1).repeat(1, self.channels, self.out_features, 1).unsqueeze(-2)
        if x.ndim == 4:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x = x.unsqueeze(-2).repeat(1, 1, self.out_features, 1, 1)

        x = x.gather(
            -1, self.mask.unsqueeze(0).expand(x.size(0), *self.mask.shape).unsqueeze(-2))

        weights_sign_attn = self.weights.data.detach().clone().repeat(
            x.size(0), 1, 1, 1).transpose(-2, -1).unsqueeze(-2).greater_equal(torch.tensor(0.0)).float()
        # pos_attn = x * weights_sign_attn
        # neg_attn = (1 - x) * (1 - weights_sign_attn)
        # x_attn = pos_attn + neg_attn

        mask = self.sample_mask(torch.cat([x, weights_sign_attn], dim=-1))
        mask = mask.squeeze(-2).transpose(-2, -1)
        weights = self.weights.repeat(x.size(0), 1, 1, 1) * mask
        weights = weights.squeeze(0)
        return weights
