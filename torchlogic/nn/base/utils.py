from collections import defaultdict
from copy import deepcopy

import torch
from torch import nn
import numpy as np
import numpy.typing as npt

from ._core import LukasiewiczCore
from .predicates import BasePredicates

from .constants import *


class BaseConcatenateBlocksLogic(LukasiewiczCore):

    def __init__(self, modules, outputs_key):
        super().__init__()
        self.modules_to_concat = nn.ModuleDict({f'concat_module_{i}': m for i, m in enumerate(modules)})
        self.operands = modules
        self.outputs_key = outputs_key
        self.out_features = sum([x.out_features for x in modules])

        # create operand map
        i = 0
        self.out_feature_operand_map = defaultdict(int)
        for j, op in enumerate(self.operands):
            for k in range(op.out_features):
                self.out_feature_operand_map[i] = j
                i += 1

    def _produce_logic_string(self, operand, explanation: list, print_type='logical', depth:int = 0) -> str:
        """
        Produce a string with logical operation.

        Args:
            explanation (list): list of explanation texts.
            print_type (str): one of 'logical' or 'natural'.  If logical, prints logic, otherwise, prints natural text.

        Returns:
            str: explanation text.
        """
        assert print_type in ['logical', 'logical-natural', 'natural'], \
            "'print_type' must be one of 'logical', 'logical-natural', or 'natural'"

        # joining operations for the next level down
        if print_type == 'natural':
            if operand.logic_type == 'And':
                if depth < 5:
                    joined_explanation = and_joining_options[depth].join(explanation)
                else:
                    joined_explanation = and_joining_options[-1].join(explanation)
            elif operand.logic_type == 'Or':
                if depth < 5:
                    joined_explanation = or_joining_options[depth].join(explanation)
                else:
                    joined_explanation = or_joining_options[-1].join(explanation)
        else:
            joined_explanation = ', '.join(explanation)

        if len(explanation) == 1:
            return joined_explanation
        if operand.logic_type == 'And':
            if print_type == 'logical':
                return f'AND({joined_explanation})'
            elif print_type == 'logical-natural':
                out_str = 'ALL of the following are TRUE: \n- ' + ', \n- '.join(explanation)
                return out_str.replace("\n", "\n\t")
            elif print_type == 'natural':
                if depth < 5:
                    leading_text = and_options[depth]
                else:
                    leading_text = f"{and_options[-1]}\n\n"
                return f'{leading_text}{joined_explanation}'
        if operand.logic_type == 'Or':
            if print_type == 'logical':
                return f'OR({joined_explanation})'
            elif print_type == 'logical-natural':
                out_str = 'ANY of the following are TRUE: \n- ' + ', \n- '.join(explanation)
                return out_str.replace("\n", "\n\t")
            elif print_type == 'natural':
                if depth < 5:
                    leading_text = or_options[depth]
                else:
                    leading_text = f"{or_options[-1]}\n\n"
                return f'{leading_text}{joined_explanation}'
        if operand.logic_type == 'XOr':
            if print_type == 'logical':
                return f'XOR({joined_explanation})'
            elif print_type == 'natural':
                out_str = 'ONE of the following is TRUE: \n' + ', \n- '.join(explanation)
                return out_str.replace("\n", "\n\t")

    def _produce_negation_string_natural(self, operand, text, depth):
        if depth < 5:
            condition1 = (text.lstrip('\n').lstrip("\t").lstrip().find(and_options[depth]) == 0
                          or text.lstrip('\n').lstrip("\t").lstrip().find(or_options[depth]) == 0)
            condition2 = (text.lstrip('\n').lstrip("\t").lstrip().find(and_options_negated[depth]) == 0
                          or text.lstrip('\n').lstrip("\t").lstrip().find(or_options_negated[depth]) == 0)
            if condition1:
                if operand.logic_type == 'And':
                    out_str = text.replace(and_options[depth], and_options_negated[depth], 1)
                elif operand.logic_type == 'Or':
                    out_str = text.replace(or_options[depth], or_options_negated[depth], 1)
            elif condition2:
                if operand.logic_type == 'And':
                    out_str = text.replace(and_options_negated[depth], and_options[depth], 1)
                elif operand.logic_type == 'Or':
                    out_str = text.replace(or_options_negated[depth], or_options[depth], 1)
            else:
                # handle case where text is at the predicate level.  In this case of the predicate negations are
                # flipped
                out_str = self._flip_negated_predicates(text.strip())
        else:
            condition1 = (text.lstrip('\n').lstrip("\t").lstrip().find(and_options[-1]) == 0
                          or text.lstrip('\n').lstrip("\t").lstrip().find(or_options[-1]) == 0)
            condition2 = (text.lstrip('\n').lstrip("\t").lstrip().find(and_options_negated[-1]) == 0
                          or text.lstrip('\n').lstrip("\t").lstrip().find(or_options_negated[-1]) == 0)
            if condition1:
                if operand.logic_type == 'And':
                    out_str = text.replace(and_options[-1], and_options_negated[-1], 1)
                elif operand.logic_type == 'Or':
                    out_str = text.replace(or_options[-1], or_options_negated[-1], 1)
            elif condition2:
                if operand.logic_type == 'And':
                    out_str = text.replace(and_options_negated[-1], and_options[-1], 1)
                elif operand.logic_type == 'Or':
                    out_str = text.replace(or_options_negated[-1], or_options[-1], 1)
            else:
                # handle case where text is at the predicate level.  In this case of the predicate negations are
                # flipped
                out_str = self._flip_negated_predicates(text.strip())
        out_str = self._flip_negated_logics(out_str, depth)
        return out_str

    def _produce_negation_string(
            self,
            operand,
            weight: float,
            text: str,
            negate: bool = False,
            explain_type: str = 'both',
            print_type: str = 'logical',
            depth: int = 0,
    ) -> str:
        """
        Produce a negation under certain conditions.

        Args:
            weight (float): Weight for specific predicate.
            text(str): Text for specific predicate.
            negate (bool): If True, flip operation for negation.
            explain_type (str): One of `both`, `positive`, `negative`. Control types of logic to produce.
            print_type (str): one of 'logical' or 'natural'.  If logical, prints logic, otherwise, prints natural text.

        Returns:
            str: The text with negation applied.
        """
        assert print_type in ['logical', 'logical-natural', 'natural'], \
            "'print_type' must be one of 'logical' or 'natural'"

        # conditions that produce a negative
        if (((weight < 0 and not negate) or (weight >= 0 and negate)) and text != "") \
                and (explain_type == 'both' or explain_type == 'negative'):
            if print_type == 'logical':
                return self._produce_negation_string_logical(text)
            elif print_type == 'logical-natural':
                return self._produce_negation_string_logical_natural(text)
            elif print_type == 'natural':
                return self._produce_negation_string_natural(operand, text, depth)
            else:
                raise ValueError("`print_type` must be one of `logical` or `natural`")
        elif (((weight < 0 and not negate) or (weight >= 0 and negate)) and text != "") \
                and explain_type == 'positive':
            return ""
        # conditions that produce a positive
        if (((weight < 0 and negate) or (weight >= 0 and not negate)) and text != "") \
                and (explain_type == 'both' or explain_type == 'positive'):
            return self._natural_language_negated_predicates(text)
        elif (((weight < 0 and negate) or (weight >= 0 and not negate)) and text != "") \
                and explain_type == 'negative':
            return ""
        else:
            return self._natural_language_negated_predicates(text)

    def explain_sample(
            self,
            required_output_thresholds: float,
            outputs_dict: dict = None,
            quantile: float = 0.5,
            threshold: float = None,
            parent_weights: npt.NDArray = None,
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

        # select ranges that need to be processed
        out_feature_range = torch.arange(self.out_features)
        parent_weights_mask = self._produce_weights_mask(parent_weights, quantile, threshold)

        if parent_mask is not None:
            out_feature_range = out_feature_range[parent_mask]
        if isinstance(out_feature_range, np.int64):
            out_feature_range = torch.tensor([out_feature_range])

        # process ranges
        explanation = []
        for i, (out_feature, parent_weight_mask) in enumerate(zip(out_feature_range, parent_weights_mask)):
            if parent_weight_mask:  # if the logic is above the quantile threshold then produce the explanation

                operand = self.operands[self.out_feature_operand_map[int(out_feature)]]
                if hasattr(operand, 'tau') and hasattr(operand, 'var_emb_dim'):
                    # variational
                    mask = operand.sample_mask()
                    weights_to_use = operand.weights * mask
                elif hasattr(operand, 'tau'):
                    # attn
                    if not isinstance(operand.operands, BasePredicates):
                        current_inputs = outputs_dict[operand.operands.outputs_key]
                    else:
                        current_inputs = input_features.to(operand.weights.device)
                        current_inputs = current_inputs.unsqueeze(0)
                    weights_to_use = operand.produce_explanation_weights(current_inputs)
                else:
                    weights_to_use = operand.weights
                mask_to_use = operand.mask

                if global_explain:
                    oo = self._compute_required_inputs(
                        parent_weights=parent_weights,
                        required_output_threshold=required_output_thresholds,
                        parent_logic_type=parent_logic_type,
                        negate=negate,
                        rounding_precision=rounding_precision
                    )
                    co = oo[i].squeeze()
                    oo = oo.squeeze()
                else:
                    co = current_outputs[0, channel, 0, out_feature].cpu()
                    oo = current_outputs[0, channel, 0, parent_mask].cpu().squeeze()

                current_outputs_mask, next_required_output_thresholds = self._produce_current_outputs_mask(
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

                if current_outputs_mask:

                    negation = negate if parent_weights[i] >= 0. else not negate
                    op_explain = operand.operands.explain_sample(
                        outputs_dict=od,
                        required_output_thresholds=next_required_output_thresholds,
                        quantile=quantile,
                        threshold=threshold,
                        parent_weights=weights_to_use[channel, :, 0].detach().cpu().squeeze(),
                        parent_mask=mask_to_use[channel, 0, :].detach().cpu(),
                        parent_logic_type=operand.logic_type,
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
                        force_negate=force_negate,
                        inverse_transform=inverse_transform,
                        show_bounds=show_bounds,
                        **kwargs
                    )

                    if op_explain:
                        explanation_str = [
                            self._produce_negation_string(
                                operand,
                                parent_weights[i],
                                f'{self._produce_logic_string(operand, op_explain, print_type)}',
                                explain_type=explain_type,
                                print_type=print_type
                            )]
                        explanation += explanation_str

                        if print_explanation:
                            print(f"Logic at depth {depth}: {explanation_str}"
                                  f"\nweights: {weights_to_use[channel, :, 0].detach().cpu().squeeze()}"
                                  f"\noutput: {co if parent_weights[i] >= 0 else 1. - co}"
                                  f"\nrequired_threshold: {next_required_output_thresholds}\n")

        explanation = np.unique([x for x in explanation if
                                 x != "" and x.replace("\t", "").replace("\n", "") != 'NOT()']).tolist()

        if explanation:
            return explanation
        return [""]

    def explain(
            self,
            required_output_thresholds: float,
            quantile: float = 0.5,
            threshold: float = None,
            parent_weights: npt.NDArray = None,
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
            inverse_transform (sklearn transform): inverse of in the inputs transform

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
            required_output_thresholds: float,
            quantile: float = 0.5,
            threshold: float = None,
            parent_weights: npt.NDArray = None,
            parent_mask: torch.Tensor = None,
            negate: bool = False,
            depth: int = 0,
            explain_type: str = 'both',
            print_type: str = 'logical',
            parent_logic_type: str = None,
            input_features: torch.tensor = None,
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
            inverse_transform (sklearn transform): inverse of in the inputs transform
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
            required_output_thresholds: float,
            quantile: float = 0.5,
            threshold: float = None,
            parent_weights: npt.NDArray = None,
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
            inverse_transform (sklearn transform): inverse of in the inputs transform
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
