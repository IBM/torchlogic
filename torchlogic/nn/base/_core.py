import os
import glob

from typing import Union, Tuple

import numpy as np

import torch
import torch.nn as nn

from .constants import *


class LukasiewiczCore(nn.Module):

    def __init__(self):
        super(LukasiewiczCore, self).__init__()

    def __produce_joined_explanation(self, explanation, print_type, depth):
        # joining operations for the next level down
        if print_type == 'logical':
            joined_explanation = ', '.join(explanation)
        elif print_type == 'logical-natural':
            joined_explanation = ', \n- '.join(explanation)
        elif print_type == 'natural':
            if self.logic_type == 'And':
                if depth < 5:
                    joined_explanation = and_joining_options[depth].join(explanation)
                else:
                    joined_explanation = and_joining_options[-1].join(explanation)
            elif self.logic_type == 'Or' or self.logic_type == 'XOr':
                if depth < 5:
                    joined_explanation = or_joining_options[depth].join(explanation)
                else:
                    joined_explanation = or_joining_options[-1].join(explanation)
            else:
                raise ValueError("logic_type must be 'And' or 'Or', 'XOr'.")
        else:
            raise ValueError("'print_type' must be one of 'logical' or 'natural'")
        return joined_explanation

    def __produce_logic_string_and(self, joined_explanation, print_type, depth):
        if print_type == 'logical':
            return f'AND({joined_explanation})'
        elif print_type == 'logical-natural':
            out_str = 'ALL the following are TRUE: \n- ' + joined_explanation
            return out_str.replace("\n", "\n\t")
        elif print_type == 'natural':
            if depth < 5:
                leading_text = and_options[depth]
            else:
                leading_text = f"{and_options[-1]}\n\n"
            return f'{leading_text}{joined_explanation}'

    def __produce_logic_string_or(self, joined_explanation, print_type, depth):
        if print_type == 'logical':
            return f'OR({joined_explanation})'
        elif print_type == 'logical-natural':
            out_str = 'ANY of the following are TRUE: \n- ' + joined_explanation
            return out_str.replace("\n", "\n\t")
        elif print_type == 'natural':
            if depth < 5:
                leading_text = or_options[depth]
            else:
                leading_text = f"{or_options[-1]}\n\n"
            return f'{leading_text}{joined_explanation}'

    def __produce_logic_string_xor(self, joined_explanation, print_type, depth):
        if print_type == 'logical':
            return f'XOR({joined_explanation})'
        elif print_type == 'logical-natural':
            out_str = 'ONE of the following is TRUE: \n- ' + joined_explanation
            return out_str.replace("\n", "\n\t")

    def _produce_logic_string(
            self, explanation: list,
            print_type: str = 'logical',
            depth: int = 0
    ) -> str:
        """
        Produce a string with logical operation.

        Args:
            explanation (list): list of explanation texts.
            print_type (str): one of 'logical' or 'natural'.  If logical, prints logic, otherwise, prints natural text.

        Returns:
            str: explanation text.
        """
        assert print_type in ['logical', 'logical-natural', 'natural'], \
            "'print_type' must be one of 'logical' or 'natural'"

        joined_explanation = self.__produce_joined_explanation(explanation, print_type, depth)

        # add leading text
        if len(explanation) == 1:
            return joined_explanation
        if self.logic_type == 'And':
            return self.__produce_logic_string_and(joined_explanation, print_type, depth)
        if self.logic_type == 'Or':
            return self.__produce_logic_string_or(joined_explanation, print_type, depth)
        if self.logic_type == 'XOr':
            return self.__produce_logic_string_xor(joined_explanation, print_type, depth)

    @staticmethod
    def _natural_language_negated_predicates(text):
        out_str = text.split(",")
        out_str_conv = []
        for pred_str in out_str:
            if pred_str.find("NOT") > -1:
                if pred_str.find("and NOT ") > -1:
                    out_str_conv += [pred_str.replace("and NOT ", "and it was NOT true ")]
                elif pred_str.find("or NOT ") > -1:
                    out_str_conv += [pred_str.replace("or NOT ", "or it was NOT true ")]
                elif pred_str[:4] == 'NOT ' and pred_str[:17] != "NOT the following":
                    out_str_conv += [f"it was NOT true {pred_str.replace('NOT ', '', 1)}"]
                else:
                    out_str_conv += [pred_str]
            else:
                out_str_conv += [pred_str]
        out_str = ','.join(out_str_conv)
        return out_str

    @staticmethod
    def _flip_negated_logics(text, depth):
        if depth > 5:
            raise Warning("Natural language explanations with a network depth greater than 6 levels and negations"
                          "are not guaranteed to produce correct results")

        and_joining_str = and_joining_options[depth].replace(".", "").strip()
        or_joining_str = or_joining_options[depth].replace(".", "").strip()
        and_joining_str_negated = and_joining_options_negated[depth].replace(".", "").strip()
        or_joining_str_negated = or_joining_options_negated[depth].replace(".", "").strip()

        out_str = text.split(".")
        out_str_conv = []
        for sub_str in out_str:
            if sub_str.find(and_joining_str) > -1:
                out_str_conv += [sub_str.replace(and_joining_str, and_joining_str_negated)]
            elif sub_str.find(or_joining_str) > -1:
                out_str_conv += [sub_str.replace(or_joining_str, or_joining_str_negated)]
            else:
                out_str_conv += [sub_str]
        out_str = ".".join(out_str_conv)
        return out_str

    @ staticmethod
    def _flip_negated_predicates(text):
        # handle case where text is at the predicate level.  In this case of the predicate negations are
        # flipped
        out_str = text.split(",")
        out_str_conv = []
        for pred_str in out_str:
            if pred_str.find("NOT") == -1:
                if pred_str.find("- ") > -1:
                    out_str_conv += [pred_str.replace("- ", "- NOT ")]
                elif pred_str.find(" and ") == 0:
                    out_str_conv += [pred_str.replace("and ", "or it was NOT true ")]
                elif pred_str.find(" or ") == 0:
                    out_str_conv += [pred_str.replace("or ", "and it was NOT true ")]
                else:
                    out_str_conv += ["it was NOT true " + pred_str]
            else:
                if pred_str.find("- NOT") > -1:
                    out_str_conv += [pred_str.replace("- NOT ", "- ")]
                elif pred_str.find("and it was NOT true") > -1:
                    out_str_conv += [pred_str.replace("and it was NOT true ", "or ")]
                elif pred_str.find("and NOT") > -1:
                    out_str_conv += [pred_str.replace("and NOT ", "or ")]
                elif pred_str.find("or it was NOT true") > -1:
                    out_str_conv += [pred_str.replace("or it was NOT true ", "and ")]
                elif pred_str.find("or NOT") > -1:
                    out_str_conv += [pred_str.replace("or NOT ", "and ")]
                elif pred_str.find("it was NOT true") > -1:
                    out_str_conv += [pred_str.replace("it was NOT true ", "")]
                elif pred_str.find("NOT ") > -1:
                    out_str_conv += [pred_str.replace("NOT ", "")]
        out_str = ','.join(out_str_conv)
        return out_str

    @staticmethod
    def _produce_negation_string_logical(text):
        # if (not text.replace('\n', '').replace('\t- ', '')[:8] == 'NOT(AND('
        if (not text.replace('\n', '').replace('\t- ', '')[:8] in ['NOT(AND(', 'NOT(AND[']
                # and not text.replace('\n', '').replace('\t- ', '')[:7] == 'NOT(OR('):
                and not text.replace('\n', '').replace('\t- ', '')[:7] in ['NOT(OR(', 'NOT(OR[']):
            out_str = f"NOT({text})"
        else:
            out_str = text.replace("NOT(", '', 1)[:-1]
        return out_str

    def _produce_negation_string_logical_natural(self, text):
        if not text.replace('\n', '').replace('\t- ', '')[:17] == 'NOT the following':
            out_str = f"NOT the following: \n- {text}"  # original
        elif text.replace('\n', '').replace('\t- ', '').find(
                "NOT the following:") == 0:
            out_str = text.replace(
                "NOT the following:", "", 1).replace('\n\t- ', '').strip()
        else:
            out_str = self._flip_negated_predicates(text)
        return out_str.replace("\n", "\n\t")

    def _produce_negation_string_natural(self, text, depth):
        if depth < 5:
            condition1 = (text.lstrip('\n').lstrip("\t").lstrip().find(and_options[depth]) == 0
                          or text.lstrip('\n').lstrip("\t").lstrip().find(or_options[depth]) == 0)
            condition2 = (text.lstrip('\n').lstrip("\t").lstrip().find(and_options_negated[depth]) == 0
                          or text.lstrip('\n').lstrip("\t").lstrip().find(or_options_negated[depth]) == 0)
            if condition1:
                if self.logic_type == 'And':
                    out_str = text.replace(and_options[depth], and_options_negated[depth], 1)
                elif self.logic_type == 'Or':
                    out_str = text.replace(or_options[depth], or_options_negated[depth], 1)
            elif condition2:
                if self.logic_type == 'And':
                    out_str = text.replace(and_options_negated[depth], and_options[depth], 1)
                elif self.logic_type == 'Or':
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
                if self.logic_type == 'And':
                    out_str = text.replace(and_options[-1], and_options_negated[-1], 1)
                elif self.logic_type == 'Or':
                    out_str = text.replace(or_options[-1], or_options_negated[-1], 1)
            elif condition2:
                if self.logic_type == 'And':
                    out_str = text.replace(and_options_negated[-1], and_options[-1], 1)
                elif self.logic_type == 'Or':
                    out_str = text.replace(or_options_negated[-1], or_options[-1], 1)
            else:
                # handle case where text is at the predicate level.  In this case of the predicate negations are
                # flipped
                out_str = self._flip_negated_predicates(text.strip())
        out_str = self._flip_negated_logics(out_str, depth)
        return out_str

    def _produce_negation_string(
            self,
            weight: float,
            text: str,
            negate: bool = False,
            explain_type: str = 'both',
            print_type: str = 'logical',
            depth: int = 0
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
                return self._produce_negation_string_natural(text, depth)
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

    @staticmethod
    def _produce_weights_mask(weights: torch.Tensor, quantile: float = None, threshold: float = None) -> torch.Tensor:
        """
        Produce a mask based on weights.

        Args:
            weights (np.array): Array of weights.
            quantile (float): Quantile of logic to produce.
            threshold (float): Weight threshold to use.  If not None, then overrides quantile

        Returns:
            np.array: Boolean mask on weights
        """
        zero_weights = weights == 0
        if isinstance(zero_weights, torch.BoolType):
            zero_weights = torch.tensor(zero_weights)
        if torch.all(zero_weights):
            return ~zero_weights
        if threshold is None:
            threshold = torch.quantile(torch.abs(weights[~zero_weights]), 1 - quantile)
        return torch.tensor(torch.abs(weights) >= threshold) & torch.tensor(torch.abs(weights) > 0)

    def _produce_current_outputs_mask(
            self,
            current_outputs: torch.Tensor,
            input_features: float,
            all_parent_weights,
            parent_weights: torch.Tensor,
            parent_logic_type: str,
            required_output_threshold: torch.Tensor,
            negate: bool = False,
            force_negate: bool = False,
            rounding_precision: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produce a mask based on the current outputs.

        Args:
            current_outputs (Union[npt.NDArray, float, int]): the outputs from the current logic.
            parent_weights (Union[npt.NDArray, float, int]): the parent weights for the current logic.
            parent_logic_type (str): one of 'Or', 'And'.
            required_output_threshold (float): the required output threshold, below which the mask is False.
            negate (bool): If True, flip operation for negation.

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: mask, thresholds
        """
        # if the logic is negated then the result of the current logic can't produce an output
        # greater than 1 - required thresholds
        ROUND_PRECISION = 32

        if negate:
            required_output_threshold = 1 - required_output_threshold
        required_output_threshold = torch.round(required_output_threshold, decimals=ROUND_PRECISION)

        aws = all_parent_weights.greater_equal(torch.tensor(0.0)).float()
        input_features_negated = aws * input_features + (1.0 - aws) * (1.0 - input_features)

        ws = parent_weights.greater_equal(torch.tensor(0.0)).float()
        current_outputs_negated = ws * current_outputs + (1.0 - ws) * (1.0 - current_outputs)

        if parent_logic_type == 'Or':
            # compute other outputs
            other_outputs = (input_features_negated @ torch.abs(all_parent_weights)
                             - current_outputs_negated * torch.abs(parent_weights))
            other_outputs = torch.round(other_outputs, decimals=ROUND_PRECISION)

            # force to zero
            if torch.logical_and(other_outputs >= -1e-7, other_outputs <= 1e-7):
                other_outputs = torch.tensor(0.0)

            # compute thresholds
            thresholds = torch.clip(
                (required_output_threshold - other_outputs) / torch.abs(parent_weights), 0.0, 1.0)
            thresholds = torch.round(thresholds, decimals=ROUND_PRECISION)

        elif parent_logic_type == 'And':
            # compute other outputs
            other_outputs = ((1 - input_features_negated) @ torch.abs(all_parent_weights)
                             - (1 - current_outputs_negated) * torch.abs(parent_weights))
            other_outputs = torch.round(other_outputs, decimals=ROUND_PRECISION)

            # force to zero
            if torch.logical_and(other_outputs >= -1e-7, other_outputs <= 1e-7):
                other_outputs = torch.tensor(0.0)

            # compute thresholds
            thresholds = torch.clip(
                1.0 - (1.0 - required_output_threshold - other_outputs) / np.abs(parent_weights),
                0.0, 1.0)
            thresholds = torch.round(thresholds, decimals=ROUND_PRECISION)
        elif parent_logic_type == 'XOr':
            # compute other outputs
            other_outputs = ((1 - input_features_negated) @ np.abs(all_parent_weights)
                             - (1 - current_outputs_negated) * np.abs(parent_weights))
            other_outputs = torch.round(other_outputs, decimals=ROUND_PRECISION)

            # force to zero
            if torch.logical_and(other_outputs >= -1e-7, other_outputs <= 1e-7):
                other_outputs = torch.tensor(0.0)

            # compute thresholds
            thresholds = torch.clip(
                1.0 - (1.0 - required_output_threshold + other_outputs) / torch.abs(parent_weights),
                0.0, 1.0)
            thresholds = torch.round(thresholds, decimals=ROUND_PRECISION)
        else:
            raise ValueError("parent logic type must be 'Or', 'And', 'XOr'.")

        # When force negating we don't want to change the threshold, but we want to have values below the threshold
        # instead of above it
        negate = negate if not force_negate else not negate

        # TODO: There seems to be some level of numerical instability that causes these values to not work out
        #  exactly.  This seems to be fixed after converting code to use PyTorch

        # def trunc(values, decs=0):
        #     return np.trunc(values * 10 ** decs) / (10 ** decs)
        #
        # def round_down(values, decs=0):
        #     return np.floor(values * 10 ** decs) / (10 ** decs)
        #
        # if negate:
        #     current_outputs_negated = round_down(current_outputs_negated, rounding_precision)
        #     thresholds = np.round(thresholds, rounding_precision)
        # else:
        #     current_outputs_negated = np.round(current_outputs_negated, rounding_precision)
        #     thresholds = round_down(thresholds, rounding_precision)

        # # TODO: Remove once testing for explanations is completed
        # if negate:
        #     m = current_outputs_negated.lt(thresholds)
        # else:
        #     if force_negate:
        #         m = current_outputs_negated.gt(thresholds)
        #     else:
        #         m = current_outputs_negated.ge(thresholds) | torch.allclose(current_outputs_negated, thresholds)
        # print(type(self), "CO", current_outputs, "CON", current_outputs_negated, "T", thresholds,
        #       "RT", required_output_threshold, "W", parent_weights, "APW", all_parent_weights,
        #       "IF", input_features, "IFNEG", input_features_negated,
        #       "M", m, "NEGATE", negate, "OO", other_outputs, "PT", parent_logic_type)

        # if the logic is negated then the result of the current logic can't produce an output
        # greater than 1 - required thresholds
        if negate:
            # return current_outputs_negated.lt(thresholds), thresholds
            return current_outputs_negated.lt(thresholds), thresholds, current_outputs_negated, other_outputs
        if force_negate:
            # if force_negate then both conditions must be strict inequalities since we are negating
            # return current_outputs_negated.gt(thresholds), thresholds
            return current_outputs_negated.gt(thresholds), thresholds, current_outputs_negated, other_outputs
        # return current_outputs_negated.ge(thresholds) | torch.allclose(current_outputs_negated, thresholds), thresholds
        return current_outputs_negated.ge(thresholds) | torch.allclose(current_outputs_negated, thresholds), thresholds, current_outputs_negated, other_outputs

    def _compute_required_inputs(
            self,
            parent_weights: torch.Tensor,
            required_output_threshold: torch.Tensor,
            parent_logic_type: str,
            negate: bool = False,
            force_negate: bool = False,
            rounding_precision: int = 3
    ) -> torch.Tensor:
        if isinstance(parent_weights, torch.Tensor):
            parent_weights = parent_weights.numpy()
        if isinstance(required_output_threshold, torch.Tensor):
            required_output_threshold = required_output_threshold.numpy()

        if negate:
            required_output_threshold = 1.0 - required_output_threshold

        # handle zero weights
        zero_weight_mask = parent_weights == 0
        non_zero_indices = np.where(~zero_weight_mask)[0]
        parent_weights = parent_weights[~zero_weight_mask]

        # adjust threshold for logic type
        if parent_logic_type == 'And':
            required_output_threshold = 1.0 - required_output_threshold
        elif parent_logic_type == 'Or':
            pass
        elif parent_logic_type == 'XOr':
            required_output_threshold = 1.0 - required_output_threshold
        else:
            raise AssertionError("`parent_logic_type` must be one of 'And', 'Or', 'XOr'.")

        # sign of weights to flip required inputs where necessary
        if force_negate:
            weights_sign = (parent_weights < 0).astype(int)
        else:
            weights_sign = (parent_weights > 0).astype(int)

        # handle parent weights that can't fully contribute
        parent_weights_too_small = np.round(np.abs(parent_weights)
                                            - required_output_threshold/parent_weights.shape[0], rounding_precision)
        parent_weights_too_small_mask = parent_weights_too_small < 0.0
        if np.any(parent_weights_too_small_mask):
            parent_weights_not_too_small_indices = np.where(~parent_weights_too_small_mask)[0]
            parent_weights_too_small_indices = np.where(parent_weights_too_small_mask)[0]
            # if none of the weights are large enough then there is nothing to distribute.
            if not np.any(~parent_weights_too_small_mask):
                parent_weights_too_small_mask = np.array([False])
            else:
                parent_weights_too_small_diff = parent_weights_too_small[parent_weights_too_small_mask].sum()
                parent_weights_too_small_diff_spread = (parent_weights_too_small_diff /
                                                        sum(~parent_weights_too_small_mask) * -1)

        # matrix of one-hot weights
        parent_weights = (parent_weights.reshape(
            1, -1).repeat(parent_weights.shape[0], axis=0)
                          * np.identity(parent_weights.shape[0]))

        # vector of required_output_thresholds / count of weights
        required_output_threshold = np.array(
            required_output_threshold).repeat(parent_weights.shape[0]) \
                                    / parent_weights.shape[0]

        # adjust required output thresholds based on weights that were too small
        if np.any(parent_weights_too_small_mask):
            required_output_threshold_pos_adjustment = np.zeros_like(required_output_threshold)
            required_output_threshold_neg_adjustment = np.zeros_like(required_output_threshold)
            np.put_along_axis(
                required_output_threshold_pos_adjustment, parent_weights_not_too_small_indices,
                parent_weights_too_small_diff_spread, axis=0)
            np.put_along_axis(
                required_output_threshold_neg_adjustment, parent_weights_too_small_indices,
                parent_weights_too_small_diff, axis=0)
            required_output_threshold = (required_output_threshold
                                         + required_output_threshold_pos_adjustment
                                         + required_output_threshold_neg_adjustment)

        # computes the input required for each to contribute equally to the required output
        required_inputs = np.linalg.solve(np.abs(parent_weights), required_output_threshold)

        # adjust required inputs for logic type
        if parent_logic_type == 'And':
            required_inputs = 1.0 - required_inputs
        elif parent_logic_type == 'Or':
            pass
        elif parent_logic_type == 'XOr':
            required_inputs = 1.0 - required_inputs
        else:
            raise AssertionError("`parent_logic_type` must be one of 'And', 'Or', 'XOr'.")

        required_inputs = np.round(
            weights_sign * required_inputs + (1.0 - weights_sign) * (1.0 - required_inputs), rounding_precision)

        # # TODO: remove this if you won't use a strictly < condition for negations
        # # ensure that inputs are strictly greater than or less than the required threshold
        # if negate:
        #     ws_adj_sub = (weights_sign.astype(bool)).astype(int)
        #     ws_adj_add = (~weights_sign.astype(bool)).astype(int)
        # else:
        #     ws_adj_sub = (~weights_sign.astype(bool)).astype(int)
        #     ws_adj_add = (weights_sign.astype(bool)).astype(int)
        # adj_sub = np.ones_like(required_inputs) / (10 ** rounding_precision) * ws_adj_sub
        # adj_add = np.ones_like(required_inputs) / (10 ** rounding_precision) * ws_adj_add
        # required_inputs -= adj_sub
        # required_inputs += adj_add
        #
        # # # TODO: Remove when testing is complete
        # # print("ADJ_ADD", adj_add, "ADJ_SUB", adj_sub, "WS_A", ws_adj_add, "WS_S", ws_adj_sub, "WS", weights_sign, "PW",
        # #       parent_weights, "RI", required_inputs, "NEG", negate)

        # handle zero weights by reconstructing array of original size
        if np.any(zero_weight_mask):
            tmp_required_inputs = np.zeros_like(zero_weight_mask.astype(float))
            np.put_along_axis(tmp_required_inputs, non_zero_indices, required_inputs, axis=0)
            required_inputs = tmp_required_inputs

        required_inputs = np.round(required_inputs, rounding_precision)

        # # TODO: Remove after testing is done
        # print("RI", required_inputs, "ROT", required_output_threshold,
        #       "PW", parent_weights.sum(axis=1), "PLT", parent_logic_type)

        return torch.tensor(np.clip(required_inputs, 0.0, 1.0)).float()

    @staticmethod
    def _generate_feature_importance(
            current_outputs_negated: torch.Tensor,
            parent_weight: torch.Tensor,
            feature_importances_type: str,
            parent_logic_type: str,
            required_output_thresholds: torch.Tensor,
            other_outputs: torch.Tensor
    ) -> float:
        """
        Produce one of the feature importance values

        Args:
            current_outputs_negated (Tensor): The negated outputs of current nodes being currently evaluated
            parent_weight (Tensor): The parent weight on the current output
            i (int): The index in parent weights corresponding to the current output being evaluated
            feature_importances_type (str): The type of feature importance value to return
            parent_logic_type (str): The logic of the parent node
            required_output_thresholds (Tensor): The required output of the node
            other_outputs (Tensor): The value of the other outputs in the currently evaluated logic

        Returns:
            float: feature importance
        """
        feature_value = current_outputs_negated.item()
        contribution_weight = np.abs(parent_weight.item())
        if feature_importances_type == 'weight':
            importance_value = contribution_weight
        elif feature_importances_type == "weight_proportion":
            importance_value = contribution_weight / parent_weight.abs().sum().item()
            # print("predicate", weight, contribution_weight, parent_weights.abs().sum().item(), importance_value)
        elif feature_importances_type == "influence":
            if parent_logic_type == "And":
                importance_value = 1. - (1. - feature_value) * contribution_weight
            elif parent_logic_type == "Or":
                importance_value = feature_value * contribution_weight
            else:
                importance_value = np.nan
        elif feature_importances_type == 'incremental_influence':
            importance_value = min(max(required_output_thresholds.item() - other_outputs.item(), 0.), 1.)
        elif feature_importances_type == 'capped_weight':
            if feature_value == 0. and parent_logic_type == "And":
                importance_value = contribution_weight
            elif feature_value == 0. and parent_logic_type == "Or":
                importance_value = 0.
            else:
                importance_value = min(contribution_weight, 1. / feature_value)
        else:
            importance_value = np.nan

        return importance_value
