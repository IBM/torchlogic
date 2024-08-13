import logging

import torch
import numpy as np

from ._core import LukasiewiczCore


class BasePredicates(LukasiewiczCore):

    """
    Class for Predicates.  Aid in explanation.
    """

    def __init__(self, feature_names):
        super(BasePredicates, self).__init__()
        self.operands = feature_names
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _produce_negation_string(
            weight: float,
            text: str,
            explain_type: str = 'both',
            print_type: str = 'logical',
            negate: bool = False,
            **kwargs
    ) -> str:
        """
        Produce a negation under certain conditions.

        Args:
            weight (float): Weight for specific predicate.
            text(str): Text for specific predicate.
            explain_type (str): One of `both`, `positive`, `negative`. Control types of logic to produce.
            print_type (str): one of 'logical' or 'natural'.  If logical, prints logic, otherwise, prints natural text.

        Returns:
            str: The text with negation applied.
        """
        assert print_type in ['logical', 'logical-natural', 'natural'], \
            "'print_type' must be one of 'logical', 'logical-natural', or 'natural'"

        # conditions that produce a negative
        if (((weight < 0 and not negate) or (weight >= 0 and negate)) and text != "") \
                and (explain_type == 'both' or explain_type == 'negative'):
            if print_type == 'logical':
                return f"NOT({text})".replace("NOT(\n", "\nNOT(")
            elif print_type == 'logical-natural' or print_type == 'natural':
                text = f"NOT {text}"
                return f"{text}"
            else:
                raise ValueError("`print_type` must be one of `logical`, `logical-natural`, or `natural`")
        elif (((weight < 0 and not negate) or (weight >= 0 and negate)) and text != "") \
                and explain_type == 'positive':
            return ""
        # conditions that produce a positive
        if (((weight < 0 and negate) or (weight >= 0 and not negate)) and text != "") \
                and (explain_type == 'both' or explain_type == 'positive'):
            return text
        elif (((weight < 0 and negate) or (weight >= 0 and not negate)) and text != "") \
                and explain_type == 'negative':
            return ""
        else:
            return text

    def _produce_predicate_value(
            self,
            name: str,
            weight: float,
            input_value: float,
            required_output_threshold: float,
            ignore_uninformative: bool,
            print_type: str = 'logical',
            rounding_precision: int = 3,
            negate: bool = False,
            inverse_transform=None,
            show_bounds: bool = True,
            **kwargs
    ) -> str:
        """
        Produce the string for predicate explanation.

        Args:
            name (str): predicate name.
            weight (float): predicates weight.
            required_output_threshold (float): required output below which predicate will be excluded from explanation.
            ignore_uninformative (bool): If True, ignore uninformative logic that has threshold at 0 or 1.
            rounding_precision (int): precision for rounding percentiles during explanation.
            inverse_transform (sklearn transform): inverse of in the inputs transform
            show_bounds (bool): include numeric boundary used in logic

        Returns:
            str: predicate with value explanation.
        """
        # required_output_threshold represents the threshold for each predicate with all negations applied.
        # So we have the following scenarios:
            # if negate == True:
                # input feature negated <= required_output_threshold
                #
                # if w < 0:
                    # 1 - x <= required_output_threshold
                    # x >= 1 - required_output_threshold
                # else:
                    # x <= required_output_threshold
                    # NOT(x >= required_output_threshold)
            # else:
                # input feature negated >= threshold
                #
                # if w < 0:
                    # 1 - x >= required_output_threshold
                    # x <= 1 - required_output_threshold
                    # NOT(x >= 1 - required_output_threshold)
                # else:
                    # x >= required_output_threshold

        if not negate:
            if ((np.isclose(round(required_output_threshold, rounding_precision), 0.0) and weight >= 0)
                or (np.isclose(round(1.0 - required_output_threshold,
                          rounding_precision), 1.0) and weight <= 0)) and ignore_uninformative:
                return ""
        else:
            if ((np.isclose(round(1.0 - required_output_threshold, rounding_precision), 0.0) and weight <= 0)
                or (np.isclose(round(required_output_threshold,
                          rounding_precision), 1.0) and weight >= 0)) and ignore_uninformative:
                return ""

        if print_type == 'logical':
            gte_str = ">="
        elif print_type == 'logical-natural' or print_type == 'natural':
            gte_str = "greater than or equal to"
        else:
            raise AssertionError("print_type must be 'logical', 'logical-natural', or 'natural'.")

        # if a search has been performed still display using the originally requested rounding precision
        rounding_precision = kwargs.get('original_rounding_precision', rounding_precision)
        if inverse_transform is not None:
            input_array = np.zeros((1, len(self.operands)))
            input_array_actuals = np.zeros((1, len(self.operands)))
            idx = self.operands.index(name)
            if weight >= 0:
                input_array[0, idx] = required_output_threshold
            else:
                input_array[0, idx] = 1.0 - required_output_threshold
            transformed_output_threshold = inverse_transform(input_array)[0, idx]

            input_array_actuals[0, idx] = input_value
            transformed_input_value = inverse_transform(input_array_actuals)[0, idx]

            if show_bounds:
                # return f"{name} {gte_str} {round(transformed_output_threshold, rounding_precision)} [actual: {round(transformed_input_value, rounding_precision)}]"
                return f"{name} {gte_str} {round(transformed_output_threshold, rounding_precision)}"
            return f"{name}"

        if weight >= 0:
            if show_bounds:
                # if negate is true then produce_negation_string will wrap
                # this in NOT() which is equivalent to the LTE condition
                # TODO: do we want to show the actuals?
                # return f"{name} {gte_str} {round(required_output_threshold, rounding_precision)} [actual: {round(input_value, rounding_precision)}]"
                return f"{name} {gte_str} {round(required_output_threshold, rounding_precision)}"
            return f"{name}"
        if show_bounds:
            # if negate is true then produce_negation_string will
            # wrap this in NOT() which is equivalent to the LTE condition
            # TODO: do we want to show the actuals?
            # return f"{name} {gte_str} {round(1 - required_output_threshold, rounding_precision)} [actual: {round(input_value, rounding_precision)}]"
            return f"{name} {gte_str} {round(1 - required_output_threshold, rounding_precision)}"
        return f"{name}"

    def explain_sample(
            self,
            quantile: float = 0.5,
            required_output_thresholds: torch.Tensor = None,
            threshold: float = None,
            parent_weights: torch.Tensor = None,
            parent_mask: torch.Tensor = None,
            explain_type: str = 'both',
            parent_logic_type: str = None,
            negate: bool = False,
            input_features: torch.Tensor = None,
            force_negate: bool = False,
            depth: int = 0,
            print_explanation: bool = False,
            global_explain: bool = False,
            ignore_uninformative: bool = False,
            print_type: str = 'logical',
            rounding_precision: int = 3,
            inverse_transform=None,
            show_bounds: bool = True,
            feature_importances: bool = False,
            feature_importances_type: str = '',
            **kwargs
    ) -> list:
        """
        Produce a sample explanation.

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
            input_features (torch.Tensor): tensor of input features.
            force_negate (bool): If True, extract the negation of logic.
            global_explain (bool): If True, perform a global explanation.
            print_explanation (bool): If True, print the explanation.
            ignore_uninformative (bool): If True, ignore uninformative logic that has threshold at 0 or 1.
            rounding_precision (int): precision for rounding percentiles during explanation.
            inverse_transform (sklearn transform): inverse of in the inputs transform
            show_bounds (bool): include numeric boundary used in logic

        Returns:
            list: explanation texts.
        """
        out = np.array(self.operands)
        parent_weights_mask = self._produce_weights_mask(parent_weights.squeeze(), quantile, threshold)

        if parent_mask is not None:
            out = out[parent_mask]
            if isinstance(out, str):
                out = np.array(out)
            if not global_explain:
                input_features = input_features[parent_mask]

        # ensure all are iterable
        out = out.reshape(-1, )
        parent_weights = parent_weights.reshape(-1, )
        parent_weights_mask = parent_weights_mask.reshape(-1, )

        output = []
        for i, (name, weight, pwm) in enumerate(zip(out, parent_weights, parent_weights_mask)):
            if pwm:
                if global_explain:
                    input_features = self._compute_required_inputs(
                        parent_weights=parent_weights,
                        required_output_threshold=required_output_thresholds,
                        parent_logic_type=parent_logic_type,
                        negate=negate,
                        force_negate=force_negate,
                        rounding_precision=rounding_precision
                    )
                    co = input_features[i].squeeze()
                    input_features = input_features.squeeze()
                else:
                    co = input_features[i].squeeze()

                # current_outputs_mask, next_required_output_thresholds = self._produce_current_outputs_mask(
                current_outputs_mask, next_required_output_thresholds, current_outputs_negated, other_outputs = self._produce_current_outputs_mask(
                    co,
                    input_features,
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

                if feature_importances:
                    importance_value = self._generate_feature_importance(
                        current_outputs_negated=current_outputs_negated,
                        parent_weight=parent_weights[i],
                        feature_importances_type=feature_importances_type,
                        parent_logic_type=parent_logic_type,
                        required_output_thresholds=required_output_thresholds,
                        other_outputs=other_outputs
                    )

                if current_outputs_mask:
                    current_output = self._produce_negation_string(
                        weight, self._produce_predicate_value(
                            name, weight, float(co), float(next_required_output_thresholds), ignore_uninformative,
                            print_type, rounding_precision, negate, inverse_transform, show_bounds, **kwargs),
                        explain_type=explain_type, print_type=print_type, negate=force_negate)
                    if feature_importances:
                        if current_output[-1] == ')':
                            current_output = current_output[:-1] + ' [' + str(importance_value) + '])'
                        else:
                            current_output += ' [' + str(importance_value) + ']'
                    output += [current_output]

        output = np.unique([x for x in output if x.replace("\n", "").replace("\t", "") != "NOT()" and x != ""]).tolist()
        if feature_importances:
            output = [x for x in output if x.replace("\n", "").replace("\t", "") != "NOT()" and x != ""]
        
        if print_explanation:
            explanation_str = ', '.join(output)
            if input_features is not None:
                print(f"Logic at depth {depth}: {explanation_str}"
                      f"\noutput: {input_features.squeeze()}\n")
            else:
                print(f"Logic at depth {depth}: {explanation_str}\n")
        
        if print_type == 'natural':
            if parent_logic_type == 'And':
                output = [', and '.join(output)]
            elif parent_logic_type == 'Or':
                output = [', or '.join(output)]

        return output

    def explain(
            self,
            quantile: float = 0.5,
            required_output_thresholds: torch.Tensor = None,
            threshold: float = None,
            parent_weights: torch.Tensor = None,
            parent_mask: torch.Tensor = None,
            explain_type: str = 'both',
            print_type: str = 'logical',
            parent_logic_type: str = None,
            negate: bool = False,
            input_features: torch.Tensor = None,
            force_negate: bool = False,
            depth: int = 0,
            ignore_uninformative: bool = False,
            rounding_precision: int = 3,
            inverse_transform=None,
            feature_importance=False, # TODO check if it should be False by default
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
            input_features (torch.Tensor): tensor of input features.
            force_negate (bool): If True, extract the negation of logic.
            ignore_uninformative (bool): If True, ignore uninformative logic that has threshold at 0 or 1.
            rounding_precision (int): precision for rounding percentiles during explanation.
            inverse_transform (sklearn transform): inverse of in the inputs transform

        Returns:
            list: explanation texts.
        """
        return self.explain_sample(
            quantile=quantile,
            required_output_thresholds=required_output_thresholds,
            threshold=threshold,
            parent_weights=parent_weights,
            parent_mask=parent_mask,
            explain_type=explain_type,
            parent_logic_type=parent_logic_type,
            negate=negate,
            input_features=input_features,
            force_negate=force_negate,
            depth=depth,
            global_explain=True,
            ignore_uninformative=ignore_uninformative,
            print_type=print_type,
            rounding_precision=rounding_precision,
            inverse_transform=inverse_transform,
            feature_importance=feature_importance,
            **kwargs
        )

    def print_sample(
            self,
            quantile: float = 0.5,
            required_output_thresholds: torch.Tensor = None,
            threshold: float = None,
            parent_weights: torch.Tensor = None,
            parent_mask: torch.Tensor = None,
            explain_type: str = 'both',
            print_type: str = 'logical',
            parent_logic_type: str = None,
            negate: bool = False,
            input_features: torch.Tensor = None,
            force_negate: bool = False,
            depth: int = 0,
            ignore_uninformative: bool = False,
            rounding_precision: int = 3,
            inverse_transform=None,
            **kwargs
    ) -> list:
        """
        Print a sample explanation view of the model.

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
            input_features (torch.Tensor): tensor of input features.
            force_negate (bool): If True, extract the negation of logic.
            ignore_uninformative (bool): If True, ignore uninformative logic that has threshold at 0 or 1.
            rounding_precision (int): precision for rounding percentiles during explanation.
            inverse_transform (sklearn transform): inverse of in the inputs transform
        """
        return self.explain_sample(
            quantile=quantile,
            required_output_thresholds=required_output_thresholds,
            threshold=threshold,
            parent_weights=parent_weights,
            parent_mask=parent_mask,
            explain_type=explain_type,
            parent_logic_type=parent_logic_type,
            negate=negate,
            input_features=input_features,
            force_negate=force_negate,
            depth=depth,
            print_explanation=True,
            ignore_uninformative=ignore_uninformative,
            print_type=print_type,
            rounding_precision=rounding_precision,
            inverse_transform=inverse_transform,
            **kwargs
        )

    def print(
            self,
            quantile: float = 0.5,
            required_output_thresholds: torch.Tensor = None,
            threshold: float = None,
            parent_weights: torch.Tensor = None,
            parent_mask: torch.Tensor = None,
            explain_type: str = 'both',
            print_type: str = 'logical',
            parent_logic_type: str = None,
            negate: bool = False,
            input_features: torch.Tensor = None,
            force_negate: bool = False,
            depth: int = 0,
            ignore_uninformative: bool = False,
            rounding_precision: int = 3,
            inverse_transform=None,
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
            input_features (torch.Tensor): tensor of input features.
            force_negate (bool): If True, extract the negation of logic.
            ignore_uninformative (bool): If True, ignore uninformative logic that has threshold at 0 or 1.
            rounding_precision (int): precision for rounding percentiles during explanation.
            inverse_transform (sklearn transform): inverse of in the inputs transform
        """
        return self.explain_sample(
            quantile=quantile,
            required_output_thresholds=required_output_thresholds,
            threshold=threshold,
            parent_weights=parent_weights,
            parent_mask=parent_mask,
            explain_type=explain_type,
            parent_logic_type=parent_logic_type,
            negate=negate,
            input_features=input_features,
            force_negate=force_negate,
            depth=depth,
            global_explain=True,
            print_explanation=True,
            ignore_uninformative=ignore_uninformative,
            print_type=print_type,
            rounding_precision=rounding_precision,
            inverse_transform=inverse_transform,
            **kwargs
        )
