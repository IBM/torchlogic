import logging

import numpy as np

import torch

from torchlogic.utils.explanations import register_hooks
from ._base_mixin import BaseReasoningNetworkMixin
from torchlogic.utils.explanations import simplification


class ReasoningNetworkRegressorMixin(BaseReasoningNetworkMixin):

    ROUNDING_PRECISION = 32
    EPS = 1e-2

    def __init__(
            self,
            output_size: int
    ):
        super(ReasoningNetworkRegressorMixin, self).__init__()
        self.output_size = output_size
        self.logger = logging.getLogger(self.__class__.__name__)

    def explain(
            self,
            quantile: float = 0.5,
            required_output_thresholds: torch.Tensor = torch.tensor(0.9),
            threshold: float = None,
            explain_type: str = 'both',
            print_type: str = 'logical',
            target_names: list = ['positive'],
            explanation_prefix: str = "A sample has a",
            ignore_uninformative: bool = False,
            rounding_precision: int = 3,
            inverse_transform_features=None,
            inverse_transform_target=None,
            show_bounds: bool = True,
            simplify: bool = False,
            exclusions: list[str] = None
    ) -> str:
        """
        Produce a global explanation.

        Args:
            quantile (float): Quantile of logic to produce.
            required_output_thresholds (float): threshold below which inner logic will be masked and excluded.
            threshold (float): Weight threshold to use.  If not None, then overrides quantile
            explain_type (str): One of `both`, `positive`, `negative`. Control types of logic to produce.
            print_type (str): one of 'logical' or 'natural'.  If logical, prints logic, otherwise, prints natural text.
            target_names (list): list of target names.
            explanation_prefix (str): text to precede explanation.
            ignore_uninformative (bool): If True, ignore uninformative logic that has threshold at 0 or 1.
            rounding_precision (int): precision for rounding percentiles during explanation.
            inverse_transform_features (sklearn Transformation): Skleran inverse transformation function for features
            inverse_transform_target (sklearn Transformation): Skleran inverse transformation function for target
            show_bounds (bool): include numeric boundary used in logic
            simplify (bool): If True, return a simplified version of the explanation using logical rules

        Returns:
            str: explanation
        """
        force_negate = required_output_thresholds < 0.5

        explanation = []
        for output in range(self.output_size):
            op_explain = self.output_layer.explain(
                quantile=quantile,
                required_output_thresholds=required_output_thresholds,
                threshold=threshold,
                parent_weights=torch.tensor(1.),
                parent_logic_type='Or',
                depth=0,
                explain_type=explain_type,
                print_type=print_type if print_type == 'natural' else 'logical',
                channel=output,
                force_negate=force_negate,
                ignore_uninformative=ignore_uninformative,
                rounding_precision=self.ROUNDING_PRECISION,
                inverse_transform=inverse_transform_features,
                show_bounds=show_bounds,
                original_rounding_precision=rounding_precision,
            )
            explanation_str = ', '.join(np.unique(op_explain).tolist())

            if not explanation_str:
                UserWarning("Could not generate explanation with given settings.  "
                            "Searching setting space instead.")

                search_params = self._build_search_space(
                    quantile=quantile,
                    explain_type=explain_type,
                    ignore_uninformative=ignore_uninformative,
                    rounding_precision=rounding_precision
                )
                for params in search_params:
                    op_explain = self.root_layer.explain(
                        required_output_thresholds=required_output_thresholds,
                        threshold=threshold,
                        parent_weights=torch.ones(1),
                        parent_logic_type='Or',
                        depth=0,
                        print_type=print_type if print_type == 'natural' else 'logical',
                        channel=output,
                        force_negate=force_negate,
                        inverse_transform=inverse_transform_features,
                        show_bounds=show_bounds,
                        original_rounding_precision=rounding_precision,
                        **params
                    )
                    explanation_str = ', '.join(np.unique(op_explain).tolist())
                    if explanation_str:
                        break

            if not explanation_str:
                raise RuntimeError("Could not specify parameters to create explanation.  "
                                   "Check that required_output_thresholds is in range of model outputs.")

            if inverse_transform_target is not None:
                required_output_thresholds = inverse_transform_target(required_output_thresholds.reshape(1, -1))

            explanation_str = simplification(
                explanation_str,
                print_type,
                simplify,
                sample_level=False,
                ndigits=rounding_precision,
                exclusions=exclusions
            )
            explanation += [f"{explanation_prefix} predicted {target_names[output]} "
                            f"of {round(float(required_output_thresholds), rounding_precision)} "
                            f"because: \n\n{explanation_str}"]

        explanation = '\n\n'.join(explanation)
        if print_type == 'natural':
            explanation = '.  '.join([x[:1].capitalize() + x[1:] for x in explanation.split('.  ')])

        return explanation

    def explain_samples(
            self,
            x: torch.Tensor,
            quantile: float = 0.5,
            threshold: float = None,
            target_names: list = ['positive'],
            explain_type: str = 'both',
            print_type: str = 'logical',
            sample_explanation_prefix: str = "The sample has a",
            ignore_uninformative: bool = False,
            rounding_precision: int = 3,
            inverse_transform_features=None,
            inverse_transform_target=None,
            show_bounds: bool = True,
            simplify: bool = False,
            exclusions: list[str] = None,
            min_max_feature_dict: dict = None,
            feature_importances: bool = False,
            feature_importances_type: str = 'weight'
    ) -> str:
        """
        Produce a sample explanation.

        Args:
            x (torch.Tensor): features for sample.
            quantile (float): Quantile of logic to produce.
            threshold (float): Weight threshold to use.  If not None, then overrides quantile
            target_names (list): list of target names.
            explain_type (str): One of `both`, `positive`, `negative`. Control types of logic to produce.
            print_type (str): one of 'logical' or 'natural'.  If logical, prints logic, otherwise, prints natural text.
            sample_explanation_prefix (str): text to precede explanation.
            ignore_uninformative (bool): If True, ignore uninformative logic that has threshold at 0 or 1.
            rounding_precision (int): precision for rounding percentiles during explanation.
            inverse_transform_features (sklearn Transformation): Skleran inverse transformation function for features
            inverse_transform_target (sklearn Transformation): Skleran inverse transformation function for target
            show_bounds (bool): include numeric boundary used in logic
            simplify (bool): If True, return a simplified version of the explanation using logical rules

        Returns:
            str: explanation
        """
        x = x.to(self.output_layer.weights.device)
        sample_explanations = []
        for i in range(x.size(0)):

            # collect predictions and outputs from each layer
            outputs = {}
            register_hooks(self.rn, outputs)
            predictions = self.rn(x[i].unsqueeze(0))

            # boosted predictions
            boosted_predictions = None
            if hasattr(self, 'xgb'):
                if self.xgb_is_fitted:
                    boosted_predictions = predictions.cpu().detach() + self.xgb.predict(
                        x[i].unsqueeze(0).detach().cpu())

            # collect explanations
            explanation = []
            parent_weights = torch.tensor(1.)
            required_output_threshold = predictions[0, 0].detach().cpu().squeeze()

            if boosted_predictions is not None:
                prediction_value = boosted_predictions.squeeze()
            else:
                prediction_value = required_output_threshold
            force_negate = prediction_value < 0.5

            op_explain = self.output_layer.explain_sample(
                outputs_dict=outputs,
                # the negative class is asking the question "why didn't you get a higher value?"
                # also solves for numerical instability.
                required_output_thresholds=min(
                    required_output_threshold * (1 + self.EPS if force_negate else 1 - self.EPS), 1.0),
                quantile=quantile,
                threshold=threshold,
                parent_weights=parent_weights,
                parent_logic_type='And',
                depth=0,
                explain_type=explain_type,
                print_type=print_type if print_type == 'natural' else 'logical',
                input_features=x[i].cpu(),
                channel=0,
                force_negate=force_negate,
                ignore_uninformative=ignore_uninformative,
                rounding_precision=self.ROUNDING_PRECISION,
                inverse_transform=inverse_transform_features,
                show_bounds=show_bounds,
                original_rounding_precision=rounding_precision,
            )
            explanation_str = ', '.join(np.unique(op_explain).tolist())

            if not explanation_str:
                UserWarning("Could not generate explanation with given settings.  "
                            "Searching setting space instead.")

                search_params = self._build_search_space(
                    quantile=quantile,
                    explain_type=explain_type,
                    ignore_uninformative=ignore_uninformative,
                    rounding_precision=rounding_precision
                )
                for params in search_params:
                    op_explain = self.root_layer.explain_sample(
                        outputs_dict=outputs,
                        # the negative class is asking the question "why didn't you get a higher value?"
                        # also solves for numerical instability.
                        required_output_thresholds=min(
                            required_output_threshold * (1 + self.EPS if force_negate else 1 - self.EPS), 1.0),
                        threshold=threshold,
                        parent_weights=parent_weights,
                        parent_logic_type='And',
                        depth=0,
                        print_type=print_type if print_type == 'natural' else 'logical',
                        input_features=x[i].cpu(),
                        channel=0,
                        force_negate=force_negate,
                        inverse_transform=inverse_transform_features,
                        show_bounds=show_bounds,
                        original_rounding_precision=rounding_precision,
                        **params
                    )
                    explanation_str = ', '.join(np.unique(op_explain).tolist())
                    if explanation_str:
                        break

            if not explanation_str:
                raise RuntimeError("Could not produce explanation! Check print_samples output for details.")

            if inverse_transform_target is not None:
                prediction_value = inverse_transform_target(prediction_value.reshape(1, -1))

            if print_type in ['logical', 'logical-natural'] or feature_importances:
                explanation_tree = simplification(
                    explanation_str,
                    print_type,
                    simplify=simplify,
                    sample_level=True,
                    ndigits=rounding_precision,
                    exclusions=exclusions,
                    min_max_feature_dict=min_max_feature_dict,
                    feature_importances=feature_importances,
                    verbose=False  # TODO use False when done testing
                )
                explanation_str = explanation_tree.tree_to_string()

                if feature_importances:
                    importances_dict = {}
                    self._feature_importances = explanation_tree.get_feature_importances(importances_dict=importances_dict)

            explanation += [
                f"{i}: {sample_explanation_prefix} "
                f"a predicted {target_names[0]} "
                f"of {round(float(prediction_value), rounding_precision)} "
                f"because: \n\n{explanation_str}"]

            sample_explanations += ['\n'.join(explanation)]
        sample_explanations = [x for x in sample_explanations if x]

        sample_explanations = '\n'.join(sample_explanations)
        if print_type == 'natural':
            sample_explanations = '.  '.join([x[:1].capitalize() + x[1:] for x in sample_explanations.split('.  ')])

        return sample_explanations

    def print(
            self,
            quantile=0.5,
            required_output_thresholds: torch.Tensor = None,
            threshold=None,
            explain_type='both',
            print_type='logical',
            target_names=['positive'],
            ignore_uninformative: bool = False,
            rounding_precision: int = 3,
            inverse_transform=None,
            show_bounds: bool = True
    ):
        """
        Produce a global explanation view of the model.

        Args:
            required_output_thresholds (float): threshold below which inner logic will be masked and excluded.
            quantile (float): Quantile of logic to produce.
            threshold (float): Weight threshold to use.  If not None, then overrides quantile
            explain_type (str): One of `both`, `positive`, `negative`. Control types of logic to produce.
            print_type (str): one of 'logical' or 'natural'.  If logical, prints logic, otherwise, prints natural text.
            target_names (list): list of target names.
            ignore_uninformative (bool): If True, ignore uninformative logic that has threshold at 0 or 1.
            rounding_precision (int): precision for rounding percentiles during explanation.
            inverse_transform (sklearn Transformation): Skleran inverse transformation function for features
            show_bounds (bool): include numeric boundary used in logic
        """
        force_negate = required_output_thresholds < 0.5

        for output in range(self.output_size):
            print(f"REASONING NETWORK MODEL FOR: {target_names[output]}")
            self.output_layer.print(
                quantile=quantile,
                required_output_thresholds=required_output_thresholds,
                threshold=threshold,
                parent_weights=torch.tensor(1.),
                parent_logic_type='Or',
                depth=0,
                explain_type=explain_type,
                print_type=print_type,
                channel=output,
                force_negate=force_negate,
                ignore_uninformative=ignore_uninformative,
                rounding_precision=rounding_precision,
                inverse_transform=inverse_transform,
                show_bounds=show_bounds
            )

    def print_samples(
            self,
            x: torch.Tensor,
            quantile: float = 0.5,
            threshold: float = None,
            target_names: list = ['positive'],
            explain_type: str = 'both',
            print_type: str = 'logical',
            ignore_uninformative: bool = False,
            rounding_precision: int = 3,
            inverse_transform=None,
            show_bounds: bool = True
    ):
        """
        Print a sample explanation view of the model.

        Args:
            x (torch.Tensor): features for sample.
            quantile (float): Quantile of logic to produce.
            threshold (float): Weight threshold to use.  If not None, then overrides quantile
            target_names (list): list of target names.
            explain_type (str): One of `both`, `positive`, `negative`. Control types of logic to produce.
            print_type (str): one of 'logical' or 'natural'.  If logical, prints logic, otherwise, prints natural text.
            ignore_uninformative (bool): If True, ignore uninformative logic that has threshold at 0 or 1.
            rounding_precision (int): precision for rounding percentiles during explanation.
            inverse_transform (sklearn Transformation): Skleran inverse transformation function for features
            show_bounds (bool): include numeric boundary used in logic
        """
        x = x.to(self.output_layer.weights.device)
        for i in range(x.size(0)):

            # collect predictions and outputs from each layer
            outputs = {}
            register_hooks(self, outputs)
            predictions = self.rn(x[i].unsqueeze(0))

            # boosted predictions
            boosted_predictions = None
            if hasattr(self, 'xgb'):
                if self.xgb_is_fitted:
                    boosted_predictions = predictions.cpu().detach() + self.xgb.predict(
                        x[i].unsqueeze(0).detach().cpu())

            # collect explanations
            parent_weights = torch.tensor(1.)
            required_output_threshold = predictions[0, 0].detach().cpu().squeeze()

            if boosted_predictions is not None:
                prediction_value = boosted_predictions.squeeze()
            else:
                prediction_value = required_output_threshold
            force_negate = prediction_value < 0.5

            print(f"REASONING NETWORK MODEL FOR: {target_names[0]}")
            self.output_layer.print_sample(
                outputs_dict=outputs,
                # the negative class is asking the question "why didn't you get a higher value?"
                # also solves for numerical instability.
                required_output_thresholds=min(
                    required_output_threshold * (1 + self.EPS if force_negate else 1 - self.EPS), 1.0),
                quantile=quantile,
                threshold=threshold,
                parent_weights=parent_weights,
                parent_logic_type='And',
                depth=0,
                explain_type=explain_type,
                print_type=print_type,
                input_features=x[i].cpu(),
                channel=0,
                force_negate=force_negate,
                ignore_uninformative=ignore_uninformative,
                rounding_precision=rounding_precision,
                inverse_transform=inverse_transform,
                show_bounds=show_bounds
            )


__all__ = [ReasoningNetworkRegressorMixin]
