import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.special import expit

import torch
from torch.utils.data import DataLoader

from torchlogic.utils.explanations import register_hooks
from ._base_mixin import BaseReasoningNetworkMixin
from torchlogic.utils.explanations import simplification


class ReasoningNetworkClassifierMixin(BaseReasoningNetworkMixin):

    ROUNDING_PRECISION = 32
    EPS = 1e-2

    def __init__(
            self,
            output_size: int,
            logits: bool = False
    ):
        super(ReasoningNetworkClassifierMixin, self).__init__()
        self.output_size = output_size
        self.logits = logits
        self.logger = logging.getLogger(self.__class__.__name__)

    def explain(
            self,
            quantile: float = 0.5,
            required_output_thresholds: torch.Tensor = torch.tensor(0.9),
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
            inverse_transform (sklearn Transformation): Skleran inverse transformation function
            decision_boundary (float): Classification decision boundary
            show_bounds (bool): include numeric boundary used in logic
            simplify (bool): If True, return a simplified version of the explanation using logical rules

        Returns:
            str: explanation
        """
        if not isinstance(required_output_thresholds, torch.Tensor):
            required_output_thresholds = torch.tensor(required_output_thresholds).float()

        # binary classification?
        if self.output_size == 1:
            force_negate = required_output_thresholds < decision_boundary
            force_negate_str = f"not " if force_negate else ""
        else:
            force_negate = False
            force_negate_str = ""

        explanation = []
        for output in range(self.output_size):
            op_explain = self.root_layer.explain(
                quantile=quantile,
                required_output_thresholds=required_output_thresholds,
                threshold=threshold,
                parent_weights=torch.ones(1),
                parent_logic_type='Or',
                depth=0,
                explain_type=explain_type,
                print_type=print_type if print_type == 'natural' else 'logical',
                channel=output,
                force_negate=force_negate,
                ignore_uninformative=ignore_uninformative,
                rounding_precision=self.ROUNDING_PRECISION,
                inverse_transform=inverse_transform,
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
                        inverse_transform=inverse_transform,
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

            explanation_str = simplification(
                explanation_str,
                print_type,
                simplify,
                sample_level=False,
                ndigits=rounding_precision,
                exclusions=exclusions
            )
            explanation += [f"{explanation_prefix} "
                            f"{force_negate_str}{target_names[output]} because: \n\n{explanation_str}"]

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
            sample_explanation_prefix: str = "Sample was in the",
            ignore_uninformative: bool = False,
            rounding_precision: int = 3,
            inverse_transform=None,
            decision_boundary: float = None,
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
            inverse_transform (sklearn Transformation): Skleran inverse transformation function
            decision_boundary (float): Classification decision boundary
            show_bounds (bool): include numeric boundary used in logic
            simplify (bool): If True, return a simplified version of the explanation using logical rules

        Returns:
            str: explanation
        """
        x = x.to(self.root_layer.weights.device)
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

            if self.logits:
                predictions = torch.exp(predictions)/(torch.exp(predictions) + 1)

            # collect explanations
            explanation = []
            if decision_boundary is None and self.output_size > 1:
                if boosted_predictions is None:
                    class_predictions = [predictions[0].detach().cpu().argmax()]
                else:
                    class_predictions = [boosted_predictions[0].argmax()]
                required_output_threshold = [predictions[0, class_predictions[0]].detach().cpu().squeeze()]
            elif decision_boundary is not None and self.output_size > 1:
                if boosted_predictions is None:
                    required_output_threshold = [predictions[0, j].detach().cpu().squeeze()
                                                 for j in range(self.output_size)
                                                 if predictions[0, j] > decision_boundary]
                    np_predictions = predictions[0].detach().cpu().squeeze()
                else:
                    required_output_threshold = [predictions[0, j].detach().cpu().squeeze()
                                                 for j in range(self.output_size)
                                                 if boosted_predictions[0, j] > decision_boundary]
                    np_predictions = boosted_predictions[0].squeeze()
                class_predictions = torch.where(np_predictions > decision_boundary)[0]

            # binary classification?
            if self.output_size == 1:
                if boosted_predictions is None:
                    class_predictions = [predictions[0].detach().cpu().argmax()]
                    negation_threshold = [predictions[0, class_predictions[0]].detach().cpu().squeeze()]
                else:
                    class_predictions = [boosted_predictions[0].argmax()]
                    negation_threshold = [boosted_predictions[0, class_predictions[0]].squeeze()]
                required_output_threshold = [predictions[0, class_predictions[0]].detach().cpu().squeeze()]
                decision_boundary = 0.5 if decision_boundary is None else decision_boundary
                force_negate = negation_threshold[0] < decision_boundary
                force_negate_str = f"not " if force_negate else ""
                parent_weights = torch.ones(1)
            else:
                force_negate = False
                force_negate_str = ""
                parent_weights = torch.ones(1)

            if feature_importances:
                assert len(class_predictions) < 2, "Feature importances only works for binary classificaiton problems"

            for rot, class_prediction in zip(required_output_threshold, class_predictions):
                op_explain = self.root_layer.explain_sample(
                    outputs_dict=outputs,
                    # the negative class is asking the question "why didn't you get a higher value?"
                    # also solves for numerical instability.
                    required_output_thresholds=min(rot * (1 + self.EPS if force_negate else 1 - self.EPS), 1.0),
                    quantile=quantile,
                    threshold=threshold,
                    parent_weights=parent_weights,
                    parent_logic_type='And',
                    depth=0,
                    explain_type=explain_type,
                    print_type=print_type if print_type == 'natural' else 'logical',
                    input_features=x[i].cpu(),
                    channel=class_prediction,
                    force_negate=force_negate,
                    ignore_uninformative=ignore_uninformative,
                    rounding_precision=self.ROUNDING_PRECISION,
                    inverse_transform=inverse_transform,
                    show_bounds=show_bounds,
                    original_rounding_precision=rounding_precision,
                    feature_importances=feature_importances,
                    feature_importances_type=feature_importances_type
                )
                explanation_str = ', '.join(np.unique(op_explain).tolist())

                if feature_importances and not explanation_str:
                    raise RuntimeError("Could not produce explanation for feature importances! Check print_samples output for details.")

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
                            required_output_thresholds=min(rot * (1 + self.EPS if force_negate else 1 - self.EPS), 1.0),
                            threshold=threshold,
                            parent_weights=parent_weights,
                            parent_logic_type='And',
                            depth=0,
                            print_type=print_type if print_type == 'natural' else 'logical',
                            input_features=x[i].cpu(),
                            channel=class_prediction,
                            force_negate=force_negate,
                            inverse_transform=inverse_transform,
                            show_bounds=show_bounds,
                            original_rounding_precision=rounding_precision,
                            feature_importance=feature_importances,
                            feature_importances_type=feature_importances_type
                            **params
                        )
                        explanation_str = ', '.join(np.unique(op_explain).tolist())
                        if explanation_str:
                            break

                if not explanation_str:
                    raise RuntimeError("Could not produce explanation! Check print_samples output for details.")

                if print_type in ['logical', 'logical-natural'] or feature_importances:
                    # TODO: fix simplification and feature importances for the multi label case
                    explanation_tree = simplification(
                        explanation_str,
                        print_type,
                        simplify=simplify,
                        sample_level=True,
                        ndigits=rounding_precision,
                        exclusions=exclusions,
                        min_max_feature_dict=min_max_feature_dict,
                        feature_importances=feature_importances,
                        verbose=False # TODO use False when done testing
                    )
                    explanation_str = explanation_tree.tree_to_string()

                    if feature_importances:
                        importances_dict = {}
                        self._feature_importances = explanation_tree.get_feature_importances(importances_dict=importances_dict)

                explanation += [
                    f"{i}: {sample_explanation_prefix} "
                    f"{force_negate_str}{target_names[class_prediction]} because: \n\n{explanation_str}"]

                sample_explanations += ['\n'.join(explanation)]
        sample_explanations = [x for x in sample_explanations if x]

        sample_explanations = '\n\n'.join(sample_explanations)
        if print_type == 'natural':
            sample_explanations = '.  '.join([x[:1].capitalize() + x[1:] for x in sample_explanations.split('.  ')])

        return sample_explanations

    def get_feature_importances(
            self,
            x: torch.Tensor,
            quantile: float = 1.,
            decision_boundary = None,
            feature_importances_type: str = ''
            ):
        _ = self.explain_samples(
            x,
            quantile=1. if feature_importances_type in ['weight', 'weight_proportion'] else quantile,
            threshold=None,
            target_names=[f'class_{i}' for i in range(self.output_size)],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix="Sample was in the",
            ignore_uninformative=False,
            rounding_precision=3,
            inverse_transform=None,
            decision_boundary=0. if feature_importances_type in ['weight', 'weight_proportion'] else decision_boundary,
            show_bounds=False,
            use_llm=False,
            simplify=False,
            exclusions=None,
            min_max_feature_dict=None,
            feature_importances=True,
            feature_importances_type=feature_importances_type)
        feature_importances_values = {key: sum(value) for key, value in self._feature_importances.items()}
        del self._feature_importances
        self._feature_importances = dict()
        return feature_importances_values

    @staticmethod
    def aggregate_feature_importances(feat_imp: dict):
        print(feat_imp)
        agg_feat_imp = dict()
        for key, value in feat_imp.items():
            if not " was" in key:
                continue
            agg_feat = key.split(" was")[0]
            if not agg_feat in agg_feat_imp:
                agg_feat_imp[agg_feat] = value
            else:
                agg_feat_imp[agg_feat] += value
        return agg_feat_imp

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
            decision_boundary: float = 0.5,
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
            inverse_transform (sklearn Transformation): Skleran inverse transformation function
            decision_boundary (float): Classification decision boundary
            show_bounds (bool): include numeric boundary used in logic
        """
        # binary classification?
        if self.output_size == 1:
            force_negate = required_output_thresholds < decision_boundary
            force_negate_str = f"negative of " if force_negate else ""
        else:
            force_negate = False
            force_negate_str = ""

        for output in range(self.output_size):
            print(f"REASONING NETWORK MODEL FOR: {force_negate_str}{target_names[output]}")
            self.root_layer.print(
                quantile=quantile,
                required_output_thresholds=required_output_thresholds,
                threshold=threshold,
                parent_weights=torch.ones(1),
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
            decision_boundary: float = 0.5,
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
            inverse_transform (sklearn Transformation): Skleran inverse transformation function
            decision_boundary (float): Classification decision boundary
            show_bounds (bool): include numeric boundary used in logic
        """
        x = x.to(self.root_layer.weights.device)
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

            if self.logits:
                predictions = torch.exp(predictions)/(torch.exp(predictions) + 1)

            # collect explanations
            parent_weights = torch.ones(1)
            if boosted_predictions is None:
                class_prediction = predictions[0].detach().cpu().argmax()
                negation_threshold = predictions[0, class_prediction].detach().cpu().squeeze()
            else:
                class_prediction = boosted_predictions[0].argmax()
                negation_threshold = boosted_predictions[0, class_prediction].squeeze()
            required_output_threshold = predictions[0, class_prediction].detach().cpu().squeeze()

            # binary classification?
            if self.output_size == 1:
                force_negate = negation_threshold < decision_boundary
                force_negate_str = f"negative of " if force_negate else ""
            else:
                force_negate = False
                force_negate_str = ""

            print(f"REASONING NETWORK MODEL FOR: {force_negate_str}{target_names[class_prediction]}")
            self.root_layer.print_sample(
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
                channel=class_prediction,
                force_negate=force_negate,
                ignore_uninformative=ignore_uninformative,
                rounding_precision=rounding_precision,
                inverse_transform=inverse_transform,
                show_bounds=show_bounds
            )

    def predict_proba(self, dl: DataLoader) -> Tuple[pd.DataFrame, pd.DataFrame]:
        predictions, all_targets = self.predict(dl)
        if self.logits:
            predictions.iloc[:, :] = predictions.iloc[:, :] = expit(predictions.values)
        return predictions, all_targets


__all__ = [ReasoningNetworkClassifierMixin]
