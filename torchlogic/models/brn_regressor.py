import logging
from typing import List

import torch

from ..models.base import BoostedBanditNRNModel
from .mixins import ReasoningNetworkRegressorMixin


class BanditNRNRegressor(BoostedBanditNRNModel, ReasoningNetworkRegressorMixin):

    def __init__(
            self,
            target_names: str,
            feature_names: List[str],
            input_size: int,
            layer_sizes: List[int],
            n_selected_features_input: int,
            n_selected_features_internal: int,
            n_selected_features_output: int,
            perform_prune_quantile: float,
            ucb_scale: float,
            normal_form: str = 'dnf',
            delta: float = 2.0,
            prune_strategy: str = 'class',
            bootstrap: bool = True,
            swa: bool = False,
            add_negations: bool = False,
            weight_init: float = 0.2,
            policy_init: torch.Tensor = None,
            xgb_max_depth: int = 4,
            xgb_n_estimators: int = 200,
            xgb_min_child_weight: float = 1.0,
            xgb_subsample: float = 0.7,
            xgb_learning_rate: float = 0.001,
            xgb_colsample_bylevel: float = 0.7,
            xgb_colsample_bytree: float = 0.7,
            xgb_gamma: float = 1,
            xgb_reg_lambda: float = 2.0,
            xgb_reg_alpha: float = 2.0
    ):
        """
        Initialize a BERrnRegressor model.

        Example:
            model = BERrnRegressor(
                target_names='metric1',
                feature_names=['feature1', 'feature2', 'feature3'],
                input_size=3,
                layer_sizes=[3, 3]
                n_selected_features_input=2,
                n_selected_features_internal=2,
                n_selected_features_output=1,
                ucb_scale=1.96,
                perform_prune_quantile=0.7,
                normal_form='dnf',
                prune_strategy='class',
                distributed=False
            )

        Args:
            target_names (list): A list of the target names.
            feature_names (list): A list of feature names.
            input_size (int): number of features from input.
            layer_sizes (list): A list containing the number of output logics for each layer.
            n_selected_features_input (int): The number of features to include in each logic in the input layer.
            n_selected_features_internal (int): The number of logics to include in each logic in the internal layers.
            n_selected_features_output (int): The number of logics to include in each logic in the output layer.
            perform_prune_quantile (float): The quantile to use for pruning randomized rn.
            ucb_scale (float): The scale of the confidence interval in the multi-armed bandit policy.
                               c = 1.96 is a 95% confidence interval.
            normal_form (str): 'dnf' for disjunctive normal form network; 'cnf' for conjunctive normal form network.
            delta (float): higher values increase diversity of logic generation away from existing logics.
            prune_strategy(str): Either 'class' or 'logic'.  Determines which pruning strategy to use.
            bootstrap (bool): Use boostrap samples to evaluate each atomic logic in logic prune strategy.
            swa (bool): Use stochastic weight averaging
            add_negations (bool): add negations of logic.
            weight_init (float): Upper bound of uniform weight initialization.  Lower bound is negated value.
            xgb_max_depth (int): Max depth for XGBoost boosting model.
            xgb_n_estimators (int): Number of estimators for XGBoost boosting model.
            xgb_min_child_weight (float): Minimum child weight for XGBoost boosting model.
            xgb_subsample (float): Subsample percentage for XGBoost boosting model.
            xgb_learning_rate (float): Learning rate for XGBoost boosting model.
            xgb_colsample_bylevel (float): Column subsample percent for XGBoost boosting model.
            xgb_colsample_bytree (float): Tree subsample percent for XGBoost boosting model.
            xgb_gamma (float): Gamma parameter for XGBoost boosting model.
            xgb_reg_lambda (float): Lambda regularization parameter for XGBoost boosting model.
            xgb_reg_alpha (float): Alpha regularization parameter for XGBoost boosting model.
        """
        ReasoningNetworkRegressorMixin.__init__(self, output_size=1)
        BoostedBanditNRNModel.__init__(
            self,
            target_names=[target_names],
            input_size=input_size,
            output_size=1,
            layer_sizes=layer_sizes,
            feature_names=feature_names,
            n_selected_features_input=n_selected_features_input,
            n_selected_features_internal=n_selected_features_internal,
            n_selected_features_output=n_selected_features_output,
            perform_prune_quantile=perform_prune_quantile,
            ucb_scale=ucb_scale,
            normal_form=normal_form,
            delta=delta,
            prune_strategy=prune_strategy,
            bootstrap=bootstrap,
            swa=swa,
            add_negations=add_negations,
            weight_init=weight_init,
            policy_init=policy_init,
            xgb_max_depth=xgb_max_depth,
            xgb_n_estimators=xgb_n_estimators,
            xgb_min_child_weight=xgb_min_child_weight,
            xgb_subsample=xgb_subsample,
            xgb_learning_rate=xgb_learning_rate,
            xgb_colsample_bylevel=xgb_colsample_bylevel,
            xgb_colsample_bytree=xgb_colsample_bytree,
            xgb_gamma=xgb_gamma,
            xgb_reg_lambda=xgb_reg_lambda,
            xgb_reg_alpha=xgb_reg_alpha,
            logits=False
        )
        self.set_modules(self.rn)
        self.logger = logging.getLogger(self.__class__.__name__)


__all__ = [BanditNRNRegressor]
