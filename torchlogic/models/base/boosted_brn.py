from copy import deepcopy
from typing import List, Tuple

import pandas as pd

import torch
from torch.utils.data import DataLoader

from .brn import BaseBanditNRNModel
from xgboost import XGBRegressor


class BoostedBanditNRNModel(BaseBanditNRNModel):

    def __init__(
            self,
            target_names: List[str],
            feature_names: List[str],
            input_size: int,
            output_size: int,
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
            xgb_reg_alpha: float = 2.0,
            logits: bool = False
    ):
        """
        Initialize a Bandit Reinforced Reasoning Network model.

        Example:
            model = BaseBanditRRNModel(
                target_names=['class1', 'class2'],
                feature_names=['feature1', 'feature2', 'feature3'],
                input_size=3,
                output_size=2,
                layer_sizes=[3, 3]
                n_selected_features_input=2,
                n_selected_features_internal=2,
                n_selected_features_output=1,
                ucb_scale=1.96,
                perform_prune_quantile=0.7,
                normal_form='dnf',
                prune_strategy='class',
                bootstrap=False,
                swa=False
            )

        Args:
            target_names (list): A list of the target names.
            feature_names (list): A list of feature names.
            input_size (int): number of features from input.
            output_size (int): number of outputs.
            layer_sizes (list): A list containing the number of output logics for each layer.
            n_selected_features_input (int): The number of features to include in each logic in the input layer.
            n_selected_features_internal (int): The number of logics to include in each logic in the internal layers.
            n_selected_features_output (int): The number of logics to include in each logic in the output layer.
            perform_prune_quantile (float): The quantile to use for pruning randomized rn.
            ucb_scale (float): The scale of the confidence interval in the multi-armed bandit policy.
                               c = 1.96 is a 95% confidence interval.
            normal_form (str): 'dnf' for disjunctive normal form network; 'cnf' for conjunctive normal form network.
            delta (float): higher values increase diversity of logic generation away from existing logics.
            prune_strategy (str): Either 'class' or 'logic'.  Determines which pruning strategy to use.
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
        super(BoostedBanditNRNModel, self).__init__(
            target_names=target_names,
            feature_names=feature_names,
            input_size=input_size,
            output_size=output_size,
            layer_sizes=layer_sizes,
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
            logits=logits
        )
        self.xgb_is_fitted = False
        self.initialize_booster(
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
        )

    def predict(self, dl: DataLoader) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predict for data given by features

        Args:
            dl (DataLoader): data loader for predictions

        Returns:
            (DataFrame, DataFrame): data frame of predictions, data frame of targets
        """
        all_predictions = []
        all_targets = []
        all_indexes = []
        with torch.no_grad():
            for batch in dl:
                # [BATCH_SIZE, N_FEATURES, 2]
                features = batch['features']
                if 'target' in batch:
                    # [BATCH_SIZE, N_TARGETS]
                    target = batch['target']
                    if target.ndim > 2:
                        target.squeeze()
                    all_targets += [deepcopy(target)]

                if self.USE_CUDA:
                    features = features.cuda()
                elif self.USE_MPS:
                    features = features.to('mps')

                # [BATCH_SIZE, N_TARGETS]
                rn_predictions = self.rn(features)
                if self.xgb_is_fitted:
                    xgb_predictions = self.xgb.predict(features.cpu())
                    xgb_predictions = torch.tensor(xgb_predictions).to(rn_predictions.device)
                    if xgb_predictions.ndim < 2:
                        xgb_predictions = xgb_predictions.unsqueeze(1)
                    rn_predictions = rn_predictions + xgb_predictions
                all_predictions += [rn_predictions]
                # [BATCH_SIZE]
                all_indexes += [deepcopy(batch['sample_idx'])]

            all_predictions = torch.cat(all_predictions, dim=0).cpu().detach()
            all_indexes = torch.cat(all_indexes, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

        all_predictions = all_predictions.numpy()
        all_indexes = all_indexes.numpy()
        all_targets = all_targets.numpy()

        predictions = pd.DataFrame(
            data=all_predictions,
            index=all_indexes,
            columns=[f"probs_{x.replace('_label', '')}" for x in self.target_names]
        )
        del all_predictions

        all_targets = pd.DataFrame(
            data=all_targets,
            index=all_indexes,
            columns=self.target_names
        )

        return predictions, all_targets

    def initialize_booster(
            self,
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
        self.xgb_max_depth = xgb_max_depth
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_min_child_weight = xgb_min_child_weight
        self.xgb_subsample = xgb_subsample
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_colsample_bylevel = xgb_colsample_bylevel
        self.xgb_colsample_bytree = xgb_colsample_bytree
        self.xgb_gamma = xgb_gamma
        self.xgb_reg_lambda = xgb_reg_lambda
        self.xgb_reg_alpha = xgb_reg_alpha

        self.xgb = XGBRegressor(
            learning_rate=self.xgb_learning_rate,
            gamma=self.xgb_gamma,
            max_depth=self.xgb_max_depth,
            n_estimators=self.xgb_n_estimators,
            min_child_weight=self.xgb_min_child_weight,
            subsample=self.xgb_subsample,
            colsample_bylevel=self.xgb_colsample_bylevel,
            colsample_bytree=self.xgb_colsample_bytree,
            reg_lambda=self.xgb_reg_lambda,
            reg_alpha=self.xgb_reg_alpha,
            # tree_method="hist" if self.USE_CUDA else "auto",
            # device="cuda" if self.USE_CUDA else "cpu"
        )
        self.xgb_is_fitted = False


__all__ = [BoostedBanditNRNModel]
