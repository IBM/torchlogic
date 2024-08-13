from copy import deepcopy

from collections import defaultdict
from typing import List, Tuple, Callable, Union
import logging

import numpy as np
import pandas as pd
import numpy.typing as npt
from scipy.special import softmax
from sklearn.utils import resample

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel

from torchlogic.modules import BanditNRNModule
from .pruningrn import PruningReasoningNetworkModel
from torchlogic.utils.operations import register_hooks


class BaseBanditNRNModel(PruningReasoningNetworkModel):

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
        """
        PruningReasoningNetworkModel.__init__(self)
        assert n_selected_features_input <= len(feature_names), \
            "`n_selected_features_input` must be <= number of features"
        assert n_selected_features_internal <= min(layer_sizes), \
            "`n_selected_features_internal` must be <= min(layer_sizes)"
        assert n_selected_features_output <= layer_sizes[-1], \
            "`n_selected_features_output` must be <= layer_sizes[-1]"
        if not add_negations:
            assert n_selected_features_input > 1, "`n_selected_features_input` must be greater than 1."
        assert n_selected_features_internal > 1, "`n_selected_features_internal` must be greater than 1."
        assert n_selected_features_output > 1, "`n_selected_features_output` must be greater than 1."

        self.target_names = target_names
        self.feature_names = feature_names
        self.input_size = input_size
        self.output_size = output_size
        self.n_selected_features_input = n_selected_features_input
        self.n_selected_features_internal = n_selected_features_internal
        self.n_selected_features_output = n_selected_features_output
        self.perform_prune_quantile = perform_prune_quantile
        self.ucb_scale = ucb_scale
        self.normal_form = normal_form
        self.delta = delta
        self.prune_strategy = prune_strategy
        self.bootstrap = bootstrap
        self.swa = swa
        self.add_negations = add_negations
        self.weight_init = weight_init
        self.logits = logits

        self.rn = BanditNRNModule(
            input_size=input_size,
            output_size=output_size,
            layer_sizes=layer_sizes,
            feature_names=feature_names,
            n_selected_features_input=n_selected_features_input,
            n_selected_features_internal=n_selected_features_internal,
            n_selected_features_output=n_selected_features_output,
            perform_prune_quantile=perform_prune_quantile,
            ucb_scale=ucb_scale,
            normal_form=normal_form,
            add_negations=add_negations,
            weight_init=weight_init,
            logits=logits
        )

        self.averaged_rn = None
        if self.swa:
            self.averaged_rn = AveragedModel(self.rn)

        self.target_names = target_names
        self.bootstrap_sample_map = {k: defaultdict(list) for k in range(self.output_size)}

        if torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs!")
            self.rn = nn.DataParallel(self.rn)
        if self.USE_CUDA:
            self.logger.info(f"Using GPU")
            self.rn = self.rn.cuda()
        elif self.USE_MPS:
            self.logger.info(f"Using MPS")
            self.rn = self.rn.to('mps')

        self.USE_DATA_PARALLEL = isinstance(self.rn, torch.nn.DataParallel)

        # initialize uniform policy
        self.predicates_policy = torch.empty((output_size, input_size)).uniform_(0, 1)
        if policy_init is not None:
            self.predicates_policy.copy_(policy_init)
        # self.register_buffer('predicates_policy', predicates_policy)
        self.predicates_rewards_history = self._initialize_rewards_history()

        self.logger = logging.getLogger(self.__class__.__name__)

    def _initialize_rewards_history(self) -> dict:
        """
        Initialize the reward histories for the multi-armed bandit.

        Returns:
            dict: dictionary of pd.DataFrame with rewards histories by class.
        """
        predicates_rewards_history = {}
        for output in range(self.output_size):
            predicates_rewards_history[output] = pd.DataFrame(
                {'predicates': range(self.input_size), 'rewards': self.predicates_policy[output].cpu().numpy()})
        return predicates_rewards_history

    def _get_activations(self, dl: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """"
        Get logic activations, targets, predictions and sample indexes used for 'logic' pruning strategy.

        Args:
            dl (DataLoader): PyTorch data loader with training data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                logic activations, targets, sample indexes
        """
        all_sample_idx = []
        all_activations = []
        all_targets = []
        activations = {}
        register_hooks(self, activations)
        for batch in dl:
            # [BATCH_SIZE, N_FEATURES]
            features = batch['features']
            # [BATCH_SIZE, N_TARGETS]
            target = batch['target']
            sample_idx = batch['sample_idx']

            if target.ndim > 2:
                target = target.squeeze(-1)

            if self.USE_CUDA:
                features = features.cuda()
                target = target.cuda()
            elif self.USE_MPS:
                features = features.to('mps')
                target = target.to('mps')

            # [BATCH_SIZE, N_TARGETS]
            if self.USE_DATA_PARALLEL:
                _ = self.rn.module.model[0](features)
            else:
                _ = self.rn.model[0](features)

            all_activations += [activations['0']]
            all_targets += [deepcopy(target)]
            all_sample_idx += [deepcopy(sample_idx)]

        all_activations = torch.cat(all_activations, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_sample_idx = torch.cat(all_sample_idx, dim=0)

        return all_activations, all_targets, all_sample_idx

    def _evaluate_activations(
            self,
            targets: torch.Tensor,
            activations: torch.Tensor,
            sample_idx: torch.Tensor,
            input_layer_weights: torch.Tensor,
            internal_layer_mask: torch.Tensor,
            mask: torch.Tensor,
            class_indices: torch.Tensor,
            output_metric: Callable,
            rewards: torch.Tensor = None,
            dimension: int = -2,
            mode: str = 'identify_important_features',
            objective: str = 'maximize'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate activations of logic to determine important features or logic to prune.

        Args:
            targets (torch.Tensor): targets for training data.
            activations (torch.Tensor): activations for input layer logic from forward pass on training data.
            sample_idx (torch.Tensor): sample indexes from forward pass on training data.
            input_layer_weights (torch.Tensor): Detached input layer weights from the model.
            internal_layer_mask (torch.Tensor): Mask of which input layer logics are used by the network.
            mask (torch.Tensor): mask to be modified by process.
            class_indices (list): List of integers indicating which class policies to update.
            output_metric (sklearn.metrics): Scikit-learn performance metric to use for evaluating logic.
            rewards (torch.Tensor): rewards if any.
            dimension (int): Dimension corresponding to logics from input_layer.
            mode (str): 'identify_important_features' or 'prune'.
            objective (str): 'maximize' or 'minimize'.  Which direction is the optimal performance for output_metric.

         Returns:
            Tuple[torch.Tensor, torch.Tensor]: boolean mask of important features or features to prune, rewards
        """
        sample_idx = sample_idx.cpu().numpy()
        targets = targets.cpu().numpy()
        activations = activations.cpu().numpy()

        train_size = targets.shape[0]
        indices = list(range(train_size))
        sample_idx_to_data_idx = {k: v for k, v in zip(sample_idx, indices)}

        for i in range(self.output_size):
            if not (i in class_indices):
                mask[i] = False
            else:

                performances = []
                for j in range(input_layer_weights.size(dimension)):

                    if torch.isin(torch.tensor(j), internal_layer_mask[i].cpu()):

                        if self.bootstrap:
                            # ensures that the boostrap samples are the same for each logic, every time
                            if j not in self.bootstrap_sample_map[i]:
                                bootstrap_sample_idx = resample(sample_idx, stratify=targets).tolist()
                                self.bootstrap_sample_map[i][j] = bootstrap_sample_idx
                            train_prune_idx = [sample_idx_to_data_idx[x] for x in self.bootstrap_sample_map[i][j]]
                        else:
                            train_prune_idx = indices

                        pos_performance = output_metric(
                            targets[train_prune_idx, i].reshape(-1, 1),
                            activations[train_prune_idx, i, 0, j].reshape(-1, 1))
                        neg_performance = output_metric(
                            targets[train_prune_idx, i].reshape(-1, 1),
                            activations[train_prune_idx, i, 0, j].reshape(-1, 1) * -1)

                        if pos_performance < 0 or neg_performance < 0:
                            raise AssertionError(
                                "Performance metric produced value less than zero. Must produce a positive value.")

                        if objective == 'maximize':
                            performances += [max(pos_performance, neg_performance)]
                        elif objective == 'minimize':
                            performances += [min(pos_performance, neg_performance)]
                        else:
                            raise ValueError("'objective' must be 'maximize' or 'minimize'.")
                    else:
                        if objective == 'maximize':
                            performances += [0]
                        elif objective == 'minimize':
                            performances += [100000]

                performances = np.array(performances)
                if objective == 'minimize' and any(performances > 0):
                    performances = np.log1p(np.clip(1 / performances + 1e-8, 0.0, 100000))

                performances_quantile_threshold = np.quantile(
                    performances, self.perform_prune_quantile)
                if len(performances) > 0:
                    for j in range(input_layer_weights.size(dimension)):
                        if torch.isin(torch.tensor(j), internal_layer_mask[i].cpu()):
                            if mode == 'identify_important_features':
                                if performances[j] >= performances_quantile_threshold:
                                    mask[i, j, :] = True
                                    if rewards is not None:
                                        rewards[i, j, :] = torch.tensor(performances[j]).repeat(rewards[i, j, :].size())
                            elif mode == 'prune':
                                if performances[j] < performances_quantile_threshold:
                                    mask[i, :, j] = True
                            else:
                                raise ValueError("'mode' must be 'identify_important_features' or 'prune'.")

        if mode == 'prune':
            mask = mask.bool().transpose(-2, -1)

        return mask, rewards

    def _evaluate_weights(
            self,
            input_layer_weights: torch.Tensor,
            class_indices: Union[List[int], torch.Tensor],
            mode: str = 'identify_important_features'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform weight evaluation to determine which logic to prune or which features are important.

        Args:
            input_layer_weights (torch.Tensor): Detached input layer weights from the model.
            class_indices (list): List of integers indicating which class policies to update.
            mode (str): 'identify_important_features' or 'prune'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: boolean mask of important features or features to prune, rewards
        """
        input_layer_weights_abs = torch.abs(input_layer_weights)

        if mode == 'identify_important_features':
            try:
                quantile_mask = torch.tensor(
                    input_layer_weights_abs >= input_layer_weights_abs.data.reshape(
                        self.output_size, -1).quantile(
                        self.perform_prune_quantile, dim=1,
                        keepdim=False).unsqueeze(-1).unsqueeze(-1).expand(input_layer_weights.size()))
            except NotImplementedError as e:
                quantile_mask = torch.Tensor(
                    (input_layer_weights_abs.to('cpu') >= input_layer_weights_abs.to('cpu').reshape(
                        self.output_size, -1).quantile(
                        self.perform_prune_quantile, dim=1,
                        keepdim=False).unsqueeze(-1).unsqueeze(-1).expand(input_layer_weights.size())).clone().detach())

        elif mode == 'prune':

            try:
                quantile_mask = torch.tensor(
                    input_layer_weights_abs <= input_layer_weights_abs.data.reshape(self.output_size, -1).quantile(
                        self.perform_prune_quantile, dim=1,
                        keepdim=False).unsqueeze(-1).unsqueeze(-1).expand(input_layer_weights_abs.size()))
            except NotImplementedError as e:
                quantile_mask = torch.Tensor(
                    (input_layer_weights_abs.to('cpu') <= input_layer_weights_abs.to('cpu').reshape(
                        self.output_size, -1).quantile(
                        self.perform_prune_quantile, dim=1,
                        keepdim=False).unsqueeze(-1).unsqueeze(-1).expand(
                        input_layer_weights_abs.size())).clone().detach())
        else:
            raise ValueError("'mode' must be 'identify_important_features' or 'prune'.")

        for i in range(self.output_size):
            if not (i in class_indices):
                quantile_mask[i] = False

        if mode == 'identify_important_features':
            prune_mask = quantile_mask
            rewards = input_layer_weights_abs
        elif mode == 'prune':
            prune_mask = quantile_mask.bool().transpose(-2, -1)
            rewards = None

        return prune_mask, rewards

    def _evaluate_logic_weights(
            self,
            input_layer_weights: torch.Tensor,
            internal_layer_weights: torch.Tensor,
            internal_layer_mask: torch.Tensor,
            class_indices: Union[List[int], torch.Tensor],
            mode: str = 'identify_important_features'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform weight evaluation to determine which logic to prune or which features are important.

        Args:
            input_layer_weights (torch.Tensor): Detached input layer weights from the model.
            class_indices (list): List of integers indicating which class policies to update.
            mode (str): 'identify_important_features' or 'prune'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: boolean mask of important features or features to prune, rewards
        """
        logic_layer_weights_abs = torch.abs(internal_layer_weights)

        if mode == 'identify_important_features':
            try:
                quantile_mask = torch.tensor(
                    logic_layer_weights_abs >= logic_layer_weights_abs.data.reshape(
                        self.output_size, -1).quantile(
                        self.perform_prune_quantile, dim=1,
                        keepdim=False).unsqueeze(-1).unsqueeze(-1).expand(internal_layer_weights.size()))
            except NotImplementedError as e:
                quantile_mask = torch.Tensor(
                    (logic_layer_weights_abs.to('cpu') >= logic_layer_weights_abs.to('cpu').reshape(
                        self.output_size, -1).quantile(
                        self.perform_prune_quantile, dim=1,
                        keepdim=False).unsqueeze(-1).unsqueeze(-1).expand(internal_layer_weights.size())).clone().detach())

        elif mode == 'prune':

            try:
                quantile_mask = torch.tensor(
                    logic_layer_weights_abs <= logic_layer_weights_abs.data.reshape(self.output_size, -1).quantile(
                        self.perform_prune_quantile, dim=1,
                        keepdim=False).unsqueeze(-1).unsqueeze(-1).expand(logic_layer_weights_abs.size()))
            except NotImplementedError as e:
                quantile_mask = torch.Tensor(
                    (logic_layer_weights_abs.to('cpu') <= logic_layer_weights_abs.to('cpu').reshape(
                        self.output_size, -1).quantile(
                        self.perform_prune_quantile, dim=1,
                        keepdim=False).unsqueeze(-1).unsqueeze(-1).expand(
                        logic_layer_weights_abs.size())).clone().detach())
        else:
            raise ValueError("'mode' must be 'identify_important_features' or 'prune'.")

        for i in range(self.output_size):
            if not (i in class_indices):
                quantile_mask[i] = False

        # is the logic used anywhere?
        # iterate over classes.  take the max weight and max quantile mask from anywhere it's used
        prune_mask = torch.zeros_like(input_layer_weights).bool()
        rewards = torch.zeros_like(input_layer_weights)
        for c in range(self.output_size):
            for i in range(internal_layer_mask.size(1)):
                for k, j in enumerate(internal_layer_mask[c, i, :]):
                    if prune_mask[c, :, j].max() < quantile_mask[c, i, k]:
                        prune_mask[c, :, j] = quantile_mask[c, i, k]
                    if (0 < rewards[c, :, j].max() < internal_layer_weights[c, i, k]) or rewards[c, :, j].max() == 0:
                        rewards[c, :, j] = internal_layer_weights[c, i, k]

        if mode == 'identify_important_features':
            prune_mask = prune_mask.transpose(-2, -1)
            rewards = rewards.transpose(-2, -1)
        elif mode == 'prune':
            prune_mask = prune_mask.transpose(-2, -1)
            rewards = None

        return prune_mask, rewards

    def _identify_important_features(
            self,
            dl: DataLoader = None,
            class_indices: List[int] = None,
            output_metric: Callable = None,
            objective: str = 'maximize'
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Identify important features.

        Args:
            dl (DataLoader): PyTorch data loader with training data.
            class_indices (list): List of integers indicating which class policies to update.
            output_metric (sklearn.metrics): Scikit-learn performance metric to use for evaluating logic.
            objective (str): 'maximize' or 'minimize'.  Which direction is the optimal performance for output_metric.

        Returns:
            Tuple[List[List[int]], List[List[float]]]: important feature indexes, important feature rewards
        """
        if self.USE_DATA_PARALLEL:
            input_layer_mask = self.rn.module.model[0].mask
            input_layer_weights = torch.abs(self.rn.module.model[0].weights.clone().detach().transpose(-2, -1))
            if len(self.rn.module.model) > 1:
                internal_layer_mask = self.rn.module.model[1].mask
                internal_layer_weights = torch.abs(self.rn.module.model[1].weights.clone().detach().transpose(-2, -1))
            else:
                internal_layer_mask = self.rn.module.output_layer.mask
                internal_layer_weights = torch.abs(self.rn.module.output_layer.weights.clone().detach().transpose(-2, -1))
        else:
            input_layer_mask = self.rn.model[0].mask
            input_layer_weights = torch.abs(self.rn.model[0].weights.clone().detach().transpose(-2, -1))
            if len(self.rn.model) > 1:
                internal_layer_mask = self.rn.model[1].mask
                internal_layer_weights = torch.abs(self.rn.model[1].weights.clone().detach().transpose(-2, -1))
            else:
                internal_layer_mask = self.rn.output_layer.mask
                internal_layer_weights = torch.abs(self.rn.output_layer.weights.clone().detach().transpose(-2, -1))

        # initialize class indices if not passed
        if class_indices is None:
            class_indices = torch.arange(self.output_size)
        else:
            class_indices = torch.tensor(class_indices)

        if self.prune_strategy == 'class':
            prune_mask, rewards = self._evaluate_weights(
                input_layer_weights=input_layer_weights,
                class_indices=class_indices,
                mode='identify_important_features'
            )
        elif self.prune_strategy == 'logic':
            # PRUNE RULES BASED ON PERFORMANCE OF EACH RULE
            prune_mask = torch.zeros_like(input_layer_weights.data).bool()
            rewards = torch.zeros_like(input_layer_weights.data)
            all_activations, all_targets, all_sample_idx = self._get_activations(dl)
            prune_mask, rewards = self._evaluate_activations(
                targets=all_targets,
                activations=all_activations,
                sample_idx=all_sample_idx,
                input_layer_weights=input_layer_weights,
                internal_layer_mask=internal_layer_mask,
                mask=prune_mask,
                class_indices=class_indices,
                output_metric=output_metric,
                rewards=rewards,
                dimension=-2,
                mode='identify_important_features',
                objective=objective
            )
        elif self.prune_strategy == 'logic_class':
            prune_mask, rewards = self._evaluate_logic_weights(
                input_layer_weights=input_layer_weights.transpose(-2, -1),
                internal_layer_weights=internal_layer_weights,
                internal_layer_mask=internal_layer_mask,
                class_indices=class_indices,
                mode='identify_important_features'
            )

        # select important features based on mask
        feature_indexes = []
        feature_weights = []
        for i in range(self.output_size):
            feature_indexes += [torch.masked_select(input_layer_mask[i], prune_mask[i]).cpu().numpy().tolist()]
            feature_weights += [torch.masked_select(rewards[i], prune_mask[i]).cpu().numpy().tolist()]

        return feature_indexes, feature_weights

    def _produce_predicate_rewards(
            self,
            feature_indexes: List[int],
            feature_weights: List[float]
    ) -> List[List[float]]:
        """
        Build predicate level rewards for each class.

        Args:
            feature_indexes (List[int]): Important feature indexes.
            feature_weights (List[float]): Important feature weights.

        Returns:
            npt.NDArray: Array of rewards for bandit for each class for all features.
        """
        predicates_total_rewards = []
        for p, v in zip(feature_indexes, feature_weights):
            predicates_rewards = defaultdict(float)
            for predicate, weight in zip(p, v):
                if (not isinstance(predicate, np.int64)
                        and not isinstance(predicate, np.int32)
                        and not isinstance(predicate, int)):
                    if len(predicate) > 1:
                        raise AssertionError("Predicate is not singular!")
                    predicate = predicate[0]
                predicates_rewards[predicate] += abs(weight)
            predicates_total_rewards += [predicates_rewards]

        # rewards for multi-armed bandit
        predicates_rewards = [[p[x] if x in p else 0.0 for x in range(self.input_size)]
                              for p in predicates_total_rewards]

        return predicates_rewards

    def _bayesian_ucb_policy(self, history_df: pd.DataFrame) -> npt.NDArray:
        """
        Apply Bayesian UCB policy to generate feature selection probabilities

        Args:
            history_df (pd.DataFrame): dataframe. Dataset to apply UCB policy to.
        """
        scores = history_df[['predicates', 'rewards']].groupby('predicates').agg(
            {'rewards': ['mean', 'count', 'std']})
        scores.columns = ['mean', 'count', 'std']
        scores['ucb'] = scores['mean'] + (self.ucb_scale * scores['std'] / np.sqrt(scores['count']))
        return softmax(scores.loc[range(self.input_size), 'ucb'].values)

    def _update_policy(
            self,
            feature_indexes: List[int],
            feature_weights: List[float],
            class_indices: List[int] = None
    ):
        """
        Perform policy update for each class.

        Args:
            feature_indexes (List[int]): Important feature indexes.
            feature_weights (List[float]): Important feature weights.
            class_indices (List[int]): Classes for which to perform a policy update.
        """
        predicates_rewards = self._produce_predicate_rewards(feature_indexes, feature_weights)

        if class_indices is None:
            class_indices = range(self.output_size)

        for output in class_indices:
            # features are selected using multi-armed bandit after initial step
            self.predicates_rewards_history[output] = pd.concat([
                self.predicates_rewards_history[output],
                pd.DataFrame({'predicates': range(self.input_size), 'rewards': predicates_rewards[output]}),
            ])
            predicates_probs = self._bayesian_ucb_policy(history_df=self.predicates_rewards_history[output])
            self.predicates_policy[output, :].copy_(torch.tensor(predicates_probs))

    @staticmethod
    def _flip_un_pruned_feature_weights(
            input_layer_mask: torch.Tensor,
            input_layer_weights: torch.Tensor,
            prune_mask: torch.Tensor,
            new_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Flip weights sign for newly generated features that are in the un-pruned set.  Used to model relations like
        x > feature_value > y.

        Args:
            input_layer_mask (torch.Tensor): Input layer feature mask with feature indexes used in logic.
            input_layer_weights (torch.Tensor): Detached input layer weights from the model.
            prune_mask (torch.Tensor): Boolean mask indicating which positions of the input_layer_mask will be pruned.
            new_mask (torch.Tensor): Newly input layer feature mask.  True positions in prune_mask will be replaced
                with these values

        Returns:
            torch.Tensor: new_input_layer_weights with flipped signs and reinitialized weights
        """
        # determine sign of weights for un-pruned features
        feature_indexes = torch.masked_select(input_layer_mask, ~prune_mask).cpu()
        feature_weights = torch.masked_select(
            input_layer_weights.transpose(-2, -1), ~prune_mask).cpu()
        # if the feature is used in more than one logic, we take the average of the weights
        feature_signs = pd.DataFrame(
            {'feature_weights': feature_weights.numpy(),
             'feature_indexes': feature_indexes.numpy()}
        )
        feature_signs = feature_signs.groupby(
            'feature_indexes')['feature_weights'].mean().to_frame().apply(
            lambda x: np.sign(x)).to_dict()['feature_weights']

        # mask on positions in newly generated feature masks that contain un-pruned features
        new_mask_existing_features_mask = torch.isin(new_mask.cpu(), feature_indexes.cpu())
        # mask on positions that contain un-pruned features and will be pruned this round
        new_mask_existing_features_mask_prune_mask = new_mask_existing_features_mask.cpu() * prune_mask.cpu()
        # reorient to new_mask
        new_mask_existing_features_mask_prune_mask = new_mask_existing_features_mask_prune_mask.transpose(-2, -1)

        # features at each position
        new_mask_weights_orientation = new_mask.clone().transpose(-2, -1)  # clone because we need new_mask later
        new_input_layer_weights = input_layer_weights.clone()  # clone because we need input_layer_weights.size() later

        # view all tensors as vectors
        new_mask_weights_orientation = new_mask_weights_orientation.contiguous().view(-1)
        new_input_layer_weights = new_input_layer_weights.contiguous().view(-1)
        new_mask_existing_features_mask_prune_mask = new_mask_existing_features_mask_prune_mask.contiguous().view(-1)

        # examine each element in vector
        for i, (feature_idx, weight, mask) \
                in enumerate(zip(new_mask_weights_orientation,
                                 new_input_layer_weights,
                                 new_mask_existing_features_mask_prune_mask)):
            # if the element is an un-pruned feature in the newly generated feature mask that will be pruned
            # then flip the sign of the new feature weight
            if mask:
                new_weights_sign = feature_signs[int(feature_idx.cpu().numpy())] * -1
                if new_weights_sign > 0:
                    # new_weight = torch.ones(1).uniform_(0, 0.1)
                    new_weight = torch.ones(1).uniform_(0, 0.2)
                    # new_weight = torch.nn.init.trunc_normal_(torch.ones(1), a=0, b=1)
                    # new_weight = torch.ones(1)
                else:
                    # new_weight = torch.ones(1).uniform_(-0.1, 0)
                    new_weight = torch.ones(1).uniform_(-0.2, 0)
                    # new_weight = torch.nn.init.trunc_normal_(torch.ones(1), a=-1, b=0)
                    # new_weight = torch.ones(1) * -1
                new_input_layer_weights.data[i] = new_weight

        # update the feature weights to use the flipped signs
        new_input_layer_weights = new_input_layer_weights.view(input_layer_weights.size())

        return new_input_layer_weights

    def perform_prune(
            self,
            dl: DataLoader = None,
            class_indices: List[int] = None,
            output_metric: Callable = None,
            objective: str = 'maximize'
    ):
        """
        Prune the network.

        Args:
            dl (DataLoader): PyTorch data loader with training data.
            class_indices (list): List of integers indicating which class policies to update.
            output_metric (sklearn.metrics): Scikit-learn performance metric to use for evaluating logic.
            objective (str): 'maximize' or 'minimize'.  Which direction is the optimal performance for output_metric.
        """
        if self.USE_DATA_PARALLEL:
            input_layer_mask = self.rn.module.model[0].mask
            input_layer_weights = self.rn.module.model[0].weights.clone().detach()
            if len(self.rn.module.model) > 1:
                internal_layer_mask = self.rn.module.model[1].mask
                internal_layer_weights = torch.abs(self.rn.module.model[1].weights.clone().detach().transpose(-2, -1))
            else:
                internal_layer_mask = self.rn.output_layer.mask
                internal_layer_weights = torch.abs(
                    self.rn.module.output_layer.weights.clone().detach().transpose(-2, -1))
        else:
            input_layer_mask = self.rn.model[0].mask
            input_layer_weights = self.rn.model[0].weights.clone().detach()
            if len(self.rn.model) > 1:
                internal_layer_mask = self.rn.model[1].mask
                internal_layer_weights = torch.abs(self.rn.model[1].weights.clone().detach().transpose(-2, -1))
            else:
                internal_layer_mask = self.rn.output_layer.mask
                internal_layer_weights = torch.abs(self.rn.output_layer.weights.clone().detach().transpose(-2, -1))

        # initialize class indices if not passed
        if class_indices is None:
            class_indices = torch.arange(self.output_size)
        else:
            class_indices = torch.tensor(class_indices)

        if self.prune_strategy == 'class':
            prune_mask, _ = self._evaluate_weights(
                input_layer_weights=input_layer_weights,
                class_indices=class_indices,
                mode='prune'
            )
        elif self.prune_strategy == 'logic':
            prune_mask = torch.zeros_like(input_layer_weights.data).bool()
            all_activations, all_targets, all_sample_idx = self._get_activations(dl)
            prune_mask, _ = self._evaluate_activations(
                targets=all_targets,
                activations=all_activations,
                sample_idx=all_sample_idx,
                input_layer_weights=input_layer_weights,
                internal_layer_mask=internal_layer_mask,
                mask=prune_mask,
                class_indices=class_indices,
                output_metric=output_metric,
                rewards=None,
                dimension=-1,
                mode='prune',
                objective=objective
            )
        elif self.prune_strategy == 'logic_class':
            prune_mask, _ = self._evaluate_logic_weights(
                input_layer_weights=input_layer_weights,
                internal_layer_weights=internal_layer_weights,
                internal_layer_mask=internal_layer_mask,
                class_indices=class_indices,
                mode='prune'
            )

        # # don't select used features in new logic
        predicates_policy_to_use = self.predicates_policy.clone()
        for i in range(self.output_size):
            feature_indexes = torch.LongTensor(
                [torch.masked_select(input_layer_mask[i], ~prune_mask[i]).cpu().numpy().tolist()])
            predicates_policy_to_use[i, feature_indexes] /= self.delta

        if not self.add_negations:
            # generate new input mask
            try:
                new_mask = torch.multinomial(
                    predicates_policy_to_use, input_layer_mask.size(1) * input_layer_mask.size(2)).reshape(
                    -1, input_layer_mask.size(1), input_layer_mask.size(2))
            except RuntimeError as e:
                new_mask = torch.multinomial(
                    predicates_policy_to_use, input_layer_mask.size(1) * input_layer_mask.size(2),
                    replacement=True).reshape(
                    -1, input_layer_mask.size(1), input_layer_mask.size(2))

            new_input_layer_weights = self._flip_un_pruned_feature_weights(
                input_layer_mask=input_layer_mask,
                input_layer_weights=input_layer_weights,
                prune_mask=prune_mask,
                new_mask=new_mask
            )
        else:
            # generate new input mask
            try:
                new_mask = torch.multinomial(
                    predicates_policy_to_use, input_layer_mask.size(1) * input_layer_mask.size(2) // 2).reshape(
                    -1, input_layer_mask.size(1), input_layer_mask.size(2) // 2)
                new_mask = torch.cat([new_mask, new_mask.clone()], dim=-1)
            except RuntimeError as e:
                new_mask = torch.multinomial(
                    predicates_policy_to_use, input_layer_mask.size(1) * input_layer_mask.size(2) // 2,
                    replacement=True).reshape(
                    -1, input_layer_mask.size(1), input_layer_mask.size(2) // 2)
                new_mask = torch.cat([new_mask, new_mask.clone()], dim=-1)

            size = (input_layer_weights.size(0), input_layer_weights.size(1) // 2, input_layer_weights.size(2))
            # new_input_layer_weights = torch.ones(size).uniform_(-0.1, 0.1)
            new_input_layer_weights = torch.ones(size).uniform_(-0.2, 0.2)
            # new_input_layer_weights = torch.nn.init.trunc_normal_(torch.ones(size), a=0, b=1)
            # new_input_layer_weights = torch.nn.init.xavier_uniform_(torch.ones(size), gain=torch.nn.init.calculate_gain('relu'))
            # new_input_layer_weights = torch.where(torch.zeros(size).uniform_(0, 1) > 0.5, torch.ones(size), torch.ones(size) * -1)
            new_input_layer_weights = torch.cat([new_input_layer_weights, new_input_layer_weights.clone() * -1], dim=1)

        # update mask
        try:
            input_layer_mask.masked_scatter_(
                prune_mask.to(input_layer_mask.device), new_mask.to(input_layer_mask.device))
        except NotImplementedError as e:
            input_layer_mask = input_layer_mask.to('cpu')
            input_layer_mask.masked_scatter_(
                prune_mask.to(input_layer_mask.device), new_mask.to(input_layer_mask.device))
            input_layer_mask.to(input_layer_weights.device)

        if self.USE_DATA_PARALLEL:
            self.rn.module.model[0].mask.copy_(input_layer_mask)
            self.rn.module.model[0].weights.data.copy_(new_input_layer_weights.to(input_layer_weights.device))
        else:
            self.rn.model[0].mask.copy_(input_layer_mask)
            self.rn.model[0].weights.data.copy_(new_input_layer_weights.to(input_layer_weights.device))

    def update_policy(
            self,
            dl: DataLoader,
            class_indices: List[int] = None,
            output_metric: Callable = None,
            objective='maximize'
    ):
        """
        Update the policy.

        Args:
            dl (DataLoader): PyTorch data loader with training data.
            class_indices (list): List of integers indicating which class policies to update.
            output_metric (sklearn.metrics): Scikit-learn performance metric to use for evaluating logic.
            objective (str): 'maximize' or 'minimize'.  Which direction is the optimal performance for output_metric.
        """
        if self.USE_DATA_PARALLEL:
            feature_indexes, feature_weights = self._identify_important_features(
                dl, class_indices, output_metric, objective)
        else:
            feature_indexes, feature_weights = self._identify_important_features(
                dl, class_indices, output_metric, objective)
        self._update_policy(feature_indexes, feature_weights, class_indices)


__all__ = [BaseBanditNRNModel]
