import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from minepy import cstats

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from torchlogic.models import BanditNRNRegressor
from torchlogic.utils.trainers import BanditNRNTrainer

from ..datasets.simple_dataset import SimpleDataset
from ..base.base_estimator import BaseSKLogicEstimator


class RNRNRegressor(BaseSKLogicEstimator):

    def __init__(
            self,
            target_name: list = None,
            feature_names: list = None,
            layer_sizes: list = [8, 8],
            n_selected_features_input: int = 4,
            n_selected_features_internal: int = 4,
            n_selected_features_output: int = 4,
            perform_prune_quantile: float = 0.5,
            ucb_scale: float = 1.5,
            prune_strategy: str = 'class',
            delta: float = 2.0,
            bootstrap: bool = False,
            swa: bool = False,
            add_negations: bool = False,
            normal_form: str = 'cnf',
            weight_init: float = 0.2,
            binarization: bool = True,
            tree_num: int = 10,
            tree_depth: int = 5,
            tree_feature_selection: float = 0.5,
            thresh_round: int = 3,
            loss_func=nn.MSELoss,
            learning_rate: float = 0.1,
            weight_decay: float = 0.001,
            t_0: int = 3,
            t_mult: int = 2,
            epochs: int = 200,
            batch_size: int = 32,
            holdout_pct: float = 0.2,
            early_stopping_plateau_count: int = 20,
            perform_prune_plateau_count: int = 3,
            increase_prune_plateau_count: int = 10,
            increase_prune_plateau_count_plateau_count: int = 10,
            lookahead_steps: int = 0,
            lookahead_steps_size: float = 0.0,
            evaluation_metric=mean_squared_error,
            pin_memory: bool = False,
            persistent_workers: bool = False,
            num_workers: int = 0
    ):
        """
        RNRN Scikit-Learn compatible regressor.

        Args:
            target_name (str): Names for targets
            feature_names List[str]: List of desired feature names
            layer_sizes List[int]: List of layer sizes corresponding to width of each layer
            n_selected_features_input (int): Number of input features selected in each input logic
            n_selected_features_internal (int): Number of hidden features selected in each hidden logic
            n_selected_features_output (int): Number of hidden features selected for each output logic
            perform_prune_quantile (float): Quantile of model to prune during pruning phase
            ucb_scale (float): UCB scale to use for multi-armed bandit
            prune_strategy (str): Prune strategy, one of 'class', 'logic', 'class_logic'
            delta (float): Factor used to reduce likelihood of sampling un-pruned logic during growth phase
            bootstrap (bool): If using logic prune strategy, evaluate with bootstrap sampling
            swa (bool): If true, use swa weight averaging
            add_negations (bool): If true, add negated logic on initialization
            normal_form (str): Conjunctive normal form or disjunctive normal form structure, 'cnf' or 'dnf'
            weight_init (float): Weigh initialization magnitude
            binarization (bool): if true, use feature binarization from trees
            tree_num (int): number of trees for feature binarization from trees
            tree_depth (int): depth of trees for feature binarization from trees
            tree_feature_selection (float): feature selection rate for feature binarization from trees
            thresh_round (int): rounding threshold in decimal places for feature binarzation from trees
            loss_func (Loss): PyTorch loss
            learning_rate (float): learning rate for AdamW optimizer
            weight_decay (float): weight decay for AdamW optimizer
            t_0 (int): T_0 for CosineAnnealingWithWarmRestarts scheduler
            t_mult (int): T_mult for CosineAnnealingWithWarmRestarts scheduler
            epochs (int): epochs for training procedure
            batch_size (int): batch size for data loading
            holdout_pct (float): percentage of training data used as holdout for early stopping
            early_stopping_plateau_count (int): number of epochs without improvement for early stopping
            perform_prune_plateau_count (int): Perform pruning phase after plateau count epochs without improvement
            increase_prune_plateau_count (int): Increase the perform_prune_plateau_count by this number if plateaued
            increase_prune_plateau_count_plateau_count (int): Increase perform_prune_plateau_count when reached .
            lookahead_steps (int): number of steps for lookahead optimization. If zero, lookahead not used.
            lookahead_steps_size (float): step size for lookahead optimization.  If lookahead_steps is zero not used.
            evaluation_metric (metric): Scikit-Learn metric function
            pin_memory (bool): pin_memory for data loaders.
            persistent_workers (bool): persistent workers for data loaders.
            num_workers (int): number of workers for data loaders.
        """
        super(RNRNRegressor, self).__init__(
            binarization=binarization,
            tree_num=tree_num,
            tree_depth=tree_depth,
            tree_feature_selection=tree_feature_selection,
            thresh_round=thresh_round,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            t_0=t_0,
            t_mult=t_mult,
            epochs=epochs,
            batch_size=batch_size,
            holdout_pct=holdout_pct,
            early_stopping_plateau_count=early_stopping_plateau_count,
            lookahead_steps=lookahead_steps,
            lookahead_steps_size=lookahead_steps_size,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            num_workers=num_workers
        )

        # handle empty target names
        if target_name:
            self.target_name = target_name
        else:
            self.target_name = None

        # handle empty feature names
        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = None

        # add hyper-parameters
        self.layer_sizes = layer_sizes
        self.n_selected_features_input = n_selected_features_input
        self.n_selected_features_internal = n_selected_features_internal
        self.n_selected_features_output = n_selected_features_output
        self.perform_prune_quantile = perform_prune_quantile
        self.ucb_scale = ucb_scale
        self.prune_strategy = prune_strategy
        self.delta = delta
        self.bootstrap = bootstrap
        self.swa = swa
        self.add_negations = add_negations
        self.normal_form = normal_form
        self.weight_init = weight_init
        self.loss_func = loss_func
        self.perform_prune_plateau_count = perform_prune_plateau_count
        self.increase_prune_plateau_count = increase_prune_plateau_count
        self.increase_prune_plateau_count_plateau_count = increase_prune_plateau_count_plateau_count
        self.evaluation_metric = evaluation_metric

        # data encoding
        self.mms = MinMaxScaler()

    def _fit_transform_encode_target(self, y: pd.DataFrame):
        return pd.DataFrame(self.mms.fit_transform(y), columns=[self.target_name], dtype=np.float32)

    def _transform_encode_target(self, y: pd.DataFrame):
        return pd.DataFrame(self.mms.transform(y), columns=[self.target_name])

    def _inverse_transform_encode_target(self, y: pd.DataFrame):
        return pd.DataFrame(self.mms.inverse_transform(y), columns=[self.target_name])

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Fit the model

        Args:
            X (pd.DataFrame): Dataframe of features
            y (pd.DataFrame): Dataframe of targets

        Returns:
            None
        """
        if not isinstance(X, pd.DataFrame):
            X = self._handle_non_dataframe_features(X)
        X = X.copy()

        if not isinstance(y, pd.DataFrame):
            y = self._handle_non_dataframe_targets(y)
        y = y.copy()

        X = self._handle_empty_feature_names(X)

        # create textual target name if it is are not given by user
        if self.target_name is None:
            self.target_name = 'Target'
            y.columns = [self.target_name]

        # ecode data in necessary format
        X = self._fit_transform_encode_data(X)
        if self.binarization:
            X = self._fit_transform_binarize_features(X, y > y.mean())
        feature_names = X.columns
        y = self._fit_transform_encode_target(y)

        # pytorch data
        dataset = SimpleDataset(X.values, y.values)
        train_dl, train_holdout_dl = self._generate_training_data_loaders(dataset)

        # initial bandit policy
        assert len(feature_names) == X.shape[1], f"feat names: {len(feature_names)}; x: {X.shape[1]}"
        mic_c_policy, _ = cstats(X.T, y.T, alpha=9, c=5, est="mic_e")
        mic_c_policy = torch.tensor(mic_c_policy.T)

        if X.shape[1] < self.n_selected_features_input:
            Warning(
                "The number of features is less than 'n_selected_features_input'.  Using number of features instead.")
            self.n_selected_features_input = X.shape[1]

        # init model
        self.model = BanditNRNRegressor(
            target_names=self.target_name,
            feature_names=list(feature_names),
            input_size=len(feature_names),
            layer_sizes=self.layer_sizes,
            n_selected_features_input=self.n_selected_features_input,
            n_selected_features_internal=self.n_selected_features_internal,
            n_selected_features_output=self.n_selected_features_output,
            perform_prune_quantile=self.perform_prune_quantile,
            ucb_scale=self.ucb_scale,
            prune_strategy=self.prune_strategy,
            normal_form=self.normal_form,
            delta=self.delta,
            bootstrap=self.bootstrap,
            swa=self.swa,
            add_negations=self.add_negations,
            weight_init=self.weight_init,
            policy_init=mic_c_policy
        )

        optimizer = optim.AdamW(self.model.rn.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.t_0, T_mult=self.t_mult)
        trainer = BanditNRNTrainer(
            model=self.model,
            loss_func=self.loss_func(),
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=self.epochs,
            accumulation_steps=1,
            l1_lambda=0.,
            early_stopping_plateau_count=self.early_stopping_plateau_count,
            perform_prune_plateau_count=self.perform_prune_plateau_count,
            increase_prune_plateau_count=self.increase_prune_plateau_count,
            increase_prune_plateau_count_plateau_count=self.increase_prune_plateau_count_plateau_count,
            lookahead_steps=self.lookahead_steps,
            lookahead_steps_size=self.lookahead_steps_size,
            augment=None,
            augment_alpha=0.
        )

        # train model
        # The trainer defaults to optimizing the validation roc_auc_score.  To optimize
        # against a different metric pass the sklearn metric to the 'evaluation_metric' parameter
        trainer.train(train_dl, train_holdout_dl, evaluation_metric=self.evaluation_metric, multi_class=False)
        trainer.set_best_state()

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict classes with fitted model on new data.

        Args:
            X (pd.DataFrame): Untransformed input features

        Returns:
            pd.DataFrame: Predictions from fitted model
        """
        assert self.model is not None, "must fit before prediction"

        if not isinstance(X, pd.DataFrame):
            X = self._handle_non_dataframe_features(X)
        X = X.copy()

        X = self._handle_empty_feature_names(X)

        X = self._encode_prediction_data(X)
        dataset = SimpleDataset(X.values, np.ones(shape=(X.shape[0], 1)))
        prediction_dl = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, num_workers=self.num_workers
            # very important to optimize these settings in production
        )

        predictions, _ = self.model.predict(prediction_dl)

        return self._inverse_transform_encode_target(predictions)

    def score(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        Score a model with new data using the evaluation metric

        Args:
            X (pd.DataFrame): Untransformed input features
            y (pd.DataFrame): target input

        Returns:
            float: score from evaluation metric
        """
        assert self.model is not None, "must fit before scoring"

        if not isinstance(X, pd.DataFrame):
            X = self._handle_non_dataframe_features(X)
        X = X.copy()

        if not isinstance(y, pd.DataFrame):
            y = self._handle_non_dataframe_targets(y)
        y = y.copy()

        X = self._handle_empty_feature_names(X)

        # create textual target name if it is are not given by user
        y.columns = [self.target_name]

        X = self._encode_prediction_data(X)

        dataset = SimpleDataset(X.values, y.values)

        prediction_dl = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, num_workers=self.num_workers
            # very important to optimize these settings in production
        )

        predictions, _ = self.model.predict(prediction_dl)
        predictions = self.mms.inverse_transform(predictions)

        return self.evaluation_metric(y, predictions)

    def _get_instance_params(self):
        """
        Get parameters for this model type.

        Returns:
            dict: parameters and values
        """
        return {
            'layer_sizes': self.layer_sizes,
            'n_selected_features_input': self.n_selected_features_input,
            'n_selected_features_internal': self.n_selected_features_internal,
            'n_selected_features_output': self.n_selected_features_output,
            'perform_prune_quantile': self.perform_prune_quantile,
            'ucb_scale': self.ucb_scale,
            'prune_strategy': self.prune_strategy,
            'delta': self.delta,
            'bootstrap': self.bootstrap,
            'swa': self.swa,
            'add_negations': self.add_negations,
            'weight_init': self.weight_init,
            'loss_func': self.loss_func,
            'perform_prune_plateau_count': self.perform_prune_plateau_count,
            'increase_prune_plateau_count': self.increase_prune_plateau_count,
            'increase_prune_plateau_count_plateau_count': self.increase_prune_plateau_count_plateau_count,
            'evaluation_metric': self.evaluation_metric,
        }

    def explain_sample(
            self,
            X: pd.DataFrame,
            sample_index: int = 0,
            quantile: float = 1.0
    ) -> str:
        """
        Generate a sample explanation

        Args:
            X (pd.DataFrame): DataFrame of input features.
            sample_index (int): Index of sample to explain.
            quantile (float): Percent of model to explain

        Returns:
            str: explanation for selected sample
        """
        # create textual feature names if they are not given by user
        if self.feature_names is None:
            self.feature_names = [f"the {x} was" for x in X.columns]
            X.columns = self.feature_names
        else:
            X.columns = self.feature_names

        X = self._encode_prediction_data(X)

        self.min_max_features_dict = {
            col: {'min': self.numeric_features.iloc[:, i].min(), 'max': self.numeric_features.iloc[:, i].max()}
            for i, col in enumerate(self.numeric_features.columns)
        }

        dataset = SimpleDataset(X.values, np.ones(shape=(X.shape[0], 1)))

        return self.model.explain_samples(
            dataset[sample_index]['features'].unsqueeze(0),
            quantile=quantile,
            target_names=[self.target_name],
            explain_type='both',
            sample_explanation_prefix="The sample has",
            print_type='logical',
            ignore_uninformative=True,
            rounding_precision=3,
            show_bounds=not self.binarization,
            simplify=True,
            exclusions=None,
            inverse_transform_target=self.mms.inverse_transform
        )
