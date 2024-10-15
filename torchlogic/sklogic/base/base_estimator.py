from typing import Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from aix360.algorithms.rbm import FeatureBinarizerFromTrees


class BaseSKLogicEstimator:

    def __init__(
            self,
            binarization: bool = True,
            tree_num: int = 10,
            tree_depth: int = 5,
            tree_feature_selection: float = 0.5,
            thresh_round: int = 3,
            learning_rate: float = 0.1,
            weight_decay: float = 0.001,
            t_0: int = 3,
            t_mult: int = 2,
            epochs: int = 200,
            batch_size: int = 32,
            holdout_pct: float = 0.2,
            early_stopping_plateau_count: int = 20,
            lookahead_steps: int = 0,
            lookahead_steps_size: float = 0.0,
            pin_memory: bool = False,
            persistent_workers: bool = False,
            num_workers: int = 0
    ):
        """
        Base Scikit-Learn torchlogic compatible estimator.

        Args:
            binarization (bool): if true, use feature binarization from trees
            tree_num (int): number of trees for feature binarization from trees
            tree_depth (int): depth of trees for feature binarization from trees
            tree_feature_selection (float): feature selection rate for feature binarization from trees
            thresh_round (int): rounding threshold in decimal places for feature binarzation from trees
            learning_rate (float): learning rate for AdamW optimizer
            weight_decay (float): weight decay for AdamW optimizer
            t_0 (int): T_0 for CosineAnnealingWithWarmRestarts scheduler
            t_mult (int): T_mult for CosineAnnealingWithWarmRestarts scheduler
            epochs (int): epochs for training procedure
            batch_size (int): batch size for data loading
            holdout_pct (float): percentage of training data used as holdout for early stopping
            early_stopping_plateau_count (int): number of epochs without improvement for early stopping
            lookahead_steps (int): number of steps for lookahead optimization. If zero, lookahead not used.
            lookahead_steps_size (float): step size for lookahead optimization.  If lookahead_steps is zero not used.
            pin_memory (bool): pin_memory for data loaders.
            persistent_workers (bool): persistent workers for data loaders.
            num_workers (int): number of workers for data loaders.
        """
        # add hyper-parameters
        self.binarization = binarization
        self.tree_num = tree_num
        self.tree_depth = tree_depth
        self.tree_feature_selection = tree_feature_selection
        self.thresh_round = thresh_round
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.t_0 = t_0
        self.t_mult = t_mult
        self.epochs = epochs
        self.batch_size = batch_size
        self.holdout_pct = holdout_pct
        self.early_stopping_plateau_count = early_stopping_plateau_count
        self.lookahead_steps = lookahead_steps
        self.lookahead_steps_size = lookahead_steps_size
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers

        self.feature_names = None

        # data encoding
        self.mixed_pipe = None
        self.numeric_features = None
        self.categorical_features = None
        self._fbt_is_fitted = False
        self.min_max_features_dict = None

        # create feature binarizer
        if self.binarization:
            self.fbt = FeatureBinarizerFromTrees(
                treeNum=self.tree_num,
                treeDepth=self.tree_depth,
                treeFeatureSelection=self.tree_feature_selection,
                threshRound=self.thresh_round,
                randomState=0
            )

        self.model = None

    def _create_holdout_samplers(
            self,
            train_dataset: Dataset
    ) -> Tuple[SubsetRandomSampler, SubsetRandomSampler]:
        """
        Create holdout data sampler for training.

        Args:
            train_dataset (Dataset): the training dataset

        Returns:
            Tuple[SubsetRandomSampler, SubsetRandomSampler]: training data sampler, holdout data sampler
        """
        assert self.holdout_pct > 0 and self.holdout_pct < 1, "holdout_pct must be between 0 and 1"
        g = torch.Generator()
        g.manual_seed(0)

        train_size = len(train_dataset)
        indices = list(range(train_size))
        np.random.seed(0)
        np.random.shuffle(indices)

        train_holdout_split_index = int(np.floor(self.holdout_pct * train_size))
        train_idx, train_holdout_idx = indices[train_holdout_split_index:], indices[:train_holdout_split_index]

        train_sampler = SubsetRandomSampler(train_idx)
        train_holdout_sampler = SubsetRandomSampler(train_holdout_idx)

        return train_sampler, train_holdout_sampler

    def _generate_training_data_loaders(self, dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
        """
        Generate the training data loaders.

        Args:
            dataset (Dataset): the training data pytorch dataset

        Returns:
            Tuple[DataLoader, DataLoader]: the training data loader
        """
        g = torch.Generator()
        g.manual_seed(0)

        train_sampler, train_holdout_sampler = self._create_holdout_samplers(dataset)

        train_dataloader = DataLoader(
            dataset, batch_size=self.batch_size, generator=g, sampler=train_sampler,
            pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, num_workers=self.num_workers
            # very important to optimize these settings in production
        )
        train_holdout_dataloader = DataLoader(
            dataset, batch_size=self.batch_size, generator=g, sampler=train_holdout_sampler,
            pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, num_workers=self.num_workers
            # very important to optimize these settings in production
        )

        return train_dataloader, train_holdout_dataloader

    def _handle_non_binarized_categorical_feature_names(self, X: pd.DataFrame):
        if not self.binarization:
            new_cols = []
            for col in X.columns:
                if col.find("was_") > -1:
                    col = col.replace("was_", " ") + " was"
                new_cols += [col]
            X.columns = new_cols
        return X

    def _fit_transform_encode_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the features for the model.

        Args:
            X (pd.DataFrame): DataFrame of features

        Returns:
            pd.DataFrame: Transformed DataFrame of features
        """
        self.numeric_features = X.select_dtypes(include='number')
        self.categorical_features = X.select_dtypes(exclude='number')
        numeric_feature_names = list(self.numeric_features.columns)
        categorical_feature_names = list(self.categorical_features.columns)

        if not self.binarization:
            mixed_encoded_preprocessor = ColumnTransformer(
                [
                    (
                        "numeric",
                        MinMaxScaler(),
                        numeric_feature_names
                    ),
                    (
                        "categorical",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.bool_),
                        categorical_feature_names,
                    )
                ],
                verbose_feature_names_out=False,
            )
        else:
            mixed_encoded_preprocessor = ColumnTransformer(
                [
                    (
                        "numeric",
                        "passthrough",
                        numeric_feature_names
                    ),
                    (
                        "categorical",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.bool_),
                        categorical_feature_names,
                    )
                ],
                verbose_feature_names_out=False,
            )

        mixed_encoded_preprocessor.set_output(transform='pandas')

        self.mixed_pipe = make_pipeline(
            mixed_encoded_preprocessor,
        )

        X = self.mixed_pipe.fit_transform(X)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not self.binarization:
            self.numeric_features = X[numeric_feature_names].copy()

        X = self._handle_non_binarized_categorical_feature_names(X)

        return X

    def _transform_encode_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted transformation of features to new data.

        Args:
            X (pd.DataFrame): DataFrame of features.

        Returns:
            pd.DataFrame: Transformed DataFrame of features
        """
        X = self.mixed_pipe.transform(X)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = self._handle_non_binarized_categorical_feature_names(X)

        return X

    @staticmethod
    def _transform_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the column names of encoded features for use in explanations.

        Args:
            df (pd.DataFrame): DataFrame of transformed data.

        Returns:
            pd.DataFrame: DataFrame of transformed data with column names needed for explanations.
        """
        df.columns = list(map(
            lambda x: str(x).replace("(", "").replace(")", "").replace(" ", "_")
            .replace("/", "_").replace(",", "").replace("'", "")
            .replace("_", " ").replace(">=", "greater than or equal to")
            .replace("<=", "less than or equal to").replace(">", "greater than")
            .replace("<", "less than").replace("cat  ", "")
            .replace("num  ", ""),
            df.columns.to_flat_index()))
        return df

    def initialize_binarizer(self, fitted_binarizer: FeatureBinarizerFromTrees) -> None:
        """
        Initialize the feature binarizer from a previously fitted binarizer.

        Args:
            fitted_binarizer (FeatureBinarizerFromTrees): Fitted FeatureBinarizerFromTrees object

        Returns:
            None
        """
        self.fbt = fitted_binarizer
        self._fbt_is_fitted = True

    def _fit_transform_binarize_features(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform input data with FeatureBinarizationFromTrees

        Args:
            X (pd.DataFrame): DataFrame of transformed features.
            y (pd.DataFrame): DataFrame of targets.

        Returns:
            pd.DataFrame: Binarized features.
        """
        X_numeric = X.select_dtypes(include='number')
        X_categorical = X.select_dtypes(exclude='number')
        numeric_columns = list(X_numeric.columns)
        categorical_columns = list(X_categorical.columns)

        if not self._fbt_is_fitted:
            X_numeric = self.fbt.fit_transform(X_numeric, y)
        else:
            X_numeric = self.fbt.transform(X_numeric)
        X_numeric = self._transform_column_names(X_numeric)

        feature_names = list(X_numeric.columns) + list(categorical_columns)
        X = np.hstack([X_numeric.values, X_categorical.values])

        X = pd.DataFrame(X, columns=feature_names)
        X.rename(columns=lambda x: x.replace("_", " "), inplace=True)
        self._fbt_is_fitted = True

        return X

    def _transform_binarize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input features with fitted FeatureBinarizationFromTrees.

        Args:
            X (pd.DataFrame): Un-binarized features.

        Returns:
            pd.DataFrame: Binarized features.
        """
        X_numeric = X.select_dtypes(include='number')
        X_categorical = X.select_dtypes(exclude='number')
        numeric_columns = list(X_numeric.columns)
        categorical_columns = list(X_categorical.columns)

        X_numeric = self.fbt.transform(X_numeric)
        X_numeric = self._transform_column_names(X_numeric)

        feature_names = list(X_numeric.columns) + list(categorical_columns)
        X = np.hstack([X_numeric.values, X_categorical.values])

        X = pd.DataFrame(X, columns=feature_names)
        X.rename(columns=lambda x: x.replace("_", " "), inplace=True)

        return X

    def _encode_prediction_data(self, X):
        # ecode data in necessary format
        X = self._transform_encode_data(X)
        if self.binarization:
            X = self._transform_binarize_features(X)
        return X

    @staticmethod
    def _handle_non_dataframe_features(X):
        X = pd.DataFrame(X, columns=[f'feature {i}' for i in range(X.shape[1])])
        for col in X.columns:
            try:
                X[col] = X[col].astype(float).copy()
            except ValueError:
                pass
        return X

    @staticmethod
    def _handle_non_dataframe_targets(y):
        y = pd.DataFrame(y, columns=[f'Class {i}' for i in range(y.shape[1])])
        y = y.apply(lambda x: x.astype(np.float32), axis=1)
        return y

    def _handle_empty_feature_names(self, X: pd.DataFrame):
        # create textual feature names if they are not given by user
        if self.feature_names is None:
            self.feature_names = [f"the {x} was" for x in X.columns]
            X.columns = self.feature_names
        else:
            X.columns = self.feature_names
        return X

    def get_base_params(self, deep=False) -> dict:
        """
        Get parameters of the model

        Args:
            deep (bool): unused parameter

        Returns:
            dict: dictionary of parameters
        """
        return {
            'binarization': self.binarization,
            'tree_num': self.tree_num,
            'tree_depth': self.tree_depth,
            'tree_feature_selection': self.tree_feature_selection,
            'thresh_round': self.thresh_round,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            't_0': self.t_0,
            't_mult': self.t_mult,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'holdout_pct': self.holdout_pct,
            'early_stopping_plateau_count': self.early_stopping_plateau_count,
            'lookahead_steps': self.lookahead_steps,
            'lookahead_steps_size': self.lookahead_steps_size,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers,
            'num_workers': self.num_workers
        }

    def _get_instance_params(self):
        """Base class has no instance params"""
        return {}

    def get_params(self, deep=False):
        """
        Get parameters of the model

        Args:
            deep (bool): unused parameter

        Returns:
            dict: dictionary of parameters
        """
        instance_params = self._get_instance_params()
        base_params = self.get_base_params()
        base_params.update(instance_params)

        return base_params

    def _get_param_names(self):
        """
        Get the parameter names.

        Returns:
            list: list of parameter names
        """
        return list(self.get_params().keys())

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params()

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self
