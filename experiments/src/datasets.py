import openml
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class OpemlMLDataset:
    def __init__(self, openml_id, train_size, test_size, random_state):
        mms = MinMaxScaler()
        encoder = LabelEncoder()

        dataset = openml.datasets.get_dataset(openml_id)
        target_attribute = dataset.default_target_attribute
        X, y, _, feature_names = dataset.get_data(target=target_attribute)
        target_values = list(set(y))
        # X = pd.DataFrame(mms.fit_transform(X), columns=X.columns)
        df = pd.concat([X, y], axis=1)

        # To avoid ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
        for target_value in target_values:
            indices = df[df[target_attribute] == target_value].index

            if len(indices) < 5: 
                df = df.drop(indices, axis=0)

        target_values = list(df[target_attribute].unique())

        test_pct = (1.0 - train_size) * test_size
        train_val_data, test_data = train_test_split(
            df,
            test_size=test_pct,
            random_state=random_state,
            stratify=df[target_attribute],
        )

        if test_data.shape[0] > 6000:
            n_splits = 1
        elif 6000 >= test_data.shape[0] > 3000:
            n_splits = 2
        elif 3000 >= test_data.shape[0] > 1000:
            n_splits = 3
        else:
            n_splits = 5

        skf = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size / (1 - test_pct), random_state=42)
        X_train_folds = []
        X_val_folds = []
        y_train_folds = []
        y_val_folds = []
        for i, (train_index, val_index) in enumerate(skf.split(
                train_val_data.drop([target_attribute], axis=1), train_val_data[target_attribute])):

            # truncate training data to 10K https://arxiv.org/pdf/2207.08815.pdf
            if len(train_index) > 10000:
                rng = np.random.default_rng(0)
                train_index = rng.choice(train_index, 10000, replace=False)

            # truncate validation data to 50K https://arxiv.org/pdf/2207.08815.pdf
            if len(val_index) > 50000:
                rng = np.random.default_rng(0)
                val_index = rng.choice(val_index, 50000, replace=False)

            X_train_folds += [train_val_data.iloc[train_index].drop([target_attribute], axis=1)]
            X_val_folds += [train_val_data.iloc[val_index].drop([target_attribute], axis=1)]

            y_train = train_val_data.iloc[train_index][target_attribute]
            y_val = train_val_data.iloc[val_index][target_attribute]

            if y_train.dtype == 'category':
                y_train = y_train.cat.remove_unused_categories()
                y_val = y_val.cat.remove_unused_categories()

            if len(target_values) > 2:
                y_train_folds += [pd.get_dummies(y_train).astype(int).values]
                y_val_folds += [pd.get_dummies(y_val).astype(int).values]
            else:
                y_train_folds += [encoder.fit_transform(y_train).reshape(-1, 1)]
                y_val_folds += [encoder.fit_transform(y_val).reshape(-1, 1)]

            if n_splits == 1:
                break

        # create test data
        # truncate validation data to 50K https://arxiv.org/pdf/2207.08815.pdf
        if test_data.shape[0] > 50000:
            test_data = test_data.sample(n=50000, random_state=0)
        X_test = test_data.drop([target_attribute], axis=1)
        y_test = test_data[target_attribute]
        if y_test.dtype == 'category':
            y_test = y_test.cat.remove_unused_categories()

        # create train_val data
        # truncate training data to 10K https://arxiv.org/pdf/2207.08815.pdf
        if train_val_data.shape[0] > 10000:
            train_val_data = train_val_data.sample(n=10000, random_state=0)
        X_train_val = train_val_data.drop([target_attribute], axis=1)
        y_train_val = train_val_data[target_attribute]

        if len(target_values) > 2:
            y_test = pd.get_dummies(y_test).astype(int).values
            y_train_val = pd.get_dummies(y_train_val).astype(int).values
        else:
            y_test = encoder.fit_transform(y_test).reshape(-1, 1)
            y_train_val = encoder.fit_transform(y_train_val).reshape(-1, 1)

        self.numerical_features = [feature.name for feature in
                                   list(dataset.features.values()) if feature.data_type == 'numerical'
                                   or feature.data_type == 'numeric'
                                   and feature.name != target_attribute]
        self.categorical_features = [feature.name for feature in
                                     list(dataset.features.values()) if feature.data_type == 'string'
                                     and feature.name != target_attribute]
        # TODO: OpenML doesn't seem to specify the order of the nominal data.  Is there some way to infer it?
        self.ordinal_features = [feature.name for feature in
                                 list(dataset.features.values()) if feature.data_type == 'nominal'
                                 and feature.name != target_attribute]

        self.numerical_features = [x for x in self.numerical_features if x in df.columns]
        self.categorical_features = [x for x in self.categorical_features if x in df.columns]
        self.ordinal_features = [x for x in self.ordinal_features if x in df.columns]

        self.numeric_idxs = [df.columns.get_loc(x) for x in self.numerical_features]
        if len(self.categorical_features) + len(self.ordinal_features) != 0:
            self.non_numeric_idxs = [df.columns.get_loc(x) for x in self.categorical_features + self.ordinal_features]
            self.non_numeric_idxs_counts = tuple(df.iloc[:, self.non_numeric_idxs].nunique().values.tolist())
        else:
            self.non_numeric_idxs = [-1]
            self.non_numeric_idxs_counts = (1,)

        self.dataset = dataset
        self.target_attribute = target_attribute
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.target_values = target_values
        self.df = df
        self.X_train_val = X_train_val
        self.y_train_val = y_train_val
        self.X_train_folds = X_train_folds
        self.X_val_folds = X_val_folds
        self.X_test = X_test
        self.y_train_folds = y_train_folds
        self.y_val_folds = y_val_folds
        self.y_test = y_test


class TorchDataset(Dataset):
    def __init__(self, X: np.array, y: np.array, numeric_idxs=None, categorical_idxs=None):
        """
        Dataset suitable for BanditRRN model from torchlogic

        Args:
            X (np.array): features data scaled to [0, 1]
            y (np.array): target data of classes 0, 1
        """
        super(TorchDataset, self).__init__()
        self.X = X
        self.y = y
        self.numeric_idxs = numeric_idxs
        self.categorical_idxs = categorical_idxs
        self.sample_idx = np.arange(X.shape[0])  # index of samples
        if self.categorical_idxs is not None:
            if self.categorical_idxs[0] == -1:
                self.X = np.hstack([self.X, np.zeros(shape=(self.X.shape[0], 1))])
                self.categorical_idxs = [self.X.shape[1] - 1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        features = torch.from_numpy(self.X[idx, :]).float()
        target = torch.from_numpy(self.y[idx, :])
        if self.categorical_idxs is not None:
            return {"features": features, "target": target, "sample_idx": idx,
                    "num_idxs": torch.tensor(self.numeric_idxs), "cat_idxs": torch.tensor(self.categorical_idxs)}
        else:
            return {"features": features, "target": target, "sample_idx": idx}


__all__ = ["OpemlMLDataset", "TorchDataset"]
