import copy

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

import torch
from torch.utils.data import Dataset

from pytest import fixture

from torchlogic.sklogic.regressors import RNRNRegressor


class TestRNRNRegressor:

    @fixture
    def data_frames(self):
        X, y = make_regression(
            n_samples=10, n_features=4, n_informative=4, n_targets=1, tail_strength=0.1, random_state=0)
        X = pd.DataFrame(X, columns=[f"feature{i}" for i in range(X.shape[1])])
        y = pd.DataFrame(y, columns=["Target1"])
        return X, y

    @fixture
    def data_sets(self, data_frames):
        class StaticDataset(Dataset):
            def __init__(self):
                X, y = data_frames
                super(StaticDataset, self).__init__()
                self.X = X.values
                self.y = y.values
                self.sample_idx = np.arange(self.X.shape[0])  # index of samples

            def __len__(self):
                return self.X.shape[0]

            def __getitem__(self, idx):
                features = torch.from_numpy(self.X[idx, :]).float()
                target = torch.from_numpy(self.y[idx, :])
                return {'features': features, 'target': target, 'sample_idx': idx}

        return (StaticDataset(), StaticDataset())

    @fixture
    def model(self):
        torch.manual_seed(0)
        np.random.seed(0)

        model = RNRNRegressor(holdout_pct=0.5)
        return model

    @staticmethod
    def test_fit(model, data_frames):
        model1 = copy.copy(model)
        X, y = data_frames
        model1.fit(X, y)
        assert model1._fbt_is_fitted
        assert model1.fbt is not None
        assert model1.model is not None

        # should still fit if arrays are passed
        model2 = copy.copy(model)
        model2.fit(X.to_numpy(), y.to_numpy())
        assert model2._fbt_is_fitted
        assert model2.fbt is not None
        assert model2.model is not None

    @staticmethod
    def test_predict(model, data_frames):
        torch.manual_seed(0)
        np.random.seed(0)
        model1 = copy.copy(model)
        X, y = data_frames
        model1.fit(X, y)
        predictions = model1.predict(X)
        assert predictions.max().values <= y.max().values, "range was not correct"
        assert predictions.min().values >= y.min().values, "range was not correct"
        assert list(predictions.columns) == ["Target"]

        model2 = copy.copy(model)
        model2.target_name = 'The Target'
        torch.manual_seed(0)
        np.random.seed(0)
        model2.fit(X, y)
        predictions = model2.predict(X)
        assert predictions.max().values <= y.max().values, "range was not correct"
        assert predictions.min().values >= y.min().values, "range was not correct"
        assert list(predictions.columns) == ['The Target']

    @staticmethod
    def test_score(model, data_frames):
        torch.manual_seed(0)
        np.random.seed(0)

        model1 = copy.copy(model)

        X, y = data_frames

        model1.fit(X, y)
        score = model1.score(X, y)
        # TODO: what other tests are reasonable?
        assert score > 0
        assert isinstance(score, float)

    @staticmethod
    def test_explain_sample(model, data_frames):
        torch.manual_seed(0)
        np.random.seed(0)

        model1 = copy.copy(model)

        X, y = data_frames
        model1.fit(X, y)
        explanation = model1.explain_sample(X, sample_index=0)
        assert explanation == ('0: The sample has a predicted Target of -87.051 because: '
                               '\n\n\nAND \n\tThe feature0 was greater than -0.204\n\tThe '
                               'feature1 was greater than -1.479\n\tThe feature2 was greater '
                               'than -1.328\n\tThe feature3 was greater than 0.234')
