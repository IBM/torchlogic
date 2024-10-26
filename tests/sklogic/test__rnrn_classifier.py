import copy

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import pytest
from pytest import fixture

from torchlogic.sklogic.classifiers import RNRNClassifier


class TestRNRNClassifier:

    @fixture
    def data_sets(self):
        class StaticDataset(Dataset):
            def __init__(self):
                super(StaticDataset, self).__init__()
                self.X = pd.DataFrame(
                    {'feature1': [1, 1, 1, 1],
                     'feature2': [1, 1, 1, 1],
                     'feature3': [1, 1, 1, 1],
                     'feature4': [1, 1, 1, 1]}).values
                self.y = pd.DataFrame(
                    {'class1': [1, 0, 1, 0],
                     'class2': [0, 1, 0, 1]}).values
                self.sample_idx = np.arange(self.X.shape[0])  # index of samples

            def __len__(self):
                return self.X.shape[0]

            def __getitem__(self, idx):
                features = torch.from_numpy(self.X[idx, :]).float()
                target = torch.from_numpy(self.y[idx, :])
                return {'features': features, 'target': target, 'sample_idx': idx}

        return (StaticDataset(), StaticDataset())

    @fixture
    def data_frames_multi(self):
        X = pd.DataFrame(
            {'feature1': [1, 0.5, 1, 0],
             'feature2': [1, 2, 3, 5],
             'feature3': [1, 1, 1, 1],
             'feature4': [1, 1, 1, 1],
             'feature5': ['one', 'two', 'one', 'two']})
        y = pd.DataFrame(
            {'class1': [1, 0, 1, 0],
             'class2': [0, 1, 0, 1]})

        return X, y

    @fixture
    def data_frames_binary(self):
        X = pd.DataFrame(
            {'feature1': [1, 0.5, 1, 0],
             'feature2': [1, 2, 3, 5],
             'feature3': [1, 1, 1, 1],
             'feature4': [1, 1, 1, 1],
             'feature5': ['one', 'two', 'one', 'two']})
        y = pd.DataFrame(
            {'class1': [1, 0, 1, 0]})

        return X, y

    @fixture
    def data_frames_binary_varies(self):
        X = pd.DataFrame(
            {'feature1': [1, 0.5, 1, 0],
             'feature2': [1, 2, 3, 5],
             'feature3': [1, 4, 8, 16],
             'feature4': [1, 4, 8, 16],
             'feature5': ['one', 'two', 'one', 'two']})
        y = pd.DataFrame(
            {'class1': [1, 0, 1, 0]})

        return X, y

    @fixture
    def model_multi(self):
        torch.manual_seed(0)
        np.random.seed(0)

        model = RNRNClassifier(holdout_pct=0.5, multi_class=True)
        return model

    @fixture
    def model_binary(self):
        torch.manual_seed(0)
        np.random.seed(0)

        model = RNRNClassifier(holdout_pct=0.5, multi_class=False)
        return model

    @staticmethod
    def test_fit_multi(model_multi, data_frames_multi):
        model1 = copy.copy(model_multi)
        X, y = data_frames_multi
        model1.fit(X, y)
        assert model1._fbt_is_fitted
        assert model1.fbt is not None
        assert model1.model is not None

        # should still fit if arrays are passed
        model2 = copy.copy(model_multi)
        model2.fit(X.to_numpy(), y.to_numpy())
        assert model2._fbt_is_fitted
        assert model2.fbt is not None
        assert model2.model is not None

    @staticmethod
    def test_fit_binary(model_binary, data_frames_binary):
        model1 = copy.copy(model_binary)
        X, y = data_frames_binary
        model1.fit(X, y)
        assert model1._fbt_is_fitted
        assert model1.fbt is not None
        assert model1.model is not None

        # should still fit if arrays are passed
        model2 = copy.copy(model_binary)
        model2.fit(X.to_numpy(), y.to_numpy())
        assert model2._fbt_is_fitted
        assert model2.fbt is not None
        assert model2.model is not None

    @staticmethod
    def test_predict_multi(model_multi, data_frames_multi):
        torch.manual_seed(0)
        np.random.seed(0)

        model1 = copy.copy(model_multi)

        X, y = data_frames_multi
        model1.fit(X, y)
        predictions = model1.predict(X)
        predictions_test = pd.DataFrame({'Class 0': [1, 0, 1, 0], 'Class 1': [0, 1, 0, 1]})
        assert predictions.equals(predictions_test)

        predictions = model1.predict(X.values)
        assert predictions.equals(predictions_test)

        model2 = copy.copy(model_multi)
        model2.target_names = ['First Class', 'Second Class']

        torch.manual_seed(0)
        np.random.seed(0)

        model2.fit(X, y)
        predictions = model2.predict(X)
        predictions_test = pd.DataFrame({'First Class': [1, 0, 1, 0], 'Second Class': [0, 1, 0, 1]})
        assert predictions.equals(predictions_test)

    @staticmethod
    def test_predict_binary(model_binary, data_frames_binary):
        torch.manual_seed(0)
        np.random.seed(0)

        model1 = copy.copy(model_binary)

        X, y = data_frames_binary
        model1.fit(X, y)
        predictions = model1.predict(X)
        predictions_test = pd.DataFrame({'Class 0': [0, 1, 1, 0]})
        assert predictions.equals(predictions_test)

        predictions = model1.predict(X.values)
        assert predictions.equals(predictions_test)

        model2 = copy.copy(model_binary)
        model2.target_names = ['First Class']

        torch.manual_seed(0)
        np.random.seed(0)

        model2.fit(X, y)
        predictions = model2.predict(X)
        predictions_test = pd.DataFrame({'First Class': [0, 1, 1, 0]})
        assert predictions.equals(predictions_test)

    @staticmethod
    def test_predict_proba_multi(model_multi, data_frames_multi):
        torch.manual_seed(0)
        np.random.seed(0)

        model1 = copy.copy(model_multi)

        X, y = data_frames_multi
        model1.fit(X, y)
        predictions = model1.predict_proba(X)
        predictions_test = pd.DataFrame(
            {'Class 0': [0.635350, 0.598017, 0.609710, 0.587753],
             'Class 1': [0.693469, 0.789932, 0.736102, 0.827008]})
        assert np.isclose(predictions.values, predictions_test.values).all()

    @staticmethod
    def test_predict_proba_binary(model_binary, data_frames_binary):
        torch.manual_seed(0)
        np.random.seed(0)

        model1 = copy.copy(model_binary)

        X, y = data_frames_binary
        model1.fit(X, y)
        predictions = model1.predict_proba(X)
        predictions_test = pd.DataFrame(
            {'Class 0': [0.172404, 0.178666, 0.172924, 0.168914]})
        assert np.isclose(predictions.values, predictions_test.values).all()

    @staticmethod
    def test_score_multi(model_multi, data_frames_multi):
        torch.manual_seed(0)
        np.random.seed(0)

        model1 = copy.copy(model_multi)

        X, y = data_frames_multi
        model1.fit(X, y)
        score = model1.score(X, y)
        assert score == 1.0

    @staticmethod
    def test_score_binary(model_binary, data_frames_binary):
        torch.manual_seed(0)
        np.random.seed(0)

        model1 = copy.copy(model_binary)

        X, y = data_frames_binary
        model1.fit(X, y)
        score = model1.score(X, y)
        assert score == 0.5

    @staticmethod
    def test_explain_sample_multi(model_multi, data_frames_multi):
        torch.manual_seed(0)
        np.random.seed(0)

        model1 = copy.copy(model_multi)

        X, y = data_frames_multi
        model1.fit(X, y)
        with pytest.raises(AssertionError) as e:
            _ = model1.explain_sample(X, sample_index=0)

    @staticmethod
    def test_explain_sample_binary(model_binary, data_frames_binary, data_frames_binary_varies):
        torch.manual_seed(0)
        np.random.seed(0)

        model1 = copy.copy(model_binary)

        X, y = data_frames_binary
        model1.fit(X, y)
        explanation = model1.explain_sample(X, sample_index=0)
        assert explanation == "0: The prediction is in the not Class 0 because: \n\n\nAND \n\t" \
                              "The feature1 was greater than 0.5 (50th percentile)\n\tThe feature2 " \
                              "was less than or equal to 1.5 (12th percentile)\n\tThe feature5 was " \
                              "one\n\tNOT the feature5 was two"

        model2 = copy.copy(model_binary)
        model2.binarization = False

        X, y = data_frames_binary_varies
        model2.fit(X, y)
        explanation = model2.explain_sample(X, sample_index=0)
        assert explanation == ('0: The prediction is in the Class 0 because: '
                               '\n\n\nAND \n\tThe feature1 was greater than or equal to 1 '
                               '(100th percentile)\n\tThe feature2 was less than 0 (0th percentile)'
                               '\n\tThe feature3 was less than 0.276 (27th percentile)\n\t'
                               'The feature4 was less than 0 (0th percentile)\n\t'
                               'The feature5 two was less than 0')
