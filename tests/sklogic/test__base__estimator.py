import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import pytest
from pytest import fixture

from torchlogic.sklogic.base.base_estimator import BaseSKLogicEstimator


class TestBaseEstimator:

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
    def data_frames(self):
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
    def model(self):
        torch.manual_seed(0)
        np.random.seed(0)

        model = BaseSKLogicEstimator(holdout_pct=0.5)
        return model

    @staticmethod
    def test__holdout_samplers(model, data_sets):
        train_dataset, test_dataset = data_sets
        train_ds, holdout_ds = model._create_holdout_samplers(train_dataset=train_dataset)

        assert isinstance(train_ds, SubsetRandomSampler), "training sampler is not of correct type"
        assert isinstance(holdout_ds, SubsetRandomSampler), "holdout sampler is not of correct type"
        assert len(train_ds) == 2, "training data was not split correctly"
        assert len(holdout_ds) == 2, "holdout data was not split correctly"

    @staticmethod
    def test__generate_training_data_loaders(model, data_sets):
        train_dataset, test_dataset = data_sets
        train_dl, holdout_dl = model._generate_training_data_loaders(dataset=train_dataset)

        assert isinstance(train_dl, DataLoader), "training loader is not of correct type"
        assert isinstance(holdout_dl, DataLoader), "holdout loader is not of correct type"
        assert len(train_dl.dataset) == 4, "training data is not correct"
        assert len(holdout_dl.dataset) == 4, "holdout data is not correct"
        assert len(train_dl) == 1, "train data loader batches is not correct"
        assert len(holdout_dl) == 1, "holdout data loader batches is not correct"

    @staticmethod
    def test__fit_transform_encode_data(model, data_frames):
        X, y = data_frames
        model.binarization = False
        X_transformed = model._fit_transform_encode_data(X)

        X_transformed_test = pd.DataFrame({
            'feature1': [1.0, 0.5, 1.0, 0.0],
            'feature2': [0.0,0.25, 0.5, 1.0],
            'feature3': [0.0, 0.0, 0.0, 0.0],
            'feature4': [0.0, 0.0, 0.0, 0.0],
            'feature5_one': [True, False, True, False],
            'feature5_two': [False, True, False, True],
        })

        assert X_transformed.equals(X_transformed_test), "transformation was not correct"

    @staticmethod
    def test__transform_encode_data(model, data_frames):
        X, y = data_frames
        X_transformed = model._fit_transform_encode_data(X)
        X_transformed_test = model._transform_encode_data(X)

        assert X_transformed.equals(X_transformed_test), "transformation was not correct"

    @staticmethod
    def test__fit_transform_binarize_features(model, data_frames):
        X, y = data_frames
        X_transformed = model._fit_transform_encode_data(X)
        X_binarized = model._fit_transform_binarize_features(X_transformed, y)

        assert list(X_binarized.columns) == [
            'feature1 less than or equal to 0.5', 'feature1 less than or equal to 0.75',
            'feature1 greater than 0.5', 'feature1 greater than 0.75', 'feature2 less than or equal to 1.5',
            'feature2 less than or equal to 2.5', 'feature2 less than or equal to 4.0',
            'feature2 greater than 1.5', 'feature2 greater than 2.5', 'feature2 greater than 4.0',
            'feature5 one', 'feature5 two'
        ], "binarization did not complete correctly"

    @staticmethod
    def test__transform_binarize_features(model, data_frames):
        X, y = data_frames
        X_transformed = model._fit_transform_encode_data(X)
        X_binarized = model._fit_transform_binarize_features(X_transformed.copy(), y)
        X_binarized_test = model._transform_binarize_features(X_transformed.copy())

        assert X_binarized.equals(X_binarized_test), "binarization did not complete correctly"

    @staticmethod
    def test__encode_prediction_data(model, data_frames):
        X, y = data_frames
        X_transformed = model._fit_transform_encode_data(X)
        X_binarized = model._fit_transform_binarize_features(X_transformed.copy(), y)
        X_binarized_test = model._encode_prediction_data(X)
        assert X_binarized.equals(X_binarized_test), "binarization did not complete correctly"

        model.binarization = False
        X_transformed = model._fit_transform_encode_data(X)
        X_transformed_test = model._encode_prediction_data(X)
        assert X_transformed.equals(X_transformed_test), "transformation did not complete correctly"

    @staticmethod
    def test__get_base_params(model):
        default_params = [
            'binarization', 'tree_num', 'tree_depth',
            'tree_feature_selection', 'thresh_round',
            'learning_rate', 'weight_decay', 't_0',
            't_mult', 'epochs', 'batch_size', 'holdout_pct',
            'early_stopping_plateau_count', 'lookahead_steps',
            'lookahead_steps_size', 'pin_memory',
            'persistent_workers', 'num_workers']
        assert list(model.get_base_params().keys()) == default_params, "parameters not correct"

    @staticmethod
    def test_get_params(model):
        default_params = [
            'binarization', 'tree_num', 'tree_depth',
            'tree_feature_selection', 'thresh_round',
            'learning_rate', 'weight_decay', 't_0',
            't_mult', 'epochs', 'batch_size', 'holdout_pct',
            'early_stopping_plateau_count', 'lookahead_steps',
            'lookahead_steps_size', 'pin_memory',
            'persistent_workers', 'num_workers']
        assert list(model.get_params().keys()) == default_params, "parameters not correct"

    @staticmethod
    def test__get_parm_names(model):
        default_params = [
            'binarization', 'tree_num', 'tree_depth',
            'tree_feature_selection', 'thresh_round',
            'learning_rate', 'weight_decay', 't_0',
            't_mult', 'epochs', 'batch_size', 'holdout_pct',
            'early_stopping_plateau_count', 'lookahead_steps',
            'lookahead_steps_size', 'pin_memory',
            'persistent_workers', 'num_workers']
        assert model._get_param_names() == default_params, "parameters not correct"

    @staticmethod
    def test_set_params(model):
        model.set_params(**{'binarization': False})
        assert model.binarization == False

        model.set_params(**{'learning_rate': 0.1, 'lookahead_steps': 5})
        assert model.learning_rate == 0.1 and model.lookahead_steps == 5

        with pytest.raises(ValueError) as e:
            model.set_params(**{'unk': 0.1})



