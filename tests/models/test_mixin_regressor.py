import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

from pytest import fixture

from io import StringIO
from unittest.mock import patch

from torchlogic.models import BanditNRNRegressor
from torchlogic.utils.trainers import BoostedBanditNRNTrainer


class TestRegressorMixin:

    @fixture
    def one_layer_one_class_model(self):

        torch.manual_seed(0)
        np.random.seed(0)

        model = BanditNRNRegressor(
            target_names='metric1',
            feature_names=['feat1', 'feat1', 'feat3', 'feat4'],
            input_size=4,
            layer_sizes=[3, ],
            n_selected_features_input=2,
            n_selected_features_internal=2,
            n_selected_features_output=2,
            perform_prune_quantile=0.5,
            ucb_scale=2.5
        )

        for item in model.state_dict():
            if item.find('weights') > -1:
                weights = model.state_dict()[item]
                weights.data.copy_(torch.ones_like(weights.data))

        return model

    @fixture
    def one_layer_model(self):

        torch.manual_seed(0)
        np.random.seed(0)

        model = BanditNRNRegressor(
            target_names='metric1',
            feature_names=['feat1', 'feat1', 'feat3', 'feat4'],
            input_size=4,
            layer_sizes=[3, ],
            n_selected_features_input=2,
            n_selected_features_internal=2,
            n_selected_features_output=2,
            perform_prune_quantile=0.5,
            ucb_scale=2.5
        )

        for item in model.state_dict():
            if item.find('weights') > -1:
                weights = model.state_dict()[item]
                weights.data.copy_(torch.ones_like(weights.data))

        return model

    @fixture
    def two_layer_model(self):

        torch.manual_seed(0)
        np.random.seed(0)

        model = BanditNRNRegressor(
            target_names='metric1',
            feature_names=['feat1', 'feat1', 'feat3', 'feat4'],
            input_size=4,
            layer_sizes=[5, 5],
            n_selected_features_input=2,
            n_selected_features_internal=2,
            n_selected_features_output=2,
            perform_prune_quantile=0.5,
            ucb_scale=2.5
        )

        for item in model.state_dict():
            if item.find('weights') > -1:
                weights = model.state_dict()[item]
                weights.data.copy_(torch.ones_like(weights.data))

        return model

    # boosted model fixtures
    @fixture
    def data_loaders(self):
        class StaticDataset(Dataset):
            def __init__(self):
                super(StaticDataset, self).__init__()
                self.X = pd.DataFrame(
                    {'feature1': [1, 0, 1, 0] * 1000,
                     'feature2': [0, 1, 0, 1] * 1000}).values
                self.y = pd.DataFrame(
                    {'predictions': [1, 0, 1, 0] * 1000}).values
                self.sample_idx = np.arange(self.X.shape[0])  # index of samples

            def __len__(self):
                return self.X.shape[0]

            def __getitem__(self, idx):
                features = torch.from_numpy(self.X[idx, :]).float()
                target = torch.from_numpy(self.y[idx, :])
                return {'features': features, 'target': target, 'sample_idx': idx}

        return (DataLoader(StaticDataset(), batch_size=1000, shuffle=False),
                DataLoader(StaticDataset(), batch_size=1000, shuffle=False))

    @fixture
    def boosted_model(self):

        torch.manual_seed(0)
        np.random.seed(0)

        model = BanditNRNRegressor(
            target_names='prediction',
            feature_names=['feat1', 'feat2'],
            input_size=2,
            layer_sizes=[2, ],
            n_selected_features_input=2,
            n_selected_features_internal=2,
            n_selected_features_output=2,
            perform_prune_quantile=0.5,
            ucb_scale=2.5
        )

        for item in model.state_dict():
            if item.find('weights') > -1:
                weights = model.state_dict()[item]
                weights.data.copy_(torch.ones_like(weights.data) * 0.5)

        return model

    @fixture
    def boosted_trainer(self, boosted_model):
        return BoostedBanditNRNTrainer(
            boosted_model,
            torch.nn.BCELoss(),
            torch.optim.Adam(boosted_model.rn.parameters()),
            perform_prune_plateau_count=0,
            augment=None,
            partial_fit=True
        )

    @staticmethod
    def test__explain_sample(two_layer_model, one_layer_one_class_model):

        x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        out = two_layer_model.explain_samples(
            x,
            quantile=1.0,
            target_names=['metric1'],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix='The sample has'
        )
        assert (out == '0: The sample has a predicted metric1 of 1.0 because: '
                       '\n\n\nAND '
                       '\n\tAND '
                       '\n\t\tFeat3 >= 0.0'
                       '\n\t\tFeat4 >= 0.0'
                       '\n\tOR '
                       '\n\t\tAND '
                       '\n\t\t\tFeat1 >= 0.0'
                       '\n\t\t\tFeat4 >= 0.0'
                       '\n\t\tAND '
                       '\n\t\t\tFeat3 >= 0.0'
                       '\n\t\t\tFeat4 >= 0.0')

        x = torch.tensor([[1.0, 1.0, 1.0, 0.37]])
        out = two_layer_model.explain_samples(
            x,
            quantile=1.0,
            target_names=['metric1'],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix='The sample has'
        )
        assert (out == '0: The sample has a predicted metric1 of 0.48 because: \n\n\nNOT \n\tFeat4 >= 0.375')

        x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        out = two_layer_model.explain_samples(
            x,
            quantile=1.0,
            target_names=['metric1'],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix='The sample has'
        )
        assert (out == '0: The sample has a predicted metric1 of 1.0 because: '
                       '\n\n\nAND '
                       '\n\tAND '
                       '\n\t\tFeat3 >= 0.0'
                       '\n\t\tFeat4 >= 0.0'
                       '\n\tOR '
                       '\n\t\tAND '
                       '\n\t\t\tFeat1 >= 0.0'
                       '\n\t\t\tFeat4 >= 0.0'
                       '\n\t\tAND '
                       '\n\t\t\tFeat3 >= 0.0'
                       '\n\t\t\tFeat4 >= 0.0')

        x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        out = one_layer_one_class_model.explain_samples(
            x,
            quantile=1.0,
            target_names=['metric1'],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix='The sample has'
        )
        assert (out == '0: The sample has a predicted metric1 of 1.0 because: '
                       '\n\n\nOR '
                       '\n\tFeat1 >= 0.0'
                       '\n\tAND '
                       '\n\t\tFeat1 >= 0.0'
                       '\n\t\tFeat4 >= 0.0')

        x = torch.tensor([[1.0, 0.37, 1.0, 1.0]])
        out = one_layer_one_class_model.explain_samples(
            x,
            quantile=1.0,
            target_names=['metric1'],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix='The sample has'
        )
        assert (out == '0: The sample has a predicted metric1 of 0.74 because: '
                       '\n\n\nOR '
                       '\n\tAND '
                       '\n\t\tFeat1 >= 0.363'
                       '\n\t\tFeat1 >= 0.993'
                       '\n\tAND '
                       '\n\t\tFeat1 >= 0.363'
                       '\n\t\tFeat4 >= 0.993')

    @staticmethod
    def test__explain(one_layer_model, two_layer_model):

        out = two_layer_model.explain(
            quantile=1.0,
            required_output_thresholds=torch.tensor(0.5),
            target_names=['metric1'],
            explain_type='both',
            print_type='logical',
            explanation_prefix='A sample has a'
        )
        assert (out == 'A sample has a predicted metric1 of 0.5 because: \n\n\n'
                       'AND '
                       '\n\tAND '
                       '\n\t\tFeat3 >= 0.688'
                       '\n\t\tFeat4 >= 0.688'
                       '\n\tOR '
                       '\n\t\tAND '
                       '\n\t\t\tFeat1 >= 0.688'
                       '\n\t\t\tFeat4 >= 0.688'
                       '\n\t\tAND '
                       '\n\t\t\tFeat3 >= 0.688'
                       '\n\t\t\tFeat4 >= 0.688')

        out = one_layer_model.explain(
            quantile=1.0,
            required_output_thresholds=torch.tensor(0.5),
            target_names=['metric1'],
            explain_type='both',
            print_type='logical',
            explanation_prefix='A sample has a'
        )
        assert (out == 'A sample has a predicted metric1 of 0.5 because: '
                       '\n\n\nOR '
                       '\n\tFeat1 >= 0.625'
                       '\n\tAND '
                       '\n\t\tFeat1 >= 0.625'
                       '\n\t\tFeat4 >= 0.625')

        out = one_layer_model.explain(
            quantile=1.0,
            required_output_thresholds=torch.tensor(0.5),
            target_names=['metric1'],
            explain_type='both',
            print_type='logical-natural',
            explanation_prefix='A sample has a'
        )
        assert (out == 'A sample has a predicted metric1 of 0.5 because: '
                       '\n\n\nAny of the following are true: '
                       '\n\tFeat1 >= 0.625'
                       '\n\tAll the following are true: '
                       '\n\t\tFeat1 >= 0.625'
                       '\n\t\tFeat4 >= 0.625')

    @staticmethod
    @patch('sys.stdout', new_callable=StringIO)
    def test__print_sample(stdout, one_layer_model):

        x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        one_layer_model.print_samples(
            x,
            quantile=1.0,
            target_names=['metric1'],
            explain_type='both',
            print_type='logical'
        )
        output = stdout.getvalue()
        assert (output == "REASONING NETWORK MODEL FOR: metric1\n"
                          "Logic at depth 2: feat1 >= 0.0\n"
                          "output: tensor([1., 1.])\n\n"
                          "Logic at depth 1: ['feat1 >= 0.0']\n"
                          "weights: tensor([1., 1.])\n"
                          "output: 1.0\n"
                          "required_threshold: 0.0\n\n"
                          "Logic at depth 2: feat1 >= 0.0, feat4 >= 0.0\n"
                          "output: tensor([1., 1.])\n\n"
                          "Logic at depth 1: ['AND(feat1 >= 0.0, feat4 >= 0.0)']\n"
                          "weights: tensor([1., 1.])\n"
                          "output: 1.0\n"
                          "required_threshold: 0.0\n\n"
                          "Logic at depth 0: ['OR(AND(feat1 >= 0.0, feat4 >= 0.0), feat1 >= 0.0)']\n"
                          "weights: tensor([1., 1.])\n"
                          "output: 1.0\n"
                          "required_threshold: 0.9900000691413879\n\n")

    @staticmethod
    @patch('sys.stdout', new_callable=StringIO)
    def test__print(stdout):

        torch.manual_seed(0)
        np.random.seed(0)

        module = BanditNRNRegressor(
            target_names='metric1',
            feature_names=['feat1', 'feat1', 'feat3', 'feat4'],
            input_size=4,
            layer_sizes=[3, ],
            n_selected_features_input=2,
            n_selected_features_internal=2,
            n_selected_features_output=2,
            perform_prune_quantile=0.5,
            ucb_scale=2.5
        )

        for item in module.state_dict():
            if item.find('weights') > -1:
                weights = module.state_dict()[item]
                weights.data.copy_(torch.ones_like(weights.data))

        module.print(
            quantile=1.0,
            required_output_thresholds=torch.tensor(0.5),
            target_names=['metric1'],
            explain_type='both',
            print_type='logical'
        )
        output = stdout.getvalue()
        assert (output == "REASONING NETWORK MODEL FOR: metric1\n"
                          "Logic at depth 2: feat1 >= 0.625\n"
                          "output: tensor([0.6250, 0.6250])\n\n"
                          "Logic at depth 1: ['feat1 >= 0.625']\n"
                          "weights: tensor([1., 1.])\n"
                          "output: 0.25\n"
                          "required_threshold: 0.25\n\n"
                          "Logic at depth 2: feat1 >= 0.625, feat4 >= 0.625\n"
                          "output: tensor([0.6250, 0.6250])\n\n"
                          "Logic at depth 1: ['AND(feat1 >= 0.625, feat4 >= 0.625)']\n"
                          "weights: tensor([1., 1.])\n"
                          "output: 0.25\n"
                          "required_threshold: 0.25\n\n"
                          "Logic at depth 0: ['OR(AND(feat1 >= 0.625, feat4 >= 0.625), feat1 >= 0.625)']\n"
                          "weights: tensor([1., 1.])\n"
                          "output: 0.5\n"
                          "required_threshold: 0.5\n\n")

    @staticmethod
    def test__explain_sample_boosted_model(data_loaders, boosted_trainer):
        train_dl, val_dl = data_loaders
        boosted_trainer.boost(train_dl)

        # RRN and Boosted RRN have the same multi-class predictions
        x = torch.tensor([[1.0, 0.0]])
        boosted_trainer.model.rn.output_layer.weights[0].data.copy_(torch.tensor([[0.6000], [0.6000]]))
        out = boosted_trainer.model.explain_samples(
            x,
            quantile=1.0,
            target_names=['value'],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix='The sample has'
        )
        assert (out == "0: The sample has a predicted value of 0.871 because: \n\n\n"
                       "AND \n\tFeat1 >= 0.98\n\tFeat2 >= 0.0")

        # RRN and Boosted RRN have different multi-class predictions
        # TODO: This is a strange interpretation of the result but in a way its accurate because it tells you the
        #  direction to make the prediction more negative
        x = torch.tensor([[0.0, 1.0]])
        boosted_trainer.model.rn.output_layer.weights[0].data.copy_(torch.tensor([[0.6000], [0.6000]]))
        out = boosted_trainer.model.explain_samples(
            x,
            quantile=1.0,
            target_names=['value'],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix='The sample has'
        )
        assert (out == "0: The sample has a predicted value of 0.329 because: \n\n\n"
                       "NOT \n\tFeat1 >= 0.02")
