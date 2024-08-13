import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

from pytest import fixture

from io import StringIO
from unittest.mock import patch

from torchlogic.models import BanditNRNClassifier
from torchlogic.utils.trainers import BoostedBanditNRNTrainer


class TestClassifierMixin:

    @fixture
    def one_layer_one_class_model(self):

        torch.manual_seed(0)
        np.random.seed(0)

        model = BanditNRNClassifier(
            target_names=['class1'],
            feature_names=['feat1', 'feat2', 'feat3', 'feat4'],
            input_size=4,
            output_size=1,
            layer_sizes=[5,],
            n_selected_features_input=2,
            n_selected_features_internal=2,
            n_selected_features_output=2,
            perform_prune_quantile=0.5,
            ucb_scale=2.5
        )

        for item in model.state_dict():
            if item.find('weights') > -1:
                weights = model.state_dict()[item]
                weights.data.copy_(torch.ones_like(weights.data) * 2)

        return model

    @fixture
    def one_layer_one_class_model_negatives(self):

        torch.manual_seed(0)
        np.random.seed(0)

        model = BanditNRNClassifier(
            target_names=['class1'],
            feature_names=['feat1', 'feat2', 'feat3', 'feat4'],
            input_size=4,
            output_size=1,
            layer_sizes=[5,],
            n_selected_features_input=2,
            n_selected_features_internal=2,
            n_selected_features_output=2,
            perform_prune_quantile=0.5,
            ucb_scale=2.5
        )

        for item in model.state_dict():
            if item.find('weights') > -1:
                weights = model.state_dict()[item]
                weights.data.copy_(torch.ones_like(weights.data) * 2)

        model.model[0].weights.data.copy_(model.model[0].weights.data * -1)

        return model

    @fixture
    def one_layer_model(self):

        torch.manual_seed(0)
        np.random.seed(0)

        model = BanditNRNClassifier(
            target_names=['class1', 'class2'],
            feature_names=['feat1', 'feat2', 'feat3', 'feat4'],
            input_size=4,
            output_size=2,
            layer_sizes=[5,],
            n_selected_features_input=2,
            n_selected_features_internal=2,
            n_selected_features_output=2,
            perform_prune_quantile=0.5,
            ucb_scale=2.5
        )

        scale = torch.tensor([1.0, 2.0]).unsqueeze(-1).unsqueeze(-1).to(model.output_layer.weights.device)
        for item in model.state_dict():
            if item.find('weights') > -1:
                weights = model.state_dict()[item]
                weights.data.copy_(torch.ones_like(weights.data).to(model.output_layer.weights.device) * scale)

        return model

    @fixture
    def two_layer_model(self):

        torch.manual_seed(0)
        np.random.seed(0)

        model = BanditNRNClassifier(
            target_names=['class1', 'class2'],
            feature_names=['feat1', 'feat2', 'feat3', 'feat4'],
            input_size=4,
            output_size=2,
            layer_sizes=[5, 5],
            n_selected_features_input=2,
            n_selected_features_internal=2,
            n_selected_features_output=2,
            perform_prune_quantile=0.5,
            ucb_scale=2.5
        )

        scale = torch.tensor([1.0, 2.0]).unsqueeze(-1).unsqueeze(-1).to(model.output_layer.weights.device)
        for item in model.state_dict():
            if item.find('weights') > -1:
                weights = model.state_dict()[item]
                weights.data.copy_(torch.ones_like(weights.data).to(model.output_layer.weights.device) * scale)

        return model

    @fixture
    def two_layer_model_negated(self):

        torch.manual_seed(0)
        np.random.seed(0)

        model = BanditNRNClassifier(
            target_names=['class1', 'class2'],
            feature_names=['feat1', 'feat2', 'feat3', 'feat4'],
            input_size=4,
            output_size=2,
            layer_sizes=[5, 5],
            n_selected_features_input=2,
            n_selected_features_internal=2,
            n_selected_features_output=2,
            perform_prune_quantile=0.5,
            ucb_scale=2.5
        )

        scale = torch.tensor([1.0, 2.0]).unsqueeze(-1).unsqueeze(-1).to(model.output_layer.weights.device)
        for item in model.state_dict():
            if item.find('weights') > -1:
                weights = model.state_dict()[item]
                if weights.size() == (2, 2, 5):
                    negation_mask = torch.tensor([[[1, 1, 1, 1, 1], [-1, -1, -1, -1, -1]],
                                                  [[-1, -1, -1, -1, -1], [1, 1, 1, 1, 1]]]
                                                 ).to(model.output_layer.weights.device)
                    weights.data.copy_(
                        torch.ones_like(weights.data).to(model.output_layer.weights.device)
                        * scale * negation_mask)
                else:
                    weights.data.copy_(torch.ones_like(weights.data).to(model.output_layer.weights.device) * scale * -1)

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
                    {'class1': [1, 0, 1, 0] * 1000,
                     'class2': [0, 1, 0, 1] * 1000}).values
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

        model = BanditNRNClassifier(
            target_names=['class1', 'class2'],
            feature_names=['feat1', 'feat2'],
            input_size=2,
            output_size=2,
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

    @fixture
    def data_loaders_cl1(self):
        class StaticDataset(Dataset):
            def __init__(self):
                super(StaticDataset, self).__init__()
                self.X = pd.DataFrame(
                    {'feature1': [1, 0, 1, 0] * 1000,
                     'feature2': [0, 1, 0, 1] * 1000}).values
                self.y = pd.DataFrame(
                    {'class1': [1, 0, 1, 0] * 1000}).values
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
    def boosted_model_cl1(self):

        torch.manual_seed(0)
        np.random.seed(0)

        model = BanditNRNClassifier(
            target_names=['class1'],
            feature_names=['feat1', 'feat2'],
            input_size=2,
            output_size=1,
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
    def boosted_trainer_cl1(self, boosted_model_cl1):
        return BoostedBanditNRNTrainer(
            boosted_model_cl1,
            torch.nn.BCELoss(),
            torch.optim.Adam(boosted_model_cl1.rn.parameters()),
            perform_prune_plateau_count=0,
            augment=None,
            partial_fit=True
        )

    @staticmethod
    def test__explain_sample(two_layer_model, one_layer_one_class_model, one_layer_one_class_model_negatives):

        x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        out = two_layer_model.explain_samples(
            x,
            quantile=1.0,
            target_names=['class1', 'class2'],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix='Sample was in the'
        )
        assert (out == '0: Sample was in the class1 because: '
                       '\n\n\nOR '
                       '\n\tAND '
                       '\n\t\tFeat2 >= 0.0'
                       '\n\t\tFeat3 >= 0.0'
                       '\n\tAND '
                       '\n\t\tFeat2 >= 0.0'
                       '\n\t\tFeat4 >= 0.0')

        x = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
        out = two_layer_model.explain_samples(
            x,
            quantile=1.0,
            target_names=['class1', 'class2'],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix='Sample was in the'
        )
        assert (out == '0: Sample was in the class2 because: '
                       '\n\n\nAND '
                       '\n\tAND '
                       '\n\t\tFeat1 >= 0.5'
                       '\n\t\tFeat3 >= 0.5'
                       '\n\tOR '
                       '\n\t\tAND '
                       '\n\t\t\tFeat1 >= 0.5'
                       '\n\t\t\tFeat3 >= 0.5'
                       '\n\t\tAND '
                       '\n\t\t\tFeat1 >= 0.5'
                       '\n\t\t\tFeat4 >= 0.5')

        x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        out = two_layer_model.explain_samples(
            x,
            quantile=1.0,
            target_names=['class1', 'class2'],
            explain_type='both',
            print_type='logical-natural',
            sample_explanation_prefix='The sample was in'
        )
        assert (out == '0: The sample was in class1 because: '
                       '\n\n\nAny of the following are true: '
                       '\n\tAll the following are true: '
                       '\n\t\tFeat2 >= 0.0'
                       '\n\t\tFeat3 >= 0.0'
                       '\n\tAll the following are true: '
                       '\n\t\tFeat2 >= 0.0'
                       '\n\t\tFeat4 >= 0.0')

        x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        out = one_layer_one_class_model.explain_samples(
            x,
            quantile=1.0,
            target_names=['class1'],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix='The sample was in'
        )
        assert (out == '0: The sample was in class1 because: '
                       '\n\n\nOR '
                       '\n\tAND '
                       '\n\t\tFeat1 >= 0.5'
                       '\n\t\tFeat4 >= 0.5'
                       '\n\tAND '
                       '\n\t\tFeat3 >= 0.5'
                       '\n\t\tFeat4 >= 0.5')

        x = torch.tensor([[0.8, 0.75, 0.4, 0.74]])
        out = one_layer_one_class_model.explain_samples(
            x,
            quantile=1.0,
            target_names=['class1'],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix='The sample was in'
        )
        assert (out == '0: The sample was in not class1 because: '
                       '\n\n\nOR '
                       '\n\tAND '
                       '\n\t\tNOT '
                       '\n\t\t\tFeat1 >= 0.8'
                       '\n\t\tNOT '
                       '\n\t\t\tFeat4 >= 0.74'
                       '\n\tAND '
                       '\n\t\tNOT '
                       '\n\t\t\tFeat3 >= 0.76'
                       '\n\t\tNOT '
                       '\n\t\t\tFeat4 >= 1.0')

        x = torch.tensor([[0.9, 0.9, 0.375, 0.1]])
        out = one_layer_one_class_model_negatives.explain_samples(
            x,
            quantile=1.0,
            target_names=['class1'],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix='The sample was in'
        )
        assert (out == '0: The sample was in not class1 because: '
                       '\n\n\nOR '
                       '\n\tAND '
                       '\n\t\tFeat1 >= 0.4'
                       '\n\t\tFeat4 >= 0.0'
                       '\n\tAND '
                       '\n\t\tFeat3 >= 0.375'
                       '\n\t\tFeat4 >= 0.1')

        x = torch.tensor([[0.9, 0.9, 0.3, 0.0]])
        out = one_layer_one_class_model_negatives.explain_samples(
            x,
            quantile=1.0,
            target_names=['class1'],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix='The sample was in'
        )
        assert (out == '0: The sample was in class1 because: '
                       '\n\n\nOR '
                       '\n\tNOT '
                       '\n\t\tFeat4 >= 0.0'
                       '\n\tAND '
                       '\n\t\tNOT '
                       '\n\t\t\tFeat3 >= 0.302'
                       '\n\t\tNOT '
                       '\n\t\t\tFeat4 >= 0.002')

    @staticmethod
    def test__explain(one_layer_model, two_layer_model, two_layer_model_negated):

        out = two_layer_model.explain(
            quantile=1.0,
            required_output_thresholds=torch.tensor(0.5),
            target_names=['class1', 'class2'],
            explain_type='both',
            print_type='logical',
            explanation_prefix='A sample is in'
        )
        assert (out == 'A sample is in class1 because: '
                       '\n\n\nOR '
                       '\n\tAND '
                       '\n\t\tFeat2 >= 0.688'
                       '\n\t\tFeat3 >= 0.688'
                       '\n\tAND '
                       '\n\t\tFeat2 >= 0.688'
                       '\n\t\tFeat4 >= 0.688'
                       '\n\nA sample is in class2 because: '
                       '\n\n\nAND '
                       '\n\tAND '
                       '\n\t\tFeat1 >= 0.805'
                       '\n\t\tFeat3 >= 0.805'
                       '\n\tOR '
                       '\n\t\tAND '
                       '\n\t\t\tFeat1 >= 0.805'
                       '\n\t\t\tFeat3 >= 0.805'
                       '\n\t\tAND '
                       '\n\t\t\tFeat1 >= 0.805'
                       '\n\t\t\tFeat4 >= 0.805')

        out = two_layer_model.explain(
            quantile=1.0,
            required_output_thresholds=torch.tensor(0.5),
            target_names=['class1', 'class2'],
            explain_type='both',
            print_type='logical-natural',
            explanation_prefix='A sample is in',
        )
        assert (out == 'A sample is in class1 because: '
                       '\n\n\nAny of the following are true: '
                       '\n\tAll the following are true: '
                       '\n\t\tFeat2 >= 0.688'
                       '\n\t\tFeat3 >= 0.688'
                       '\n\tAll the following are true: '
                       '\n\t\tFeat2 >= 0.688'
                       '\n\t\tFeat4 >= 0.688'
                       '\n\nA sample is in class2 because: '
                       '\n\n\nAll the following are true: '
                       '\n\tAll the following are true: '
                       '\n\t\tFeat1 >= 0.805'
                       '\n\t\tFeat3 >= 0.805'
                       '\n\tAny of the following are true: '
                       '\n\t\tAll the following are true: '
                       '\n\t\t\tFeat1 >= 0.805'
                       '\n\t\t\tFeat3 >= 0.805'
                       '\n\t\tAll the following are true: '
                       '\n\t\t\tFeat1 >= 0.805'
                       '\n\t\t\tFeat4 >= 0.805')

        torch.manual_seed(0)
        out = two_layer_model.explain(
            quantile=1.0,
            required_output_thresholds=torch.tensor(0.5),
            target_names=['class1', 'class2'],
            explain_type='both',
            print_type='natural',
            explanation_prefix='A sample is in',
        )
        assert (out == 'A sample is in class1 because: '
                       '\n\nIn order for this prediction to remain accurate, certain scenarios need to be fulfilled.  '
                       'The first scenario is as follows.  At least one of these requirements needs to be met.  '
                       'The first requirement is as follows. '
                       'feat2 greater than or equal to 0.688, and feat3 greater than or equal to 0.688.  '
                       'An additional requirement that could be met is the following.  '
                       'Feat2 greater than or equal to 0.688, and feat4 greater than or equal to 0.688.'
                       '\n\nThe next scenario that must be met is as follows.  '
                       'One or more of these requirements must be fulfilled.  The first requirement is as follows. '
                       'feat2 greater than or equal to 0.688, and feat3 greater than or equal to 0.688.  '
                       'An additional requirement that could be met is the following.  '
                       'Feat2 greater than or equal to 0.688, and feat4 greater than or equal to 0.688'
                       '\n\nA sample is in class2 because: \n\nFor the forecast to stay precise, '
                       'specific scenarios must be met.  The first scenario is as follows.  '
                       'One or more of these requirements must be fulfilled.  '
                       'The first requirement is as follows. feat1 greater than or equal to 0.805, '
                       'and feat3 greater than or equal to 0.805.  '
                       'An additional requirement that could be met is the following.  '
                       'Feat1 greater than or equal to 0.805, and feat4 greater than or equal to 0.805.'
                       '\n\nThe next scenario that must be met is as follows.  '
                       'Feat1 greater than or equal to 0.805, and feat3 greater than or equal to 0.805')

        out = two_layer_model_negated.explain(
            quantile=1.0,
            required_output_thresholds=torch.tensor(0.9),
            target_names=['class1', 'class2'],
            explain_type='both',
            print_type='logical',
            explanation_prefix='A sample is in'
        )
        assert out == ('A sample is in class1 because: '
                       '\n\n\nAND '
                       '\n\tNOT '
                       '\n\t\tOR '
                       '\n\t\t\tNOT '
                       '\n\t\t\t\tAND '
                       '\n\t\t\t\t\tFeat4 >= 0.512'
                       '\n\t\t\t\t\tNOT '
                       '\n\t\t\t\t\t\tFeat2 >= 0.488'
                       '\n\t\t\tAND '
                       '\n\t\t\t\tFeat3 >= 0.988'
                       '\n\t\t\t\tNOT '
                       '\n\t\t\t\t\tFeat2 >= 0.012'
                       '\n\tNOT '
                       '\n\t\tOR '
                       '\n\t\t\tNOT '
                       '\n\t\t\t\tAND '
                       '\n\t\t\t\t\tFeat3 >= 0.512'
                       '\n\t\t\t\t\tNOT '
                       '\n\t\t\t\t\t\tFeat2 >= 0.488'
                       '\n\t\t\tAND '
                       '\n\t\t\t\tFeat2 >= 0.988'
                       '\n\t\t\t\tNOT '
                       '\n\t\t\t\t\tFeat4 >= 0.012'
                       '\n\nA sample is in class2 because: '
                       '\n\n\nAND '
                       '\n\tFeat1 >= 0.752'
                       '\n\tNOT '
                       '\n\t\tFeat3 >= 0.248')

        out = two_layer_model_negated.explain(
            quantile=1.0,
            required_output_thresholds=torch.tensor(0.9),
            target_names=['class1', 'class2'],
            explain_type='both',
            print_type='logical-natural',
            explanation_prefix='A sample is in',
        )
        assert (out == 'A sample is in class1 because: '
                       '\n\n\nAll the following are true: '
                       '\n\tIt was not true that '
                       '\n\t\tAny of the following are true: '
                       '\n\t\t\tIt was not true that '
                       '\n\t\t\t\tAll the following are true: '
                       '\n\t\t\t\t\tFeat4 >= 0.512'
                       '\n\t\t\t\t\tIt was not true that '
                       '\n\t\t\t\t\t\tFeat2 >= 0.488'
                       '\n\t\t\tAll the following are true: '
                       '\n\t\t\t\tFeat3 >= 0.988'
                       '\n\t\t\t\tIt was not true that '
                       '\n\t\t\t\t\tFeat2 >= 0.012'
                       '\n\tIt was not true that '
                       '\n\t\tAny of the following are true: '
                       '\n\t\t\tIt was not true that '
                       '\n\t\t\t\tAll the following are true: '
                       '\n\t\t\t\t\tFeat3 >= 0.512'
                       '\n\t\t\t\t\tIt was not true that '
                       '\n\t\t\t\t\t\tFeat2 >= 0.488'
                       '\n\t\t\tAll the following are true: '
                       '\n\t\t\t\tFeat2 >= 0.988'
                       '\n\t\t\t\tIt was not true that '
                       '\n\t\t\t\t\tFeat4 >= 0.012'
                       '\n\nA sample is in class2 because: '
                       '\n\n\nAll the following are true: '
                       '\n\tFeat1 >= 0.752'
                       '\n\tIt was not true that '
                       '\n\t\tFeat3 >= 0.248')

        out = two_layer_model_negated.explain(
            quantile=1.0,
            required_output_thresholds=torch.tensor(0.9),
            target_names=['class1', 'class2'],
            explain_type='both',
            print_type='natural',
            explanation_prefix='A sample is in',
        )
        assert (out == 'A sample is in class1 because: '
                       '\n\nTo maintain accuracy in predictions, certain scenarios need to be fulfilled.  '
                       'The first scenario is as follows.  None of these requirements are fulfilled. '
                       'The first requirement is as follows.  '
                       'Feat2 greater than or equal to 0.488, or it was NOT true feat4 greater than or equal to 0.512.'
                       '  An additional requirement that must NOT be met is the following.  '
                       'It was NOT true feat2 greater than or equal to 0.012, and feat3 greater than or equal to 0.988.'
                       '\n\nThe next scenario that must be met is as follows.  '
                       'Not a single one of these requirements is met. The first requirement is as follows.  '
                       'Feat2 greater than or equal to 0.488, or it was NOT true feat3 greater than or equal to 0.512.'
                       '  An additional requirement that must NOT be met is the following.  '
                       'It was NOT true feat4 greater than or equal to 0.012, and feat2 greater than or equal to 0.988'
                       '\n\nA sample is in class2 because: '
                       '\n\nit was NOT true feat3 greater than or equal to 0.248, '
                       'and feat1 greater than or equal to 0.752')

        out = one_layer_model.explain(
            quantile=1.0,
            required_output_thresholds=torch.tensor(0.5),
            target_names=['class1', 'class2'],
            explain_type='both',
            print_type='logical',
            explanation_prefix='A sample is in'
        )
        assert (out == 'A sample is in class1 because: '
                       '\n\n\nAND '
                       '\n\tFeat2 >= 0.625'
                       '\n\tFeat4 >= 0.625'
                       '\n\nA sample is in class2 because: '
                       '\n\n\nOR '
                       '\n\tAND '
                       '\n\t\tFeat1 >= 0.781'
                       '\n\t\tFeat3 >= 0.781'
                       '\n\tAND '
                       '\n\t\tFeat1 >= 0.781'
                       '\n\t\tFeat4 >= 0.781')

        out = one_layer_model.explain(
            quantile=1.0,
            required_output_thresholds=torch.tensor(0.5),
            target_names=['class1', 'class2'],
            explain_type='both',
            print_type='logical-natural',
            explanation_prefix='A sample is in'
        )
        assert (out == 'A sample is in class1 because: '
                       '\n\n\nAll the following are true: '
                       '\n\tFeat2 >= 0.625'
                       '\n\tFeat4 >= 0.625'
                       '\n\nA sample is in class2 because: '
                       '\n\n\nAny of the following are true: '
                       '\n\tAll the following are true: '
                       '\n\t\tFeat1 >= 0.781'
                       '\n\t\tFeat3 >= 0.781'
                       '\n\tAll the following are true: '
                       '\n\t\tFeat1 >= 0.781'
                       '\n\t\tFeat4 >= 0.781')

    @staticmethod
    @patch('sys.stdout', new_callable=StringIO)
    def test__print_sample(stdout, one_layer_model):

        x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        one_layer_model.print_samples(
            x,
            quantile=1.0,
            target_names=['class1', 'class2'],
            explain_type='both',
            print_type='logical'
        )
        output = stdout.getvalue()
        assert (output == "REASONING NETWORK MODEL FOR: class1"
                          "\nLogic at depth 2: feat2 >= 0.0, feat4 >= 0.0"
                          "\noutput: tensor([1., 1.])"
                          "\n\nLogic at depth 1: ['AND(feat2 >= 0.0, feat4 >= 0.0)']"
                          "\nweights: tensor([1., 1.])"
                          "\noutput: 1.0"
                          "\nrequired_threshold: 0.0"
                          "\n\nLogic at depth 2: feat2 >= 0.0, feat4 >= 0.0"
                          "\noutput: tensor([1., 1.])"
                          "\n\nLogic at depth 1: ['AND(feat2 >= 0.0, feat4 >= 0.0)']"
                          "\nweights: tensor([1., 1.])"
                          "\noutput: 1.0"
                          "\nrequired_threshold: 0.0"
                          "\n\nLogic at depth 0: ['AND(feat2 >= 0.0, feat4 >= 0.0)']"
                          "\nweights: tensor([1., 1.])"
                          "\noutput: 1.0"
                          "\nrequired_threshold: 0.9900000691413879\n\n" )

    @staticmethod
    @patch('sys.stdout', new_callable=StringIO)
    def test__print(stdout):

        torch.manual_seed(0)
        np.random.seed(0)

        module = BanditNRNClassifier(
            target_names=['class1', 'class2'],
            feature_names=['feat1', 'feat1', 'feat3', 'feat4'],
            input_size=4,
            output_size=2,
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
            target_names=['class1', 'class2'],
            explain_type='both',
            print_type='logical'
        )
        output = stdout.getvalue()
        assert (output == "REASONING NETWORK MODEL FOR: class1"
                          "\nLogic at depth 2: feat1 >= 0.625, feat4 >= 0.625"
                          "\noutput: tensor([0.6250, 0.6250])"
                          "\n\nLogic at depth 1: ['AND(feat1 >= 0.625, feat4 >= 0.625)']"
                          "\nweights: tensor([1., 1.])"
                          "\noutput: 0.25"
                          "\nrequired_threshold: 0.25"
                          "\n\nLogic at depth 2: feat3 >= 0.625, feat4 >= 0.625"
                          "\noutput: tensor([0.6250, 0.6250])"
                          "\n\nLogic at depth 1: ['AND(feat3 >= 0.625, feat4 >= 0.625)']"
                          "\nweights: tensor([1., 1.])"
                          "\noutput: 0.25"
                          "\nrequired_threshold: 0.25"
                          "\n\nLogic at depth 0: ['OR(AND(feat1 >= 0.625, feat4 >= 0.625), AND(feat3 >= 0.625, feat4 >= 0.625))']"
                          "\nweights: tensor([1., 1.])"
                          "\noutput: 0.5"
                          "\nrequired_threshold: 0.5"
                          "\n\nREASONING NETWORK MODEL FOR: class2"
                          "\nLogic at depth 2: feat1 >= 0.625"
                          "\noutput: tensor([0.6250, 0.6250])"
                          "\n\nLogic at depth 1: ['feat1 >= 0.625']"
                          "\nweights: tensor([1., 1.])"
                          "\noutput: 0.25"
                          "\nrequired_threshold: 0.25"
                          "\n\nLogic at depth 2: feat1 >= 0.625, feat4 >= 0.625"
                          "\noutput: tensor([0.6250, 0.6250])"
                          "\n\nLogic at depth 1: ['AND(feat1 >= 0.625, feat4 >= 0.625)']"
                          "\nweights: tensor([1., 1.])"
                          "\noutput: 0.25"
                          "\nrequired_threshold: 0.25"
                          "\n\nLogic at depth 0: ['OR(AND(feat1 >= 0.625, feat4 >= 0.625), feat1 >= 0.625)']"
                          "\nweights: tensor([1., 1.])"
                          "\noutput: 0.5"
                          "\nrequired_threshold: 0.5\n\n")

    @staticmethod
    def test__explain_sample_boosted_model(data_loaders, boosted_trainer, data_loaders_cl1, boosted_trainer_cl1):
        train_dl, val_dl = data_loaders
        boosted_trainer.boost(train_dl)

        # RRN and Boosted RRN have the same multi-label predictions
        x = torch.tensor([[1.0, 0.0]])
        out = boosted_trainer.model.explain_samples(
            x,
            quantile=1.0,
            target_names=['class1', 'class2'],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix='The sample was in',
            decision_boundary=0.4
        )
        assert (out == '0: The sample was in class1 because: \n\n\nAND \n\tFeat1 >= 0.98\n\tFeat2 >= 0.0')

        # RRN and Boosted RRN have different multi-label predictions
        x = torch.tensor([[1.0, 0.0]])
        out = boosted_trainer.model.explain_samples(
            x,
            quantile=1.0,
            target_names=['class1', 'class2'],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix='The sample was in',
            decision_boundary=0.6
        )
        assert (out == '0: The sample was in class1 because: \n\n\nAND \n\tFeat1 >= 0.98\n\tFeat2 >= 0.0')

        # RRN and Boosted RRN have the same multi-class predictions
        x = torch.tensor([[1.0, 0.0]])
        boosted_trainer.model.rn.output_layer.weights[0].data.copy_(torch.tensor([[0.6000], [0.6000]]))
        boosted_trainer.model.rn.output_layer.weights[1].data.copy_(torch.tensor([[0.4000], [0.4000]]))
        out = boosted_trainer.model.explain_samples(
            x,
            quantile=1.0,
            target_names=['class1', 'class2'],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix='The sample was in'
        )
        assert (out == '0: The sample was in class1 because: \n\n\nAND \n\tFeat1 >= 0.98\n\tFeat2 >= 0.0')

        # RRN and Boosted RRN have different multi-class predictions
        x = torch.tensor([[0.0, 1.0]])
        boosted_trainer.model.rn.output_layer.weights[0].data.copy_(torch.tensor([[0.6000], [0.6000]]))
        boosted_trainer.model.rn.output_layer.weights[1].data.copy_(torch.tensor([[0.4000], [0.4000]]))
        out = boosted_trainer.model.explain_samples(
            x,
            quantile=1.0,
            target_names=['class1', 'class2'],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix='The sample was in'
        )
        assert (out == '0: The sample was in class2 because: \n\n\nAND \n\tFeat1 >= 0.0\n\tFeat2 >= 0.98')

        train_dl, val_dl = data_loaders_cl1
        boosted_trainer_cl1.boost(train_dl)

        # RRN and Boosted RRN have different binary-class predictions
        x = torch.tensor([[1.0, 0.0]])
        boosted_trainer_cl1.model.rn.output_layer.weights[0].data.copy_(torch.tensor([[0.6000], [0.6000]]))
        out = boosted_trainer_cl1.model.explain_samples(
            x,
            quantile=1.0,
            target_names=['class1'],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix='The sample was in'
        )
        assert (out == '0: The sample was in class1 because: \n\n\nAND \n\tFeat1 >= 0.98\n\tFeat2 >= 0.0')

        # RRN and Boosted RRN have different binary-class predictions
        # TODO: This is a strange interpretation of the result but in a way its accurate because it tells you the
        #  direction to make the prediction more negative
        x = torch.tensor([[0.0, 1.0]])
        boosted_trainer_cl1.model.rn.output_layer.weights[0].data.copy_(torch.tensor([[0.6000], [0.6000]]))
        out = boosted_trainer_cl1.model.explain_samples(
            x,
            quantile=1.0,
            target_names=['class1'],
            explain_type='both',
            print_type='logical',
            sample_explanation_prefix='The sample was in'
        )
        assert (out == '0: The sample was in not class1 because: \n\n\nNOT \n\tFeat1 >= 0.02')
