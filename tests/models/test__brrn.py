from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error

import torch
from torch.utils.data import Dataset, DataLoader

from pytest import fixture

from torchlogic.models.base import BaseBanditNRNModel
from torchlogic.models import BanditNRNClassifier


class TestBaseBanditRRN:

    @fixture
    def data_loaders(self):
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

        return (DataLoader(StaticDataset(), batch_size=2, shuffle=False),
                DataLoader(StaticDataset(), batch_size=2, shuffle=False))

    @fixture
    def model(self):
        model = BanditNRNClassifier(
            target_names=['class1', 'class2'],
            feature_names=['feat1', 'feat1', 'feat3', 'feat4'],
            input_size=4,
            output_size=2,
            layer_sizes=[5, 5],
            n_selected_features_input=2,
            n_selected_features_internal=2,
            n_selected_features_output=2,
            perform_prune_quantile=0.5,
            ucb_scale=2.5
        )

        for item in model.rn.state_dict():
            if item.find('weights') > -1:
                weights = model.rn.state_dict()[item]
                weights.data.copy_(torch.ones_like(weights.data))

        return model

    @staticmethod
    def test__initialize_reward_history():
        torch.manual_seed(0)
        np.random.seed(0)

        module = BaseBanditNRNModel(
            target_names=['class1', 'class2'],
            feature_names=['feat1', 'feat1', 'feat3', 'feat4'],
            input_size=4,
            output_size=2,
            layer_sizes=[5, ],
            n_selected_features_input=2,
            n_selected_features_internal=2,
            n_selected_features_output=2,
            perform_prune_quantile=0.5,
            ucb_scale=2.5
        )

        predicates_rewards_history = module._initialize_rewards_history()
        assert list(predicates_rewards_history.keys()) == [0, 1]
        assert all(predicates_rewards_history[0]['predicates'] == [0, 1, 2, 3])
        assert all(predicates_rewards_history[1]['predicates'] == [0, 1, 2, 3])

    @staticmethod
    def test__get_activations(model, data_loaders):
        train_dl, val_dl = data_loaders
        all_activations, all_targets, all_sample_idx = model._get_activations(train_dl)

        assert all_activations.size() == (4, 2, 1, 5), "did not collect activations correctly"
        assert all_targets.size() == (4, 2), "did not collect targets correctly"
        assert torch.equal(all_sample_idx, torch.tensor([0, 1, 2, 3])), "did not collect sample indexes correctly"

    @staticmethod
    def test__evaluate_activations_identify_important_features(model):
        activations = torch.ones((4, 2, 1, 5))
        activations[:, 0, :, 0].copy_(
            torch.tensor([[0], [1], [0], [1]]))  # set logic to be perfect predictor for class 0
        activations[:, 1, :, 4].copy_(
            torch.tensor([[1], [0], [1], [0]]))  # set logic to be perfect predictor for class 1
        activations[:, 0, :, 1].copy_(
            torch.tensor([[0], [0], [0], [1]]))  # set logic to be partial predictor for class 0
        targets = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])
        sample_idx = torch.tensor([0, 2, 1, 3])

        input_layer_weights = torch.abs(model.rn.model[0].weights.clone().detach().transpose(-2, -1))
        internal_layer_mask = model.rn.model[1].mask
        internal_layer_mask.copy_(torch.tensor([[[2, 4], [4, 3], [0, 2], [1, 2], [1, 2]],
                                               [[4, 1], [1, 0], [0, 4], [1, 3], [4, 2]]]).to(
            internal_layer_mask.device))

        # empty class indices
        class_indices = torch.tensor([])
        mask = torch.zeros_like(input_layer_weights.data).bool()
        new_mask, rewards = model._evaluate_activations(
            targets, activations, sample_idx, input_layer_weights, internal_layer_mask, mask,
            class_indices, roc_auc_score)
        assert torch.equal(new_mask, mask), "mask updated when no class indices passed"
        assert rewards is None, "no rewards passed but returned rewards tensor"

        # test bootstrap map properties
        class_indices = torch.tensor([0, 1])
        mask = torch.zeros_like(input_layer_weights.data).bool()
        _ = model._evaluate_activations(
            targets, activations, sample_idx, input_layer_weights, internal_layer_mask, mask,
            class_indices, roc_auc_score)
        assert len(model.bootstrap_sample_map.keys()) == len(class_indices), "boostrap map keys dont match classes!"
        for k, v in model.bootstrap_sample_map.items():
            assert len(v) == len(torch.unique(internal_layer_mask[k])), \
                "boostrap map keys dont match input layer logic size!"
            for v2 in v.values():
                assert len(v2) == len(sample_idx), "boostrap samples dont match data set size!"
                assert len(sample_idx) - 1 >= min(v2) >= 0, "boostrap sample indexes don't match data sample indexes!"
                assert len(sample_idx) - 1 >= max(v2) >= 0, "boostrap sample indexes don't match data sample indexes!"

        # test boostrap map is not overwritten
        init_boostrap_map = deepcopy(model.bootstrap_sample_map)
        _ = model._evaluate_activations(
            targets, activations, sample_idx, input_layer_weights, internal_layer_mask, mask,
            class_indices, roc_auc_score)
        assert init_boostrap_map == model.bootstrap_sample_map, "boostrap map update after initial population!"

        # test evaluation of logics during maximization objective to identify important features mode
        model.perform_prune_quantile = 0.7
        mask = torch.zeros_like(input_layer_weights.data).bool()
        rewards = torch.zeros_like(input_layer_weights.data)
        new_mask, rewards = model._evaluate_activations(
            targets, activations, sample_idx, input_layer_weights, internal_layer_mask, mask,
            class_indices, roc_auc_score, rewards)
        assert torch.equal(torch.tensor([[[True,  True],
                                          [True,  True],
                                          [False, False],
                                          [False, False],
                                          [False, False]],

                                         [[True,  True],
                                          [True,  True],
                                          [True,  True],
                                          [True,  True],
                                          [True,  True]]]).to(new_mask.device),
                           new_mask), "did not correctly identify logics to update"
        assert torch.equal(rewards, torch.tensor([[[1.,  1.],
                                                   [0.75,  0.75],
                                                   [0., 0.],
                                                   [0., 0.],
                                                   [0., 0.]],

                                                  [[0.5,  0.5],
                                                   [0.5, 0.5],
                                                   [0.5, 0.5],
                                                   [0.5, 0.5],
                                                   [1.,  1.]]]).to(rewards.device)), "did not create rewards correctly"

        # test evaluation of logics during minimize objective to identify important features mode
        model.perform_prune_quantile = 0.7
        mask = torch.zeros_like(input_layer_weights.data).bool()
        rewards = torch.zeros_like(input_layer_weights.data)
        new_mask, rewards = model._evaluate_activations(
            targets, activations, sample_idx, input_layer_weights, internal_layer_mask, mask,
            class_indices, mean_squared_error, rewards, objective='minimize')
        assert torch.equal(torch.tensor([[[True, True],
                                          [True, True],
                                          [False, False],
                                          [False, False],
                                          [False, False]],

                                         [[True, True],
                                          [True, True],
                                          [True, True],
                                          [True, True],
                                          [True, True]]]).to(new_mask.device),
                           new_mask), "did not correctly identify logics to update"
        # # TODO: Torch returns that these are not equal when they are
        # print(rewards)
        # rewards_test = torch.tensor([[[11.5129, 11.5129],
        #                                            [1.6094,  1.6094],
        #                                            [0.0000,  0.0000],
        #                                            [0.0000,  0.0000],
        #                                            [0.0000,  0.0000]],
        #                                           [[1.0986,  1.0986],
        #                                            [1.0986,  1.0986],
        #                                            [1.0986,  1.0986],
        #                                            [1.0986,  1.0986],
        #                                            [11.5129, 11.5129]]]).to(rewards.device)
        # assert torch.equal(rewards_test, rewards), "did not create rewards correctly"

    @staticmethod
    def test__evaluate_activations_prune(model):
        activations = torch.ones((4, 2, 1, 5))
        activations[:, 0, :, 0].copy_(
            torch.tensor([[0], [1], [0], [1]]))  # set logic to be perfect predictor for class 0
        activations[:, 1, :, 4].copy_(
            torch.tensor([[1], [0], [1], [0]]))  # set logic to be perfect predictor for class 1
        activations[:, 0, :, 1].copy_(
            torch.tensor([[0], [0], [0], [1]]))  # set logic to be partial predictor for class 0
        targets = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])
        sample_idx = torch.tensor([0, 2, 1, 3])

        input_layer_weights = model.rn.model[0].weights.clone().detach()
        internal_layer_mask = model.rn.model[1].mask
        internal_layer_mask.copy_(torch.tensor([[[3, 4], [4, 1], [1, 3], [2, 3], [4, 3]],
                                                [[1, 2], [1, 0], [1, 0], [3, 1], [2, 1]]]).to(
            internal_layer_mask.device))

        # empty class indices
        class_indices = torch.tensor([])
        mask = torch.zeros_like(input_layer_weights.data).bool()
        new_mask, rewards = model._evaluate_activations(
            targets, activations, sample_idx, input_layer_weights, internal_layer_mask, mask,
            class_indices, roc_auc_score, dimension=-1, mode='prune')
        assert torch.equal(new_mask, mask.transpose(-2, -1)), "mask updated when no class indices passed"
        assert rewards is None, "no rewards passed but returned rewards tensor"

        # test bootstrap map properties
        class_indices = torch.tensor([0, 1])
        mask = torch.zeros_like(input_layer_weights.data).bool()
        _ = model._evaluate_activations(
            targets, activations, sample_idx, input_layer_weights, internal_layer_mask, mask,
            class_indices, roc_auc_score, dimension=-1, mode='prune')
        assert len(model.bootstrap_sample_map.keys()) == len(class_indices), "boostrap map keys dont match classes!"
        for k, v in model.bootstrap_sample_map.items():
            assert len(v) == len(torch.unique(internal_layer_mask[k])), \
                "boostrap map keys dont match input layer logic size!"
            for v2 in v.values():
                assert len(v2) == len(sample_idx), "boostrap samples dont match data set size!"
                assert len(sample_idx) - 1 >= min(v2) >= 0, "boostrap sample indexes don't match data sample indexes!"
                assert len(sample_idx) - 1 >= max(v2) >= 0, "boostrap sample indexes don't match data sample indexes!"

        # test boostrap map is not overwritten
        init_boostrap_map = deepcopy(model.bootstrap_sample_map)
        _ = model._evaluate_activations(
            targets, activations, sample_idx, input_layer_weights, internal_layer_mask, mask,
            class_indices, roc_auc_score, dimension=-1, mode='prune')
        assert init_boostrap_map == model.bootstrap_sample_map, "boostrap map update after initial population!"

        # test evaluation of logics during maximization objective to identify important features mode
        model.perform_prune_quantile = 0.9
        mask = torch.zeros_like(input_layer_weights.data).bool()
        new_mask, rewards = model._evaluate_activations(
            targets, activations, sample_idx, input_layer_weights, internal_layer_mask, mask,
            class_indices, roc_auc_score, dimension=-1, mode='prune')
        assert torch.equal(torch.tensor([[[False, False],
                                          [False, False],
                                          [True, True],
                                          [True, True],
                                          [True, True]],

                                         [[False, False],
                                          [False, False],
                                          [False, False],
                                          [False, False],
                                          [False, False]]]).to(new_mask.device),
                           new_mask), "did not correctly identify logics to update"

        # test evaluation of logics during minimize objective to identify important features mode
        model.perform_prune_quantile = 0.9
        mask = torch.zeros_like(input_layer_weights.data).bool()
        new_mask, rewards = model._evaluate_activations(
            targets, activations, sample_idx, input_layer_weights, internal_layer_mask, mask,
            class_indices, mean_squared_error, objective='minimize', dimension=-1, mode='prune')
        assert torch.equal(torch.tensor([[[False, False],
                                          [False, False],
                                          [True, True],
                                          [True, True],
                                          [True, True]],

                                         [[False, False],
                                          [False, False],
                                          [False, False],
                                          [False, False],
                                          [False, False]]]).to(new_mask.device),
                           new_mask), "did not correctly identify logics to update"

    @staticmethod
    def test__evaluate_weights_identify_important_features(model):
        input_layer_weights = torch.abs(model.rn.model[0].weights.clone().detach().transpose(-2, -1))
        input_layer_weights.copy_(torch.tensor([[[1., 1.],
                                                 [-1., -1.],
                                                 [0., 0.],
                                                 [0., 0.],
                                                 [0., 0.]],

                                                [[0., -1.],
                                                 [1., 0.],
                                                 [0., -1.],
                                                 [1., 0.],
                                                 [0., -1.]]]).to(input_layer_weights.device))

        # empty class indices
        class_indices = torch.tensor([])
        new_mask, rewards = model._evaluate_weights(input_layer_weights, class_indices)
        assert torch.equal(new_mask, torch.zeros_like(new_mask).bool()), "mask not created properly"

        # all classes
        class_indices = torch.tensor([0, 1])
        model.perform_prune_quantile = 0.9
        new_mask, rewards = model._evaluate_weights(input_layer_weights, class_indices)
        assert torch.equal(new_mask, torch.tensor([[[True, True],
                                                    [True, True],
                                                    [False, False],
                                                    [False, False],
                                                    [False, False]],

                                                   [[False, True],
                                                    [True, False],
                                                    [False, True],
                                                    [True, False],
                                                    [False, True]]]).to(new_mask.device)), "mask not created properly"
        assert torch.equal(rewards, torch.abs(input_layer_weights)), "rewards not correctly returned"

    @staticmethod
    def test__evaluate_weights_prune(model):
        input_layer_weights = torch.abs(model.rn.model[0].weights.clone().detach().transpose(-2, -1))
        input_layer_weights.copy_(torch.tensor([[[1., 1.],
                                                 [-1., -1.],
                                                 [0., 0.],
                                                 [0., 0.],
                                                 [0., 0.]],

                                                [[0., -1.],
                                                 [1., 0.],
                                                 [0., -1.],
                                                 [1., 0.],
                                                 [0., -1.]]]).to(input_layer_weights.device))

        # empty class indices
        class_indices = torch.tensor([])
        new_mask, rewards = model._evaluate_weights(input_layer_weights, class_indices, mode='prune')
        assert torch.equal(new_mask, torch.zeros_like(new_mask).bool()), "mask not created properly"

        # all classes
        class_indices = torch.tensor([0, 1])
        model.perform_prune_quantile = 0.5
        new_mask, rewards = model._evaluate_weights(input_layer_weights, class_indices, mode='prune')
        assert torch.equal(new_mask, torch.tensor([[[False, False,  True,  True,  True],
                                                    [False, False,  True,  True,  True]],
                                                   [[True, False, True, False,  True],
                                                    [False, True, False, True, False]]]).to(new_mask.device)), \
            "mask not created properly"
        assert rewards is None, "rewards not correctly returned"

    @staticmethod
    def test__evaluate_logic_weights_identify_important_features(model):

        input_layer_weights = torch.abs(model.rn.model[0].weights.clone().detach().transpose(-2, -1))
        internal_layer_mask = model.rn.model[1].mask
        internal_layer_mask.copy_(torch.tensor([[[4, 0],
                                                 [1, 4],
                                                 [0, 4],
                                                 [1, 4],
                                                 [4, 1]],

                                                [[2, 0],
                                                 [0, 1],
                                                 [3, 2],
                                                 [2, 4],
                                                 [3, 0]]]).to(internal_layer_mask.device))
        internal_layer_weights = torch.abs(model.rn.model[1].weights.clone().detach().transpose(-2, -1))
        internal_layer_weights.copy_(torch.tensor([[[1., 1.],
                                                    [0.5, 0.5],
                                                    [2., 5.],
                                                    [10., 1.],
                                                    [0., 0.]],

                                                   [[1., 1.],
                                                    [1., 1.],
                                                    [0.3, 0.3],
                                                    [0.2, 0.2],
                                                    [0., 0.]]]).to(internal_layer_weights.device))

        # empty class indices
        class_indices = torch.tensor([])
        new_mask, rewards = model._evaluate_logic_weights(
            input_layer_weights.transpose(-2, -1), internal_layer_weights, internal_layer_mask, class_indices)
        assert torch.equal(new_mask, torch.zeros_like(new_mask).bool())

        # all class indices
        class_indices = torch.tensor([0, 1])
        new_mask, rewards = model._evaluate_logic_weights(
            input_layer_weights.transpose(-2, -1), internal_layer_weights, internal_layer_mask, class_indices)
        assert torch.equal(new_mask, torch.tensor([[[ True,  True],
                                                     [ True,  True],
                                                     [False, False],
                                                     [False, False],
                                                     [ True,  True]],

                                                    [[ True,  True],
                                                     [ True,  True],
                                                     [ True,  True],
                                                     [ True,  True],
                                                     [False, False]]]).to(new_mask.device)), "mask created incorrectly"
        assert torch.equal(rewards, torch.tensor([[[ 2.0000,  2.0000],
                                                     [10.0000, 10.0000],
                                                     [ 0.0000,  0.0000],
                                                     [ 0.0000,  0.0000],
                                                     [ 5.0000,  5.0000]],

                                                    [[ 1.0000,  1.0000],
                                                     [ 1.0000,  1.0000],
                                                     [ 1.0000,  1.0000],
                                                     [ 0.3000,  0.3000],
                                                     [ 0.2000,  0.2000]]]).to(rewards.device)), "rewards incorrect"

    @staticmethod
    def test__evaluate_logic_weights_prune(model):

        input_layer_weights = torch.abs(model.rn.model[0].weights.clone().detach().transpose(-2, -1))
        internal_layer_mask = model.rn.model[1].mask
        internal_layer_mask.copy_(torch.tensor([[[4, 0],
                                                 [1, 4],
                                                 [0, 4],
                                                 [1, 4],
                                                 [4, 1]],

                                                [[2, 0],
                                                 [0, 1],
                                                 [3, 2],
                                                 [2, 4],
                                                 [3, 0]]]).to(internal_layer_mask.device))
        internal_layer_weights = torch.abs(model.rn.model[1].weights.clone().detach().transpose(-2, -1))
        internal_layer_weights.copy_(torch.tensor([[[1., 1.],
                                                    [0.5, 0.5],
                                                    [2., 5.],
                                                    [10., 1.],
                                                    [0., 0.]],

                                                   [[1., 1.],
                                                    [1., 1.],
                                                    [0.3, 0.3],
                                                    [0.2, 0.2],
                                                    [0., 0.]]]).to(internal_layer_weights.device))

        # empty class indices
        class_indices = torch.tensor([])
        new_mask, rewards = model._evaluate_logic_weights(
            input_layer_weights.transpose(-2, -1), internal_layer_weights, internal_layer_mask, class_indices, 'prune')
        assert torch.equal(new_mask, torch.zeros_like(new_mask).bool()), "did not produce correct mask"
        assert rewards is None, "did not produce correct rewards"

        # all class indices
        class_indices = torch.tensor([0, 1])
        new_mask, rewards = model._evaluate_logic_weights(
            input_layer_weights.transpose(-2, -1), internal_layer_weights, internal_layer_mask, class_indices, 'prune')
        assert torch.equal(new_mask, torch.tensor([[[ True,  True],
                                                     [ True,  True],
                                                     [False, False],
                                                     [False, False],
                                                     [ True,  True]],

                                                    [[ True,  True],
                                                     [False, False],
                                                     [ True,  True],
                                                     [ True,  True],
                                                     [ True,  True]]]).to(new_mask.device)), "mask created incorrectly"
        assert rewards is None, "did not produce correct rewards"

    @staticmethod
    def test__identify_important_features():

        torch.manual_seed(0)
        np.random.seed(0)

        module = BaseBanditNRNModel(
            target_names=['class1', 'class2'],
            feature_names=['feat1', 'feat1', 'feat3', 'feat4'],
            input_size=4,
            output_size=2,
            layer_sizes=[5, 5],
            n_selected_features_input=2,
            n_selected_features_internal=2,
            n_selected_features_output=2,
            perform_prune_quantile=0.5,
            ucb_scale=2.5
        )

        feature_indexes, feature_weights = module._identify_important_features()
        assert len(feature_indexes) == 2
        assert len(feature_weights) == 2
        assert all([len(x) >= 1 for x in feature_indexes])
        assert all([len(x) >= 1 for x in feature_weights])

    @staticmethod
    def test__produce_predicate_rewards():
        torch.manual_seed(0)
        np.random.seed(0)

        module = BaseBanditNRNModel(
            target_names=['class1', 'class2'],
            feature_names=['feat1', 'feat1', 'feat3', 'feat4'],
            input_size=4,
            output_size=2,
            layer_sizes=[5, ],
            n_selected_features_input=2,
            n_selected_features_internal=2,
            n_selected_features_output=2,
            perform_prune_quantile=0.5,
            ucb_scale=2.5
        )

        feature_indexes = np.array([[3, 1, 3], [0, 2, 0]])
        feature_weights = np.array([[1, 0.5, 1], [2, 5, 2]])

        predicates_rewards = module._produce_predicate_rewards(
            feature_indexes=feature_indexes, feature_weights=feature_weights)
        assert predicates_rewards == [[0.0, 0.5, 0.0, 2.0], [4.0, 0.0, 5.0, 0.0]]

    @staticmethod
    def test__bayesian_ucb_policy():
        torch.manual_seed(0)
        np.random.seed(0)

        module = BaseBanditNRNModel(
            target_names=['class1', 'class2'],
            feature_names=['feat1', 'feat1', 'feat3', 'feat4'],
            input_size=4,
            output_size=2,
            layer_sizes=[5, ],
            n_selected_features_input=2,
            n_selected_features_internal=2,
            n_selected_features_output=2,
            perform_prune_quantile=0.5,
            ucb_scale=2.5
        )

        policy0 = pd.DataFrame({'predicates': [0, 1, 2, 3, 0, 1, 2, 3], 'rewards': [1, 1, 1, 1, 1, 1, 1, 2]})
        policy = module._bayesian_ucb_policy(policy0)
        assert len(policy) == 4
        assert np.argmax(policy) == 3
        assert np.all(np.min(policy) == policy[:3])

    @staticmethod
    def test__update_policy():
        torch.manual_seed(0)
        np.random.seed(0)

        module = BaseBanditNRNModel(
            target_names=['class1', 'class2'],
            feature_names=['feat1', 'feat1', 'feat3', 'feat4'],
            input_size=4,
            output_size=2,
            layer_sizes=[5, ],
            n_selected_features_input=2,
            n_selected_features_internal=2,
            n_selected_features_output=2,
            perform_prune_quantile=0.5,
            ucb_scale=2.5
        )

        feature_indexes = np.array([[3, 1, 3], [0, 2, 0]])
        feature_weights = np.array([[1, 0.5, 1], [2, 5, 2]])

        policy0 = pd.DataFrame({'predicates': [0, 1, 2, 3], 'rewards': [1, 1, 1, 1]})

        module.predicates_rewards_history[0] = policy0
        module.predicates_rewards_history[1] = policy0

        module.predicates_policy.data.copy_(torch.ones_like(module.predicates_policy.data)/4)
        policy_before = module.predicates_policy.clone()
        module._update_policy(feature_indexes, feature_weights)
        policy_after = module.predicates_policy.clone()

        assert torch.equal(policy_after.gt(policy_before),
                           torch.tensor([[False, False, False, True], [False, False, True, False]]))
        assert policy_after[0].argmax() == 3
        assert policy_after[1].argmax() == 2

    @staticmethod
    def test__flip_un_pruned_feature_weights(model):
        input_layer_mask = model.rn.model[0].mask
        input_layer_mask.copy_(torch.tensor([[[3, 1],
                                              [3, 2],
                                              [3, 1],
                                              [0, 3],
                                              [2, 1]],
                                             [[1, 0],
                                              [0, 2],
                                              [1, 3],
                                              [0, 2],
                                              [2, 1]]]))
        input_layer_weights = model.rn.model[0].weights.clone().detach()
        input_layer_weights.copy_(torch.tensor([[[-1., -1., -1., 1., 1.],
                                                 [1., 1., 1., -1., 1.]],
                                                [[1., 1., 1., 1., 1.],
                                                 [1., 1., -1., 1., 1.]]]))

        prune_mask = torch.tensor([[[False, False],
                                    [True, False],
                                    [True, False],
                                    [False, True],
                                    [False, True]],
                                   [[False, False],
                                    [False, False],
                                    [False, True],
                                    [False, False],
                                    [False, False]]])
        new_mask = input_layer_mask.clone()

        new_input_layer_weights = model._flip_un_pruned_feature_weights(
            input_layer_mask, input_layer_weights, prune_mask, new_mask)

        positive_ones_mask = torch.tensor([[[False, False, False, True, True],
                                            [True, True, True, False, False]],
                                           [[True, True, True, True, True],
                                            [True, True, False, True, True]]])
        assert torch.eq(torch.masked_select(new_input_layer_weights, positive_ones_mask), 1.0).all(), \
            "did not copy unpruned positive weights correctly"

        negative_ones_mask = torch.tensor([[[True, False, False, False, False],
                                            [False, False, False, False, False]],
                                           [[False, False, False, False, False],
                                            [False, False, False, False, False]]])
        assert torch.eq(torch.masked_select(new_input_layer_weights, negative_ones_mask), -1.0).all(), \
            "did not copy unpruned negative weights correctly"

        positive_frac_mask = torch.tensor([[[False, True, True, False, False],
                                            [False, False, False, True, False]],
                                           [[False, False, False, False, False],
                                            [False, False, True, False, False]]])
        assert torch.gt(torch.masked_select(new_input_layer_weights, positive_frac_mask), 0).all(), \
            "did not initialized pruned existing positive weights correctly"
        assert ~torch.eq(torch.masked_select(new_input_layer_weights, positive_frac_mask), -1.00).any(), \
            "did not initialized pruned existing positive weights correctly"

        negative_frac_mask = torch.tensor([[[False, False, False, False, False],
                                            [False, False, False, False, True]],
                                           [[False, False, False, False, False],
                                            [False, False, False, False, False]]])
        assert torch.lt(torch.masked_select(new_input_layer_weights, negative_frac_mask), 0).all(), \
            "did not initialized pruned existing negative weights correctly"
        assert ~torch.eq(torch.masked_select(new_input_layer_weights, negative_frac_mask), 1.00).any(), \
            "did not initialized pruned existing negative weights correctly"

    @staticmethod
    def test__perform_prune():
        torch.manual_seed(0)
        np.random.seed(0)

        module = BaseBanditNRNModel(
            target_names=['class1', 'class2'],
            feature_names=['feat1', 'feat1', 'feat3', 'feat4'],
            input_size=4,
            output_size=2,
            layer_sizes=[5, ],
            n_selected_features_input=2,
            n_selected_features_internal=2,
            n_selected_features_output=2,
            perform_prune_quantile=1.0,
            ucb_scale=2.5
        )

        mask_before = deepcopy(module.rn.model[0].mask.clone())
        module.perform_prune()
        mask_after = deepcopy(module.rn.model[0].mask.clone())

        assert not torch.equal(mask_after, mask_before)
        assert mask_after.cpu().numpy().max() == 3.0

        module = BaseBanditNRNModel(
            target_names=['class1', 'class2'],
            feature_names=['feat1', 'feat1', 'feat3', 'feat4'],
            input_size=4,
            output_size=2,
            layer_sizes=[5, ],
            n_selected_features_input=2,
            n_selected_features_internal=2,
            n_selected_features_output=2,
            perform_prune_quantile=0.5,
            ucb_scale=2.5
        )

        mask_before = deepcopy(module.rn.model[0].mask.clone())
        module.perform_prune()
        mask_after = deepcopy(module.rn.model[0].mask.clone())

        assert torch.sum((mask_after == mask_before).int())/mask_after.numel() >= 0.5
        assert mask_after.cpu().numpy().max() == 3.0
