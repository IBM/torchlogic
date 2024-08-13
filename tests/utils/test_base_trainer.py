import os
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from torchlogic.modules import BanditNRNModule
from torchlogic.models.base import BaseBanditNRNModel
from torchlogic.utils.trainers.base import BaseReasoningNetworkDistributedTrainer

import pytest
from pytest import fixture

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(ROOT_DIR)


class TestBaseTrainer:

    @fixture
    def data_loaders(self):

        class StaticDataset(Dataset):
            def __init__(self):

                super(StaticDataset, self).__init__()
                self.X = pd.DataFrame(
                    {'feature1': [1, 1, 1, 1],
                     'feature2': [1, 0, 1, 0],
                     'feature3': [1, 0, 1, 0],
                     'feature4': [1, 0, 1, 0]}).values
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
        model = BaseBanditNRNModel(
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

        for item in model.rn.state_dict():
            if item.find('weights') > -1:
                weights = model.rn.state_dict()[item]
                weights.data.copy_(torch.ones_like(weights.data))

        return model

    @fixture
    def trainer(self, model):
        return BaseReasoningNetworkDistributedTrainer(
            model,
            torch.nn.BCELoss(),
            torch.optim.Adam(model.rn.parameters())
        )

    @staticmethod
    def test__validate_state_dicts(model, trainer):
        state_dict1 = deepcopy(model.rn.state_dict())

        for item in model.rn.state_dict():
            if item.find('weights') > -1:
                weights = model.rn.state_dict()[item]
                weights.data.copy_(torch.ones_like(weights.data) * 2)

        state_dict2 = deepcopy(model.rn.state_dict())

        assert not trainer._validate_state_dicts(state_dict1, state_dict2)
        assert trainer._validate_state_dicts(state_dict1, state_dict1)

    @staticmethod
    def test__save_best_state(model, trainer):
        trainer.save_best_state()

        assert trainer._validate_state_dicts(model.best_state['rn'].state_dict(), model.rn.state_dict())
        assert model.best_state['epoch'] == 0
        assert not model.best_state['was_pruned']

    @staticmethod
    def test__set_best_state(model, trainer):
        trainer.initialized_optimizer = torch.optim.Adam(model.rn.parameters())
        trainer.save_best_state()

        # change states
        for item in model.rn.state_dict():
            if item.find('weights') > -1:
                weights = model.rn.state_dict()[item]
                weights.data.copy_(torch.ones_like(weights.data) * 2)
        trainer.epoch = 10
        trainer.was_pruned = True

        assert model.best_state['epoch'] == 0
        assert not model.best_state['was_pruned']
        assert trainer.set_best_state()

    @staticmethod
    def test__evaluate_step(data_loaders, trainer):
        train_dl, val_dl = data_loaders

        out = trainer.evaluate_step(dl=val_dl, epoch=0, plateau_counter=0)

        assert out == 0
        assert trainer.best_val_performance == 0.5
        assert isinstance(trainer.model.best_state['rn'], BanditNRNModule)
        assert not trainer.model.best_state['was_pruned']
        assert trainer.model.best_state['epoch'] == 0

    @staticmethod
    def test___validation_step(data_loaders, trainer):
        train_dl, val_dl = data_loaders

        out = trainer._validation_step(val_dl=val_dl, epoch=0, plateau_counter=0)

        assert out == {'plateau_counter': 0}
        assert trainer.best_val_performance == 0.5
        assert isinstance(trainer.model.best_state['rn'], BanditNRNModule)
        assert not trainer.model.best_state['was_pruned']
        assert trainer.model.best_state['epoch'] == 0

    @staticmethod
    def test__train(data_loaders, model, trainer):
        train_dl, val_dl = data_loaders

        # basic trainer
        trainer.epochs = 2
        trainer.train(train_dl, val_dl)

        assert trainer.epoch == 1
        assert trainer.best_val_performance == 0.5
        assert isinstance(trainer.model.best_state['rn'], BanditNRNModule)
        assert not trainer.model.best_state['was_pruned']
        assert trainer.model.best_state['epoch'] == 1

        # basic trainer without validation set provided
        trainer.epochs = 2
        trainer.train(train_dl)

        assert trainer.epoch == 1
        assert trainer.best_val_performance == 0.5
        assert isinstance(trainer.model.best_state['rn'], BanditNRNModule)
        assert not trainer.model.best_state['was_pruned']
        assert trainer.model.best_state['epoch'] == 1
