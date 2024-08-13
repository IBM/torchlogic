import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score

from torchlogic.models.base import BaseBanditNRNModel
from torchlogic.modules import BanditNRNModule
from torchlogic.utils.trainers import BanditNRNTrainer

from pytest import fixture

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(ROOT_DIR)


class TestBanditRRNTrainer:

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
    def data_loaders_bs1(self):

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

        return (DataLoader(StaticDataset(), batch_size=1, shuffle=False),
                DataLoader(StaticDataset(), batch_size=1, shuffle=False))

    @fixture
    def data_loaders_cl1(self):

        class StaticDataset(Dataset):
            def __init__(self):
                super(StaticDataset, self).__init__()
                self.X = pd.DataFrame(
                    {'feature1': [1, 1, 1, 1],
                     'feature2': [1, 1, 1, 1],
                     'feature3': [1, 1, 1, 1],
                     'feature4': [1, 1, 1, 1]}).values
                self.y = pd.DataFrame(
                    {'class1': [1, 0, 1, 0]}).values
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
    def model_cl1(self):
        model = BaseBanditNRNModel(
            target_names=['class1'],
            feature_names=['feat1', 'feat1', 'feat3', 'feat4'],
            input_size=4,
            output_size=1,
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
        return BanditNRNTrainer(
            model,
            torch.nn.BCELoss(),
            torch.optim.Adam(model.rn.parameters()),
            perform_prune_plateau_count=0,
            augment=None
        )

    @fixture
    def trainer_cl1(self, model_cl1):
        return BanditNRNTrainer(
            model_cl1,
            torch.nn.BCELoss(),
            torch.optim.Adam(model_cl1.rn.parameters()),
            perform_prune_plateau_count=0,
            augment=None
        )

    @fixture
    def trainer_accumulation(self, model):
        return BanditNRNTrainer(
            model,
            torch.nn.BCELoss(),
            torch.optim.Adam(model.rn.parameters()),
            perform_prune_plateau_count=0,
            augment=None,
            accumulation_steps=100
        )

    @fixture
    def trainer_scheduler(self, model):
        optimizer = torch.optim.Adam(model.rn.parameters())
        return BanditNRNTrainer(
            model,
            torch.nn.BCELoss(),
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.LinearLR(optimizer=optimizer),
            perform_prune_plateau_count=0,
            augment=None
        )

    @fixture
    def trainer_minimize(self, model):
        return BanditNRNTrainer(
            model,
            torch.nn.BCELoss(),
            torch.optim.Adam(model.rn.parameters()),
            perform_prune_plateau_count=0,
            objective='minimize'
        )

    @fixture
    def class_independent_trainer(self, model):
        return BanditNRNTrainer(
            model,
            torch.nn.BCELoss(),
            torch.optim.Adam(model.rn.parameters()),
            perform_prune_plateau_count=0,
            class_independent=True
        )

    @fixture
    def class_independent_trainer_minimize(self, model):
        return BanditNRNTrainer(
            model,
            torch.nn.BCELoss(),
            torch.optim.Adam(model.rn.parameters()),
            perform_prune_plateau_count=0,
            class_independent=True,
            objective='minimize'
        )

    @fixture
    def trainer_cm_augment(self, model):
        return BanditNRNTrainer(
            model,
            torch.nn.BCELoss(),
            torch.optim.Adam(model.rn.parameters()),
            perform_prune_plateau_count=0,
            augment='CM'
        )

    @fixture
    def trainer_cm_augment_cl1(self, model_cl1):
        return BanditNRNTrainer(
            model_cl1,
            torch.nn.BCELoss(),
            torch.optim.Adam(model_cl1.rn.parameters()),
            perform_prune_plateau_count=0,
            augment='CM'
        )

    @fixture
    def trainer_mu_augment(self, model):
        return BanditNRNTrainer(
            model,
            torch.nn.BCELoss(),
            torch.optim.Adam(model.rn.parameters()),
            perform_prune_plateau_count=0,
            augment='MU'
        )

    @fixture
    def trainer_mu_augment_cl1(self, model_cl1):
        return BanditNRNTrainer(
            model_cl1,
            torch.nn.BCELoss(),
            torch.optim.Adam(model_cl1.rn.parameters()),
            perform_prune_plateau_count=0,
            augment='MU'
        )

    # @fixture
    # def trainer_at_augment(self, model):
    #     return BanditRRNTrainer(
    #         model,
    #         torch.nn.BCELoss(),
    #         torch.optim.Adam(model.rn.parameters()),
    #         perform_prune_plateau_count=0,
    #         augment='AT'
    #     )
    #
    # @fixture
    # def trainer_at_augment_cl1(self, model_cl1):
    #     return BanditRRNTrainer(
    #         model_cl1,
    #         torch.nn.BCELoss(),
    #         torch.optim.Adam(model_cl1.rn.parameters()),
    #         perform_prune_plateau_count=0,
    #         augment='AT'
    #     )

    @staticmethod
    def test__save_best_state(model, trainer):
        trainer.save_best_state(mode='prune')

        assert trainer._validate_state_dicts(model.best_state['prune_rn'].state_dict(), model.rn.state_dict())
        assert model.best_state['prune_epoch'] == 0
        assert not model.best_state['prune_was_pruned']

        trainer.save_best_state(mode='eval')

        assert trainer._validate_state_dicts(model.best_state['rn'].state_dict(), model.rn.state_dict())
        assert model.best_state['epoch'] == 0
        assert not model.best_state['was_pruned']

    @staticmethod
    def test__set_best_state(model, trainer):
        trainer.initialized_optimizer = torch.optim.Adam(model.rn.parameters())
        trainer.save_best_state(mode='prune')

        # change states
        for item in model.rn.state_dict():
            if item.find('weights') > -1:
                weights = model.rn.state_dict()[item]
                weights.data.copy_(torch.ones_like(weights.data) * 2)
        trainer.epoch = 10
        trainer.was_pruned = True

        assert model.best_state['prune_epoch'] == 0
        assert not model.best_state['prune_was_pruned']
        assert trainer.set_best_state()

        trainer.save_best_state(mode='eval')

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
    def test__check_improvement(trainer, trainer_minimize):
        assert trainer._check_improvement(0.5, mode='eval'), "check improvement incorrect result"
        assert trainer._check_improvement(100, mode='prune'), "check improvement incorrect result"
        assert not trainer._check_improvement(0.0, mode='eval'), "check improvement incorrect result"
        assert not trainer._check_improvement(1e13, mode='prune'), "check improvement incorrect result"

        assert trainer_minimize._check_improvement(100, mode='eval'), "check improvement incorrect result"
        assert trainer_minimize._check_improvement(100, mode='prune'), "check improvement incorrect result"
        assert not trainer_minimize._check_improvement(1e13, mode='eval'), "check improvement incorrect result"
        assert not trainer_minimize._check_improvement(1e13, mode='prune'), "check improvement incorrect result"

    @staticmethod
    def test__check_indices_to_update(class_independent_trainer, class_independent_trainer_minimize):
        # on first epoch all indices will update from nan to the passed value
        assert (np.array([0, 1]) ==
                class_independent_trainer._check_indices_to_update(np.array([0.5, 0.5]), mode='eval')).all(), \
            "check indices incorrect result"
        assert (np.array([0, 1]) ==
                class_independent_trainer._check_indices_to_update(np.array([0, 0.5]), mode='eval')).all(), \
            "check indices incorrect result"
        assert (np.array([0, 1]) ==
                class_independent_trainer._check_indices_to_update(np.array([100, 100]), mode='prune')).all(), \
            "check indices incorrect result"
        assert (np.array([0, 1]) ==
                class_independent_trainer._check_indices_to_update(np.array([1e13, 100]), mode='prune')).all(), \
            "check indices incorrect result"

        # after epoch 0 indices update on improvement
        class_independent_trainer.best_class_val_performances = np.array([0.5, 0.5])
        class_independent_trainer.best_class_train_performances = np.array([100, 100])
        class_independent_trainer.epoch = 1
        assert (np.array([0, 1]) ==
                class_independent_trainer._check_indices_to_update(np.array([0.6, 0.6]), mode='eval')).all(), \
            "check indices incorrect result"
        assert (np.array([1]) ==
                class_independent_trainer._check_indices_to_update(np.array([0.5, 0.6]), mode='eval')).all(), \
            "check indices incorrect result"
        assert (np.array([0, 1]) ==
                class_independent_trainer._check_indices_to_update(np.array([10, 10]), mode='prune')).all(), \
            "check indices incorrect result"
        assert (np.array([0]) ==
                class_independent_trainer._check_indices_to_update(np.array([10, 100]), mode='prune')).all(), \
            "check indices incorrect result"

        # on first epoch all indices will update from nan to the passed value
        assert (np.array([0, 1]) ==
                class_independent_trainer_minimize._check_indices_to_update(np.array([100, 100]), mode='eval')).all(), \
            "check indices incorrect result"
        assert (np.array([0, 1]) ==
                class_independent_trainer_minimize._check_indices_to_update(np.array([1e13, 100]), mode='eval')).all(), \
            "check indices incorrect result"
        assert (np.array([0, 1]) ==
                class_independent_trainer_minimize._check_indices_to_update(np.array([100, 100]), mode='prune')).all(), \
            "check indices incorrect result"
        assert (np.array([0, 1]) ==
                class_independent_trainer_minimize._check_indices_to_update(np.array([1e13, 100]), mode='prune')).all(), \
            "check indices incorrect result"

        # after epoch 0 indices update on improvement
        class_independent_trainer_minimize.best_class_val_performances = np.array([100, 100])
        class_independent_trainer_minimize.best_class_train_performances = np.array([100, 100])
        class_independent_trainer_minimize.epoch = 1
        assert (np.array([0, 1]) ==
                class_independent_trainer_minimize._check_indices_to_update(np.array([10, 10]), mode='eval')).all(), \
            "check indices incorrect result"
        assert (np.array([1]) ==
                class_independent_trainer_minimize._check_indices_to_update(np.array([100, 10]), mode='eval')).all(), \
            "check indices incorrect result"
        assert (np.array([0, 1]) ==
                class_independent_trainer_minimize._check_indices_to_update(np.array([10, 10]), mode='prune')).all(), \
            "check indices incorrect result"
        assert (np.array([0]) ==
                class_independent_trainer_minimize._check_indices_to_update(np.array([10, 100]), mode='prune')).all(), \
            "check indices incorrect result"

    @staticmethod
    def test__class_independent_evaluate_step(class_independent_trainer, class_independent_trainer_minimize):
        # maximize
        mean_performance, indices_to_update = (
            class_independent_trainer._class_independent_evaluate_step(np.array([0.5, 0.5]), mode='eval'))
        assert mean_performance == 0.5, "mean performance not correct"
        assert (indices_to_update == np.array([0, 1])).all()
        assert (class_independent_trainer.best_class_val_performances == np.array([0.5, 0.5])).all()

        mean_performance, indices_to_update = (
            class_independent_trainer._class_independent_evaluate_step(np.array([100, 100]), mode='prune'))
        assert mean_performance == 100, "mean performance not correct"
        assert (indices_to_update == np.array([0, 1])).all()
        assert (class_independent_trainer.best_class_train_performances == np.array([100, 100])).all()

        mean_performance, indices_to_update = (
            class_independent_trainer._class_independent_evaluate_step(np.array([1.0, 0.5]), mode='eval'))
        assert mean_performance == 0.75, "mean performance not correct"
        assert (indices_to_update == np.array([0])).all()
        assert (class_independent_trainer.best_class_val_performances == np.array([1.0, 0.5])).all()

        mean_performance, indices_to_update = (
            class_independent_trainer._class_independent_evaluate_step(np.array([0, 100]), mode='prune'))
        assert mean_performance == 50, "mean performance not correct"
        assert (indices_to_update == np.array([0])).all()
        assert (class_independent_trainer.best_class_train_performances == np.array([0, 100])).all()

        # minimize
        mean_performance, indices_to_update = (
            class_independent_trainer_minimize._class_independent_evaluate_step(np.array([100, 100]), mode='eval'))
        assert mean_performance == 100, "mean performance not correct"
        assert (indices_to_update == np.array([0, 1])).all()
        assert (class_independent_trainer_minimize.best_class_val_performances == np.array([100, 100])).all()

        mean_performance, indices_to_update = (
            class_independent_trainer_minimize._class_independent_evaluate_step(np.array([100, 100]), mode='prune'))
        assert mean_performance == 100, "mean performance not correct"
        assert (indices_to_update == np.array([0, 1])).all()
        assert (class_independent_trainer_minimize.best_class_train_performances == np.array([100, 100])).all()

        mean_performance, indices_to_update = (
            class_independent_trainer_minimize._class_independent_evaluate_step(np.array([0, 100]), mode='eval'))
        assert mean_performance == 50, "mean performance not correct"
        assert (indices_to_update == np.array([0])).all()
        assert (class_independent_trainer_minimize.best_class_val_performances == np.array([0, 100])).all()

        mean_performance, indices_to_update = (
            class_independent_trainer_minimize._class_independent_evaluate_step(np.array([0, 100]), mode='prune'))
        assert mean_performance == 50, "mean performance not correct"
        assert (indices_to_update == np.array([0])).all()
        assert (class_independent_trainer_minimize.best_class_train_performances == np.array([0, 100])).all()

    @staticmethod
    def test__increment_prune_and_grow_plateau_counter(trainer):
        prune_and_grow_plateau_counter,  increase_prune_and_grow_plateau_counter \
            = trainer._increment_prune_and_grow_plateau_counter(
            np.array([0, 1]), {0: 10, 1: 10}, {0: 10, 1: 10})
        assert prune_and_grow_plateau_counter == {0: 0, 1: 0}, "did not update prune_and_grow_plateau_counter correctly"
        assert increase_prune_and_grow_plateau_counter == {0: 0, 1: 0}, \
            "did not update increase_prune_and_grow_plateau_counter correctly"

        prune_and_grow_plateau_counter, increase_prune_and_grow_plateau_counter \
            = trainer._increment_prune_and_grow_plateau_counter(
            np.array([0]), {0: 10, 1: 10}, {0: 10, 1: 10})
        assert prune_and_grow_plateau_counter == {0: 0, 1: 11}, "did not update prune_and_grow_plateau_counter correctly"
        assert increase_prune_and_grow_plateau_counter == {0: 0, 1: 11}, \
            "did not update increase_prune_and_grow_plateau_counter correctly"

        prune_and_grow_plateau_counter, increase_prune_and_grow_plateau_counter \
            = trainer._increment_prune_and_grow_plateau_counter(
            np.array([]), {0: 10, 1: 10}, {0: 10, 1: 10})
        assert prune_and_grow_plateau_counter == {0: 11,
                                                  1: 11}, "did not update prune_and_grow_plateau_counter correctly"
        assert increase_prune_and_grow_plateau_counter == {0: 11, 1: 11}, \
            "did not update increase_prune_and_grow_plateau_counter correctly"

    @staticmethod
    def test__prune_evaluate_step(trainer, class_independent_trainer):
        # improvement
        kwargs = {
            'prune_and_grow_plateau_counter': 10,
            'increase_prune_and_grow_plateau_counter': 10,
            'total_loss': 100
        }
        (prune_and_grow_plateau_counter, increase_prune_and_grow_plateau_counter, indices_to_update,
         improvement_condition) = trainer._prune_evaluate_step(**kwargs)
        assert prune_and_grow_plateau_counter == 0, "did not update prune_and_grow_plateau_counter correctly"
        assert increase_prune_and_grow_plateau_counter == 0, \
            "did not update increase_prune_and_grow_plateau_counter correctly"
        assert indices_to_update is None, "did not process indices to update correctly"
        assert improvement_condition, "did not return correct improvement condition"

        # no improvement
        (prune_and_grow_plateau_counter, increase_prune_and_grow_plateau_counter, indices_to_update,
         improvement_condition) = trainer._prune_evaluate_step(**kwargs)
        assert prune_and_grow_plateau_counter == 11, "did not update prune_and_grow_plateau_counter correctly"
        assert increase_prune_and_grow_plateau_counter == 11, \
            "did not update increase_prune_and_grow_plateau_counter correctly"
        assert indices_to_update is None, "did not process indices to update correctly"
        assert not improvement_condition, "did not return correct improvement condition"

        # improvement class independent
        kwargs = {
            'prune_and_grow_plateau_counter': {0: 10, 1: 10},
            'increase_prune_and_grow_plateau_counter': {0: 10, 1: 10},
            'total_loss': np.array([100, 100])
        }
        (prune_and_grow_plateau_counter, increase_prune_and_grow_plateau_counter, indices_to_update,
         improvement_condition) = class_independent_trainer._prune_evaluate_step(**kwargs)
        assert prune_and_grow_plateau_counter == {0: 0, 1: 0}, "did not update prune_and_grow_plateau_counter correctly"
        assert increase_prune_and_grow_plateau_counter == {0: 0, 1: 0}, \
            "did not update increase_prune_and_grow_plateau_counter correctly"
        assert (indices_to_update == np.array([0, 1])).all(), "did not process indices to update correctly"
        assert improvement_condition, "did not return correct improvement condition"

        # no improvement class independent
        kwargs = {
            'prune_and_grow_plateau_counter': {0: 10, 1: 10},
            'increase_prune_and_grow_plateau_counter': {0: 10, 1: 10},
            'total_loss': np.array([100, 100])
        }
        (prune_and_grow_plateau_counter, increase_prune_and_grow_plateau_counter, indices_to_update,
         improvement_condition) = class_independent_trainer._prune_evaluate_step(**kwargs)
        assert prune_and_grow_plateau_counter == {0: 11, 1: 11}, "did not update prune_and_grow_plateau_counter correctly"
        assert increase_prune_and_grow_plateau_counter == {0: 11, 1: 11}, \
            "did not update increase_prune_and_grow_plateau_counter correctly"
        assert (indices_to_update == np.array([])).all(), "did not process indices to update correctly"
        assert not improvement_condition, "did not return correct improvement condition"

    @staticmethod
    def test__prune_step(trainer, class_independent_trainer, data_loaders):
        train_dl, val_dl = data_loaders

        # plateau counter criteria met before increase plateau counter met
        kwargs = {
            'prune_and_grow_plateau_counter': 10,
            'increase_prune_and_grow_plateau_counter': 19,
            'train_dl': train_dl,
            'output_metric': roc_auc_score
        }
        prune_and_grow_plateau_counter, increase_prune_and_grow_plateau_counter = trainer._prune_step(**kwargs)
        assert trainer.was_pruned, "prune step did not adjust prune status correctly"
        assert prune_and_grow_plateau_counter == 0, "prune_and_grow_plateau_counter did not update correctly"

        # plateau counter criteria met simultaneously with increase plateau counter
        kwargs = {
            'prune_and_grow_plateau_counter': 10,
            'increase_prune_and_grow_plateau_counter': 20,
            'train_dl': train_dl,
            'output_metric': roc_auc_score
        }
        prune_and_grow_plateau_counter, increase_prune_and_grow_plateau_counter = trainer._prune_step(**kwargs)
        assert not trainer.was_pruned, "prune step did not adjust prune status correctly"
        assert prune_and_grow_plateau_counter == 10, "prune_and_grow_plateau_counter did not update correctly"
        assert increase_prune_and_grow_plateau_counter == 20, \
            "increase_prune_and_grow_plateau_counter did not update correctly"

        # neither criteria is met
        kwargs = {
            'prune_and_grow_plateau_counter': 5,
            'increase_prune_and_grow_plateau_counter': 10,
            'train_dl': train_dl,
            'output_metric': roc_auc_score
        }
        trainer.prune_and_grow_plateau_count = 10
        trainer.increase_prune_plateau_count_plateau_count = 20
        trainer.was_pruned = False
        prune_and_grow_plateau_counter, increase_prune_and_grow_plateau_counter = trainer._prune_step(**kwargs)
        assert not trainer.was_pruned, "prune step did not adjust prune status correctly"
        assert prune_and_grow_plateau_counter == 5, "prune_and_grow_plateau_counter did not update correctly"
        assert increase_prune_and_grow_plateau_counter == 10, \
            "increase_prune_and_grow_plateau_counter did not update correctly"

        # class independent plateau counter criteria met before increase plateau counter met
        kwargs = {
            'prune_and_grow_plateau_counter': {0: 10, 1: 10},
            'increase_prune_and_grow_plateau_counter': {0: 19, 1: 19},
            'train_dl': train_dl,
            'output_metric': roc_auc_score
        }
        prune_and_grow_plateau_counter, increase_prune_and_grow_plateau_counter = (
            class_independent_trainer._prune_step(**kwargs))
        assert class_independent_trainer.was_pruned, "prune step did not adjust prune status correctly"
        assert prune_and_grow_plateau_counter == {0: 0, 1: 0}, "prune_and_grow_plateau_counter did not update correctly"

        # class independent plateau counter criteria met simultaneously with increase plateau counter
        kwargs = {
            'prune_and_grow_plateau_counter': {0: 10, 1: 10},
            'increase_prune_and_grow_plateau_counter': {0: 20, 1: 20},
            'train_dl': train_dl,
            'output_metric': roc_auc_score
        }
        prune_and_grow_plateau_counter, increase_prune_and_grow_plateau_counter = (
            class_independent_trainer._prune_step(**kwargs))
        assert not class_independent_trainer.was_pruned, "prune step did not adjust prune status correctly"
        assert prune_and_grow_plateau_counter == {0: 10, 1: 10}, "prune_and_grow_plateau_counter did not update correctly"
        assert increase_prune_and_grow_plateau_counter == {0: 20, 1: 20}, \
            "increase_prune_and_grow_plateau_counter did not update correctly"

        # class independent neither criteria is met
        kwargs = {
            'prune_and_grow_plateau_counter': {0: 5, 1: 5},
            'increase_prune_and_grow_plateau_counter': {0: 10, 1: 10},
            'train_dl': train_dl,
            'output_metric': roc_auc_score
        }
        class_independent_trainer.prune_and_grow_plateau_count = 10
        class_independent_trainer.increase_prune_plateau_count_plateau_count = 20
        class_independent_trainer.was_pruned = False
        prune_and_grow_plateau_counter, increase_prune_and_grow_plateau_counter = (
            class_independent_trainer._prune_step(**kwargs))
        assert not class_independent_trainer.was_pruned, "prune step did not adjust prune status correctly"
        assert prune_and_grow_plateau_counter == {0: 5, 1: 5}, "prune_and_grow_plateau_counter did not update correctly"
        assert increase_prune_and_grow_plateau_counter == {0: 10, 1: 10}, \
            "increase_prune_and_grow_plateau_counter did not update correctly"

    @staticmethod
    def test__increment_plateau_counter(trainer, class_independent_trainer):
        # no improvement
        kwargs = {'plateau_counter': 0}
        trainer.best_val_performance = 0.5
        plateau_counter = trainer._increment_plateau_counter(val_performance=0.5, **kwargs)
        assert plateau_counter == 1, "plateau counter did not increment correctly"

        # improvement
        kwargs = {'plateau_counter': 10}
        trainer.best_val_performance = 0.5
        plateau_counter = trainer._increment_plateau_counter(val_performance=0.7, **kwargs)
        assert plateau_counter == 0, "plateau counter did not increment correctly"

        # class independent no improvement
        kwargs = {'plateau_counter': 0}
        class_independent_trainer.best_val_performance = 0.5
        plateau_counter = class_independent_trainer._increment_plateau_counter(
            val_performance=np.array([0.5, 0.5]), **kwargs)
        assert plateau_counter == 1, "plateau counter did not increment correctly"

        # class independent one improved
        kwargs = {'plateau_counter': 10}
        class_independent_trainer.best_val_performance = 0.5
        plateau_counter = class_independent_trainer._increment_plateau_counter(
            val_performance=np.array([0.6, 0.5]), **kwargs)
        assert plateau_counter == 0, "plateau counter did not increment correctly"

    @staticmethod
    def test___validation_step(data_loaders, trainer, model):
        train_dl, val_dl = data_loaders

        out = trainer._validation_step(
            val_dl=val_dl,
            epoch=0,
            plateau_counter=0)

        assert out == 0
        assert trainer.best_val_performance == 0.5
        assert isinstance(trainer.model.best_state['rn'], BanditNRNModule)
        assert not trainer.model.best_state['was_pruned']
        assert trainer.model.best_state['epoch'] == 0

        out = trainer._validation_step(
            val_dl=val_dl,
            epoch=11,
            plateau_counter=11)

        assert out == 12
        assert trainer.best_val_performance == 0.5
        assert isinstance(trainer.model.best_state['rn'], BanditNRNModule)
        assert not trainer.model.best_state['was_pruned']
        assert trainer.model.best_state['epoch'] == 0

    @staticmethod
    def test__augment_data(trainer, trainer_cm_augment, trainer_mu_augment,
                           # trainer_at_augment,
                           trainer_cl1, trainer_cm_augment_cl1, trainer_mu_augment_cl1,
                           # trainer_at_augment_cl1,
                           data_loaders, data_loaders_bs1, data_loaders_cl1):
        # batch size 2 classes 2
        train_dl, val_dl = data_loaders
        for batch in train_dl:
            break

        # no augmentation
        features, target = trainer._augment_data(batch['features'], batch['target'])
        assert torch.equal(features, torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])), "did not augment features correctly"
        assert torch.equal(target, torch.tensor([[1, 0], [0, 1]])), "did not augment target correctly"

        # 'CM' augmentation
        features, target = trainer_cm_augment._augment_data(batch['features'], batch['target'])
        assert features.size(0) == 2 * batch['features'].size(0), "did not augment features correctly"
        assert features.size(1) == batch['features'].size(1), "did not augment features correctly"
        assert target.size(0) == 2 * batch['target'].size(0), "did not augment target correctly"
        assert target.size(1) == batch['target'].size(1), "did not augment features correctly"

        # 'MU' augmentation
        features, target = trainer_mu_augment._augment_data(batch['features'], batch['target'])
        assert features.size(0) == 2 * batch['features'].size(0), "did not augment features correctly"
        assert features.size(1) == batch['features'].size(1), "did not augment features correctly"
        assert target.size(0) == 2 * batch['target'].size(0), "did not augment target correctly"
        assert target.size(1) == batch['target'].size(1), "did not augment features correctly"

        # # 'AT' augmentation
        # features, target = trainer_at_augment._augment_data(batch['features'], batch['target'])
        # assert features.size(0) == 2 * batch['features'].size(0), "did not augment features correctly"
        # assert features.size(1) == batch['features'].size(1), "did not augment features correctly"
        # assert target.size(0) == 2 * batch['target'].size(0), "did not augment target correctly"
        # assert target.size(1) == batch['target'].size(1), "did not augment features correctly"

        # batch size 1 classes 2
        train_dl, val_dl = data_loaders_bs1
        for batch in train_dl:
            break

        # no augmentation
        features, target = trainer._augment_data(batch['features'], batch['target'])
        assert torch.equal(features, torch.tensor([[1, 1, 1, 1]])), "did not augment features correctly"
        assert torch.equal(target, torch.tensor([[1, 0]])), "did not augment target correctly"

        # 'CM' augmentation
        features, target = trainer_cm_augment._augment_data(batch['features'], batch['target'])
        assert features.size(0) == 2 * batch['features'].size(0), "did not augment features correctly"
        assert features.size(1) == batch['features'].size(1), "did not augment features correctly"
        assert target.size(0) == 2 * batch['target'].size(0), "did not augment target correctly"
        assert target.size(1) == batch['target'].size(1), "did not augment features correctly"

        # 'MU' augmentation
        features, target = trainer_mu_augment._augment_data(batch['features'], batch['target'])
        assert features.size(0) == 2 * batch['features'].size(0), "did not augment features correctly"
        assert features.size(1) == batch['features'].size(1), "did not augment features correctly"
        assert target.size(0) == 2 * batch['target'].size(0), "did not augment target correctly"
        assert target.size(1) == batch['target'].size(1), "did not augment features correctly"

        # # 'AT' augmentation
        # features, target = trainer_at_augment._augment_data(batch['features'], batch['target'])
        # assert features.size(0) == 2 * batch['features'].size(0), "did not augment features correctly"
        # assert features.size(1) == batch['features'].size(1), "did not augment features correctly"
        # assert target.size(0) == 2 * batch['target'].size(0), "did not augment target correctly"
        # assert target.size(1) == batch['target'].size(1), "did not augment features correctly"

        # batch size 2 classes 1
        train_dl, val_dl = data_loaders_cl1
        for batch in train_dl:
            break

        # no augmentation
        features, target = trainer_cl1._augment_data(batch['features'], batch['target'])
        assert torch.equal(features, torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])), "did not augment features correctly"
        assert torch.equal(target, torch.tensor([[1], [0]])), "did not augment target correctly"

        # 'CM' augmentation
        features, target = trainer_cm_augment_cl1._augment_data(batch['features'], batch['target'])
        assert features.size(0) == 2 * batch['features'].size(0), "did not augment features correctly"
        assert features.size(1) == batch['features'].size(1), "did not augment features correctly"
        assert target.size(0) == 2 * batch['target'].size(0), "did not augment target correctly"
        assert target.size(1) == batch['target'].size(1), "did not augment features correctly"

        # 'MU' augmentation
        features, target = trainer_mu_augment_cl1._augment_data(batch['features'], batch['target'])
        assert features.size(0) == 2 * batch['features'].size(0), "did not augment features correctly"
        assert features.size(1) == batch['features'].size(1), "did not augment features correctly"
        assert target.size(0) == 2 * batch['target'].size(0), "did not augment target correctly"
        assert target.size(1) == batch['target'].size(1), "did not augment features correctly"

        # # 'AT' augmentation
        # features, target = trainer_at_augment_cl1._augment_data(batch['features'], batch['target'])
        # assert features.size(0) == 2 * batch['features'].size(0), "did not augment features correctly"
        # assert features.size(1) == batch['features'].size(1), "did not augment features correctly"
        # assert target.size(0) == 2 * batch['target'].size(0), "did not augment target correctly"
        # assert target.size(1) == batch['target'].size(1), "did not augment features correctly"

    @staticmethod
    def test__process_batches(trainer, trainer_cm_augment, trainer_accumulation, trainer_scheduler, data_loaders):
        train_dl, val_dl = data_loaders

        optimizer = trainer.optimizer
        kwargs = {'total_steps': 0}
        _, total_steps = trainer._process_batches(train_dl, optimizer=optimizer, **kwargs)
        assert total_steps == 2

        # TODO: What can we actually test here?
        optimizer = trainer_cm_augment.optimizer
        kwargs = {'total_steps': 0}
        _, total_steps = trainer_cm_augment._process_batches(train_dl, optimizer=optimizer, **kwargs)
        assert total_steps == 2

        optimizer = trainer_accumulation.optimizer
        kwargs = {'total_steps': 0}
        _, total_steps = trainer_accumulation._process_batches(train_dl, optimizer=optimizer, **kwargs)
        assert total_steps == 1  # accumulation steps is set to 100.
        # trainer logic steps at accumulation steps or the last batch

        optimizer = trainer_scheduler.optimizer
        kwargs = {'total_steps': 0}
        _, _ = trainer_scheduler._process_batches(train_dl, optimizer=optimizer, **kwargs)
        # note that scheduler starts with step count of 1
        assert trainer_scheduler.scheduler._step_count == 3, "did not step scheduler correctly"

    @staticmethod
    def test__train(data_loaders, model, trainer):
        torch.manual_seed(0)
        np.random.seed(0)

        train_dl, val_dl = data_loaders

        # basic trainer
        trainer.epochs = 25
        trainer.train(train_dl, val_dl)

        assert trainer.epoch == 20  # stop on plateau is set to 20
        assert trainer.best_val_performance > 0
        assert isinstance(trainer.model.best_state['rn'], BanditNRNModule)
        assert not trainer.model.best_state['was_pruned']
        assert trainer.model.best_state['epoch'] == 0
        assert trainer.was_pruned  # perform prune plateau count set to 0

        # basic trainer without validation set provided
        trainer.epochs = 5
        trainer.train(train_dl)

        assert trainer.epoch == 4
        assert trainer.best_val_performance > 0
        assert isinstance(trainer.model.best_state['rn'], BanditNRNModule)
        assert not trainer.model.best_state['was_pruned']
        assert trainer.model.best_state['epoch'] == 0
        assert trainer.was_pruned  # perform prune plateau count set to 0
