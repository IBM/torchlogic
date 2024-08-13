import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from torchlogic.models.base import BoostedBanditNRNModel
from torchlogic.utils.trainers import BoostedBanditNRNTrainer

from pytest import fixture

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(ROOT_DIR)


class TestBoostedBanditRRN:

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
    def model(self):
        model = BoostedBanditNRNModel(
            target_names=['class1', 'class2'],
            feature_names=['feat1', 'feat1'],
            input_size=2,
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
                weights.data.copy_(torch.zeros_like(weights.data))

        return model

    @fixture
    def trainer(self, model):
        return BoostedBanditNRNTrainer(
            model,
            torch.nn.BCELoss(),
            torch.optim.Adam(model.rn.parameters()),
            perform_prune_plateau_count=0,
            augment=None
        )

    @staticmethod
    def test__predict(trainer, data_loaders):
        train_dl, val_dl = data_loaders
        trainer.boost(train_dl)
        predictions, all_targets = trainer.model.predict(val_dl)
        class_predictions = np.array(np.vstack([[True, False], [False, True], [True, False], [False, True]] * 1000))
        assert np.equal(predictions.values > 0.5, class_predictions).all(), "did not make correct boosted predictions"
