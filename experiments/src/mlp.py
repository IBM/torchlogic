import logging
from typing import List, Tuple
from copy import deepcopy
from collections import OrderedDict

import torch
import numpy as np
import pandas as pd
from torch import nn
from sklearn.metrics import roc_auc_score

from torchvision.transforms import v2
from pytorch_optimizer import Lookahead
# from torchattacks.attacks.fgsm import FGSM

from torch.optim.swa_utils import AveragedModel


class MLP(nn.Module):

    """
    MLP module
    """

    def __init__(self, input_size, layer_sizes, activation, dropout_pct, batch_norm):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.dropout_pct = dropout_pct
        self.batch_norm = batch_norm
        self.dropout_pct = dropout_pct

        assert isinstance(layer_sizes, tuple) or isinstance(layer_sizes, list), \
            "hidden_sizes must be type tuple or list."

        # define network
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'logistic':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("'activation' must be one of 'relu', 'logistic', 'tanh'.")

        # create input layer
        if self.batch_norm and self.dropout_pct == 0:
            self.input_layer = nn.Sequential(
                nn.BatchNorm1d(input_size),
                nn.Linear(self.input_size, self.layer_sizes[0]),
                self.activation
            )
        elif self.batch_norm and self.dropout_pct > 0:
            self.input_layer = nn.Sequential(
                nn.BatchNorm1d(input_size),
                nn.Linear(self.input_size, self.layer_sizes[0]),
                self.activation,
                nn.Dropout(p=self.dropout_pct)
            )
        elif self.dropout_pct > 0:
            self.input_layer = nn.Sequential(
                nn.Linear(self.input_size, self.layer_sizes[0]),
                self.activation,
                nn.Dropout(p=self.dropout_pct)
            )
        else:
            self.input_layer = nn.Sequential(
                nn.Linear(self.input_size, self.layer_sizes[0]),
                self.activation
            )

        # create hidden layers
        if self.batch_norm and self.dropout_pct == 0:
            self.layers = nn.Sequential(
                OrderedDict(
                    [('l_{}'.format(i), nn.Sequential(nn.BatchNorm1d(h_in), nn.Linear(h_in, h_out), self.activation))
                     for i, (h_in, h_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))]
                )
            )
        elif self.batch_norm and self.dropout_pct > 0:
            self.layers = nn.Sequential(
                OrderedDict(
                    [('l_{}'.format(i), nn.Sequential(nn.BatchNorm1d(h_in), nn.Linear(h_in, h_out),
                                                      self.activation, nn.Dropout(self.dropout_pct)))
                     for i, (h_in, h_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))]
                )
            )
        elif self.dropout_pct > 0:
            self.layers = nn.Sequential(
                OrderedDict(
                    [('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out),
                                                      self.activation, nn.Dropout(self.dropout_pct)))
                     for i, (h_in, h_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))]
                )
            )
        else:
            self.layers = nn.Sequential(OrderedDict([
                ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), self.activation))
                for i, (h_in, h_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))]))

    def forward(self, x):
        x = self.input_layer(x)
        return self.layers(x)


class MlpMultiClassModule(nn.Module):

    def __init__(
            self,
            input_size: int,
            layer_sizes: List[int],
            output_size: int,
            activation: str = 'relu',
            batch_norm: bool = False,
            dropout_pct: float = 0.0
    ):
        super(MlpMultiClassModule, self).__init__()
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.output_size = output_size
        self.batch_norm = batch_norm
        self.dropout_pct = dropout_pct

        self.mlp_module = MLP(input_size, layer_sizes, activation, batch_norm, dropout_pct)
        self.output_layer = nn.Linear(layer_sizes[-1], 1)

    def forward(self, x):
        x = self.mlp_module(x)
        x = x.squeeze().repeat(1, self.output_size).reshape(x.size(0), self.output_size, -1)
        x = self.output_layer(x)
        return x.squeeze(-1)


class MlpMultiClassTrainer(object):

    def __init__(
            self,
            input_size: int,
            layer_sizes: List[int],
            output_size: int,
            learning_rate: float,
            n_epochs: int,
            weight_decay: float,
            early_stopping_plateau_count: int = 10,
            activation: str = 'relu',
            batch_norm: bool = False,
            dropout_pct: float = 0.0,
            lookahead_steps: int = 5,
            lookahead_steps_size: float = 0.5,
            augment: str = None,
            augment_alpha: float = 0.0,
            swa: bool = False,
            t_0: int = 2,
            t_mult: int = 3
    ):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.early_stopping_plateau_count = early_stopping_plateau_count
        self.swa = swa
        self.t_0 = t_0
        self.t_mult = t_mult

        self.loss_func = None
        self.val_auc = []
        self.best_val_auc = 0.0
        self.best_params = None
        self.plateau_count = 0

        self.model = MlpMultiClassModule(
            input_size=input_size, layer_sizes=layer_sizes, output_size=output_size,
            batch_norm=batch_norm, dropout_pct=dropout_pct, activation=activation)

        self.USE_CUDA = torch.cuda.is_available()
        if self.USE_CUDA:
            self.model.cuda()

        self.averaged_model = None
        if self.swa:
            self.averaged_model = AveragedModel(self.model)

        self.augment = augment
        self.augment_alpha = augment_alpha
        if self.augment == 'CM':
            self.augmentor = v2.CutMix(num_classes=2, alpha=self.augment_alpha)
        elif self.augment == 'MU':
            self.augmentor = v2.MixUp(num_classes=2, alpha=self.augment_alpha)
        # elif self.augment == 'AT':
        #     self.attack = FGSM(self.model.rn, self.augment_alpha)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.lookahead_steps = lookahead_steps
        self.lookahead_steps_size = lookahead_steps_size
        if self.lookahead_steps > 0:
            self.optimizer = Lookahead(self.optimizer, k=self.lookahead_steps, alpha=self.lookahead_steps_size)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=self.t_0, T_mult=self.t_mult)

        self.logger = logging.getLogger(self.__class__.__name__)

    def check_validation_performance(self, dl):
        all_predictions, all_targets = self.predict(dl)
        for i in range(all_predictions.shape[1]):
            self.val_auc += [roc_auc_score(all_targets.iloc[:, i].values.reshape(-1, 1),
                                           all_predictions.iloc[:, i].values.reshape(-1, 1))]
        val_auc = np.mean(self.val_auc)

        if val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            self.best_params = {'checkpoint_dict': self.__dict__}
            self.plateau_count = 0
        else:
            self.plateau_count += 1

    def set_best_params(self):
        for (k, v) in self.best_params['checkpoint_dict'].items():
            setattr(self, k, v)

    def set_label(self, y):
        if isinstance(self.loss_func, torch.nn.MSELoss):
            label_data = torch.FloatTensor(y.values)
        else:
            label_data = torch.LongTensor(y.values)
        if label_data.ndim < 2:
            label_data = label_data.unsqueeze(-1)
        return label_data

    def _augment_data(self, features: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform data augmentation.

        Args:
            features (torch.Tensor): features from batch.
            target (torch.Tensor): targets from batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: augmented features, augmented targets
        """
        if self.augment in ['CM', 'MU']:
            augmented_features = [features]
            augmented_targets = [target]
            if self.model.output_size > 1:
                i = np.random.choice(self.model.output_size)
            else:
                i = 0
            new_features, _ = self.augmentor(features.unsqueeze(1).unsqueeze(1), target[:, i])
            augmented_features += [new_features.squeeze(1).squeeze(1)]
            augmented_targets += [target]
            features = torch.cat(augmented_features, dim=0)
            target = torch.cat(augmented_targets, dim=0)
        # elif self.augment == 'AT':
        #     augmented_features = [features]
        #     augmented_targets = [target, target]
        #     if target.ndim == 1:
        #         target = target.view(-1, 1)
        #     augmented_features += [self.attack(features, target.float()).to(features.device)]
        #     features = torch.cat(augmented_features, dim=0)
        #     target = torch.cat(augmented_targets, dim=0)

        return features, target

    def fit(self, train_dl, val_dl=None):

        for epoch in range(self.n_epochs):
            if val_dl is not None:
                self.check_validation_performance(val_dl)

            if self.plateau_count >= self.early_stopping_plateau_count:
                break

            self.model.train()
            self.optimizer.zero_grad()

            total_loss = 0
            for batch in train_dl:
                # TODO: Should this reshaping be done in another dataset?
                target = batch['target']
                features = batch['features']
                if features.ndim > 2:
                    features = features.squeeze()

                if self.USE_CUDA:
                    target = target.cuda()
                    features = features.cuda()

                # data augmentation
                if self.augment is not None:
                    features, target = self._augment_data(features, target)

                outputs = self.model(features)
                loss = self.loss_func(outputs, target.float())

                loss.backward()

                self.optimizer.step()

                # if using stochastic weight averaging
                if self.averaged_model is not None:
                    self.averaged_model.update_parameters(self.model)

                if self.scheduler is not None:
                    self.scheduler.step()

                total_loss += loss.item()

        if val_dl is None:
            self.best_params = {'checkpoint_dict': self.__dict__}

        self.set_best_params()

    def predict(self, dl):
        self.model.eval()

        all_predictions = []
        all_targets = []
        all_indexes = []

        for batch in dl:
            # TODO: Should this reshaping be done in another dataset?
            features = batch['features']
            all_indexes += [batch['sample_idx']]
            if 'target' in batch:
                all_targets += [deepcopy(batch['target'])]
            if features.ndim > 2:
                features = features.squeeze()

            if self.USE_CUDA:
                features = features.cuda()

            all_predictions += [self.model(features)]

        all_indexes = torch.cat(all_indexes, dim=0).detach()
        all_predictions = torch.cat(all_predictions, dim=0).cpu().detach()
        all_targets = torch.cat(all_targets, dim=0).detach()

        all_indexes = all_indexes.numpy()
        all_predictions = all_predictions.numpy()
        all_targets = all_targets.numpy()

        return (pd.DataFrame(all_predictions, index=all_indexes),
                pd.DataFrame(all_targets, index=all_indexes))


class MlpMultiClassClassifier(MlpMultiClassTrainer):

    def __init__(
            self,
            input_size: int,
            layer_sizes: List[int],
            output_size: int,
            learning_rate: float,
            n_epochs: int,
            weight_decay: float,
            early_stopping_plateau_count: int = 10,
            activation: str = 'relu',
            batch_norm: bool = False,
            dropout_pct: float = 0.0,
            lookahead_steps: int = 5,
            lookahead_steps_size: float = 0.5,
            augment: str = None,
            augment_alpha: float = 0.0,
            swa: bool = False,
            t_0: int = 2,
            t_mult: int = 3,
            fbt=None
    ):
        super(MlpMultiClassClassifier, self).__init__(
            input_size=input_size,
            layer_sizes=layer_sizes,
            output_size=output_size,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            weight_decay=weight_decay,
            early_stopping_plateau_count=early_stopping_plateau_count,
            activation=activation,
            batch_norm=batch_norm,
            dropout_pct=dropout_pct,
            lookahead_steps=lookahead_steps,
            lookahead_steps_size=lookahead_steps_size,
            augment=augment,
            augment_alpha=augment_alpha,
            swa=swa,
            t_0=t_0,
            t_mult=t_mult
        )
        if output_size == 1:
            self.loss_func = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()
        self.fbt = fbt

        if self.USE_CUDA:
            self.model.cuda()


class MlpMultiOutputRegressor(MlpMultiClassTrainer):

    def __init__(
            self,
            input_size: int,
            layer_sizes: List[int],
            output_size: int,
            learning_rate: float,
            n_epochs: int,
            weight_decay: float,
            early_stopping_plateau_count: int = 10,
            activation: str = 'relu',
            batch_norm: bool = False,
            dropout_pct: float = 0.0,
            lookahead_steps: int = 5,
            lookahead_steps_size: float = 0.5,
            augment: str = None,
            augment_alpha: float = 0.0,
            swa: bool = False,
            t_0: int = 2,
            t_mult: int = 3,
            fbt=None
    ):
        super(MlpMultiOutputRegressor, self).__init__(
            input_size=input_size,
            layer_sizes=layer_sizes,
            output_size=output_size,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            weight_decay=weight_decay,
            early_stopping_plateau_count=early_stopping_plateau_count,
            activation=activation,
            batch_norm=batch_norm,
            dropout_pct=dropout_pct,
            lookahead_steps=lookahead_steps,
            lookahead_steps_size=lookahead_steps_size,
            augment=augment,
            augment_alpha=augment_alpha,
            swa=swa,
            t_0=t_0,
            t_mult=t_mult
        )
        self.loss_func = torch.nn.MSELoss()
        self.fbt = fbt

        if self.USE_CUDA:
            self.model.cuda()
