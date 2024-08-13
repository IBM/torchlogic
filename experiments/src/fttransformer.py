import torch
from tab_transformer_pytorch import FTTransformer

import logging
from typing import List, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from torchvision.transforms import v2
from pytorch_optimizer import Lookahead


class FTTransformerMultiClassTrainer(object):

    def __init__(
            self,
            categories: tuple,
            num_continuous: int,
            output_size: int,
            attn_dropout: float,
            ff_dropout: float,
            depth: int,
            heads: int,
            learning_rate: float,
            n_epochs: int,
            weight_decay: float,
            early_stopping_plateau_count: int = 10,
            lookahead_steps: int = 5,
            lookahead_steps_size: float = 0.5,
            augment: str = None,
            augment_alpha: float = 0.0,
            swa: bool = False,
            t_0: int = 2,
            t_mult: int = 3
    ):
        self.categories = categories
        self.num_continuous = num_continuous
        self.output_size = output_size
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.depth = depth
        self.heads = heads
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

        self.model = FTTransformer(
            categories=self.categories, # tuple containing the number of unique values within each category
            num_continuous=self.num_continuous, # number of continuous values
            dim=32,                           # dimension, paper set at 32
            dim_out=self.output_size,                        # binary prediction, but could be anything
            depth=self.depth,                          # depth, paper recommended 6
            heads=self.heads,                          # heads, paper recommends 8
            attn_dropout=self.attn_dropout,                 # post-attention dropout
            ff_dropout=self.ff_dropout                    # feed forward dropout
        )

        self.USE_CUDA = torch.cuda.is_available()
        self.DATA_PARALLEL = bool(torch.cuda.device_count() > 1)

        if self.DATA_PARALLEL:
            self.model = torch.nn.DataParallel(self.model)
        if self.USE_CUDA:
            self.model.cuda()

        self.augment = augment
        self.augment_alpha = augment_alpha
        if self.augment == 'CM':
            self.augmentor = v2.CutMix(num_classes=2, alpha=self.augment_alpha)
        elif self.augment == 'MU':
            self.augmentor = v2.MixUp(num_classes=2, alpha=self.augment_alpha)

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
            if self.output_size > 1:
                if self.DATA_PARALLEL:
                    i = np.random.choice(self.model.module.output_size)
                else:
                    i = np.random.choice(self.model.output_size)
            else:
                i = 0
            new_features, _ = self.augmentor(features.unsqueeze(1).unsqueeze(1), target[:, i])
            augmented_features += [new_features.squeeze(1).squeeze(1)]
            augmented_targets += [target]
            features = torch.cat(augmented_features, dim=0)
            target = torch.cat(augmented_targets, dim=0)

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

                x_categ = features[:, batch["cat_idxs"][0]].long()
                x_numer = features[:, batch["num_idxs"][0]]
                outputs = self.model(x_categ, x_numer)
                loss = self.loss_func(outputs, target.float())

                loss.backward()

                self.optimizer.step()

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

            x_categ = features[:, batch["cat_idxs"][0]].long()
            x_numer = features[:, batch["num_idxs"][0]]

            all_predictions += [self.model(x_categ, x_numer)]

        all_indexes = torch.cat(all_indexes, dim=0).detach()
        all_predictions = torch.cat(all_predictions, dim=0).cpu().detach()
        all_targets = torch.cat(all_targets, dim=0).detach()

        all_indexes = all_indexes.numpy()
        all_predictions = all_predictions.numpy()
        all_targets = all_targets.numpy()

        return (pd.DataFrame(all_predictions, index=all_indexes),
                pd.DataFrame(all_targets, index=all_indexes))


class FTTransfomerMultiClassClassifier(FTTransformerMultiClassTrainer):

    def __init__(
            self,
            categories: tuple,
            num_continuous: int,
            depth: int,
            heads: int,
            output_size: int,
            attn_dropout: float,
            ff_dropout: float,
            learning_rate: float,
            n_epochs: int,
            weight_decay: float,
            early_stopping_plateau_count: int = 10,
            lookahead_steps: int = 5,
            lookahead_steps_size: float = 0.5,
            augment: str = None,
            augment_alpha: float = 0.0,
            swa: bool = False,
            t_0: int = 2,
            t_mult: int = 3,
            fbt=None
    ):
        super(FTTransfomerMultiClassClassifier, self).__init__(
            categories=categories,
            num_continuous=num_continuous,
            output_size=output_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            depth=depth,
            heads=heads,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            weight_decay=weight_decay,
            early_stopping_plateau_count=early_stopping_plateau_count,
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

        # if self.USE_CUDA:
        #     self.model.cuda()