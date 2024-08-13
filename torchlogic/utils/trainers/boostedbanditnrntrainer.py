import numpy as np

import torch
from torch.utils.data import DataLoader

from torchlogic.models.base import BaseBanditNRNModel
from .banditnrntrainer import BanditNRNTrainer


class BoostedBanditNRNTrainer(BanditNRNTrainer):

    def __init__(
            self,
            model: BaseBanditNRNModel,
            loss_func,
            optimizer,
            scheduler=None,
            epochs: int = 100,
            accumulation_steps: int = 1,
            objective: str = 'maximize',
            l1_lambda: float = 1e-5,
            lookahead_steps: int = 5,
            lookahead_steps_size: float = 0.7,
            augment: str = None,
            augment_alpha: float = 0.03,
            class_independent: bool = False,
            early_stopping_plateau_count: int = 20,
            perform_prune_plateau_count: int = 10,
            increase_prune_plateau_count: int = 20,
            increase_prune_plateau_count_plateau_count: int = 20,
            boosting_epochs: int = 1,
            partial_fit: bool = False
    ):
        """
        Train a BanditRRN model.

        Example:
            trainer = BanditRRNTrainer(
                model=model,
                loss_func=nn.CrossEntropyLoss(),
                optimizer=optim.Adam,
                epochs=100,
                learning_rate=learning_rate,
                accumulation_steps=1,
                l1_lambda=l1_lambda,
                weight_decay=weight_decay,
                early_stopping_plateau_count=early_stopping_plateau_count,
                perform_prune_plateau_count=perform_prune_plateau_count,
                explain_percentile=0.9,
                verbose=0,
                distributed=False
            )

            trainer.train(train_dl, val_dl)

        Args:
            model (BanditRRNModel): the model to be trained.
                Format:
                    This model must implement the following methods:
                        predict(DataLoader): returns Tuple[pd.DataFrame, pd.DataFrame]: prediction values, target values
                        evaluate(DataLoader): returns float: evaluation metric
            loss_func (Loss): One of pytorch's loss classes
            optimizer (Optimizer): One of pytorch's optimizer classes
            scheduler (Scheduler): One of pytorch's scheduler classes
            class_independent (bool): If true, perform pruning and policy updates independently per output.
                Used in most multi-output cases.
            epochs (int): Maximum number of epochs to train for
            l1_lambda (float): L1 regularization weight
            early_stopping_plateau_count (int): Number of epochs without improvement to end training.
            perform_prune_plateau_count (int): Number of epochs without improvement to prune and grow model.
            accumulation_steps (int): Number of optimization steps to perform before backward operation.
            objective (str): 'maximize' or 'minimize'.  Default is 'maximize'.
        """
        if class_independent:
            setattr(loss_func, "reduction", 'none')

        super(BoostedBanditNRNTrainer, self).__init__(
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            accumulation_steps=accumulation_steps,
            objective=objective,
            l1_lambda=l1_lambda,
            lookahead_steps=lookahead_steps,
            lookahead_steps_size=lookahead_steps_size,
            augment=augment,
            augment_alpha=augment_alpha,
            class_independent=class_independent,
            early_stopping_plateau_count=early_stopping_plateau_count,
            perform_prune_plateau_count=perform_prune_plateau_count,
            increase_prune_plateau_count=increase_prune_plateau_count,
            increase_prune_plateau_count_plateau_count=increase_prune_plateau_count_plateau_count
        )
        self.boosting_epochs = boosting_epochs
        self.partial_fit = partial_fit

    def boost(self, train_dl: DataLoader):
        """
        Train the rn model.

        Args:
            train_dl (DataLoader): data loader for training data
            val_dl (DataLoader): data loader for a held-out set used for early stopping
            evaluation_metric (function): scikit-learn evaluation metrics
            multi_class (bool): treat problem as multi-class for evaluation purposes

                Format:
                    DataLoaders must return batches with the keys:
                        'features': [BATCH_SIZE, N_FEATURES]
                        'target': [BATCH_SIZE, N_TARGETS]
        """
        partial_model = None
        all_features = []
        all_boosted_targets = []
        for epoch in range(self.boosting_epochs):
            for i, batch in enumerate(train_dl):
                # [BATCH_SIZE, N_FEATURES]
                features = batch['features']
                # [BATCH_SIZE, N_TARGETS]
                target = batch['target'].squeeze()

                if self.USE_CUDA:
                    features = features.cuda()
                    target = target.cuda()
                elif self.USE_MPS:
                    features = features.to('mps')
                    target = target.to('mps')

                # [BATCH_SIZE, N_TARGETS]
                predictions = self.model.rn(features)
                predictions = predictions.squeeze(-1)

                boosted_target = torch.abs(predictions - target)
                boosted_target = torch.where(torch.tensor(predictions > target), -1.0 * boosted_target, boosted_target)
                boosted_target = boosted_target.cpu().detach().numpy()
                features = features.cpu().detach().numpy()

                if self.partial_fit:
                    self.model.xgb.fit(features, boosted_target, xgb_model=partial_model)
                    partial_model = self.model.xgb
                else:
                    all_features += [features]
                    all_boosted_targets += [boosted_target]
        if not self.partial_fit:
            all_features = np.vstack(all_features)
            if self.model.output_size > 1:
                all_boosted_targets = np.vstack(all_boosted_targets)
            else:
                all_boosted_targets = np.concatenate(all_boosted_targets)
            self.model.xgb.fit(all_features, all_boosted_targets)

        self.model.xgb_is_fitted = True


__all__ = [BoostedBanditNRNTrainer]
