from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from pytorch_optimizer import Lookahead

from torchlogic.models.base import BaseBanditNRNModel
from .base import BaseReasoningNetworkDistributedTrainer


class AttnNRNTrainer(BaseReasoningNetworkDistributedTrainer):

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
            accumulation_steps (int): Number of optimization steps to perform before backward operation.
            objective (str): 'maximize' or 'minimize'.  Default is 'maximize'.
        """
        if class_independent:
            setattr(loss_func, "reduction", 'none')

        super(AttnNRNTrainer, self).__init__(
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            l1_lambda=l1_lambda,
            early_stopping_plateau_count=early_stopping_plateau_count,
            accumulation_steps=accumulation_steps,
            objective=objective,
        )

        self.lookahead_steps = lookahead_steps
        self.lookahead_steps_size = lookahead_steps_size
        if self.lookahead_steps > 0:
            self.optimizer = Lookahead(self.optimizer, k=self.lookahead_steps, alpha=self.lookahead_steps_size)

        self.augment = augment
        self.augment_alpha = augment_alpha
        if self.augment == 'CM':
            self.augmentor = v2.CutMix(num_classes=2, alpha=self.augment_alpha)
        elif self.augment == 'MU':
            self.augmentor = v2.MixUp(num_classes=2, alpha=self.augment_alpha)

        self.class_independent = class_independent
        if self.class_independent:
            self.best_class_val_performances = np.array([np.nan] * self.model.output_size)
            self.best_class_train_performances = np.array([np.nan] * self.model.output_size)
        self.model.best_state['rn'] = deepcopy(self.model.rn)
        self.model.best_state['epoch'] = 0

    def save_best_state(self, classes: npt.NDArray = None):
        """
        Save the state of the network.

        Args:
            classes (List[int]): List of integers indicating which class states to save.
        """
        if classes is not None:
            new_state_dict = deepcopy(self.model.best_state['rn'].state_dict())
            current_state_dict = deepcopy(self.model.rn.state_dict())
            for k, v in new_state_dict.items():
                if k.find('weights') > -1 or k.find('mask') > -1:
                    for c in classes:
                        new_state_dict[k][c].data.copy_(current_state_dict[k][c].data)
            self.model.best_state['rn'].load_state_dict(new_state_dict)
        else:
            self.model.best_state['rn'] = deepcopy(self.model.rn)

        self.model.best_state['epoch'] = self.epoch

    def set_best_state(self) -> bool:
        """
        Set the best state from the last saved.

        Returns:
            bool: was the state loaded successfully
        """
        self.model.rn.load_state_dict(self.model.best_state['rn'].state_dict())

        return self._validate_state_dicts(
            self.model.rn.state_dict(), self.model.best_state['rn'].state_dict())
    
    def _check_improvement(self, performance):
        """
        Check if performances have improved
        
        Args:
            mode (str): 'eval' or 'prune'.  Determines which set of performances to check. 'eval' is for early stopping
                and the final model.  'prune' is for the pruning process.
        """
        if self.objective == 'maximize':
            return performance > self.best_val_performance
        elif self.objective == 'minimize':
            return performance < self.best_val_performance
        else:
            raise AssertionError("`objective` must be one of 'maximize' or 'minimize'")

    def _check_indices_to_update(self, performance: npt.NDArray) -> npt.NDArray:
        """
        Check which class indices should be updated based on if the performance has improved.

        Args:
            performance (npt.NDArray): Array of performance values by class.
            mode (str): 'eval' or 'prune'.  Determines which set of performances to check. 'eval' is for early stopping
                and the final model.  'prune' is for the pruning process.

        Returns:
            npt.NDArray: class indices that have improved.
        """
        best_performances = self.best_class_val_performances

        if self.epoch <= 1 and self.objective == 'maximize':
            condition = ((performance > best_performances)
                         | np.isnan(best_performances))
        elif self.epoch > 1 and self.objective == 'maximize':
            condition = (performance > best_performances)
        elif self.epoch <= 1 and (self.objective == 'minimize'):
            condition = ((performance < best_performances)
                         | np.isnan(best_performances))
        elif self.epoch > 1 and (self.objective == 'minimize'):
            condition = (performance < best_performances)
        else:
            raise AssertionError("`objective` must be one of 'maximize' or 'minimize'")

        return np.where(condition)[0]

    def _class_independent_evaluate_step(
            self,
            performance: npt.NDArray,
    ) -> Tuple[float, npt.NDArray]:
        """
        Evaluate which classes have improved.  Report and save results.

        Args:
            performance (npt.NDArray): Performance by class

        Returns:
            Tuple[float, npt.NDArray]: mean performance over classes, classes that improved
        """
        indices_to_update = self._check_indices_to_update(performance)
        if len(indices_to_update) > 0:
            self.logger.info(f"CURRENT PERFORMANCE FOR CLASSES {indices_to_update} "
                             f"> BEST VAL PERFORMANCE FOR CLASSES {indices_to_update}")
            self.save_best_state(indices_to_update)

        for c in range(self.model.output_size):
            if c in indices_to_update:
                self.best_class_val_performances[c] = performance[c]

        best_performances = self.best_class_val_performances

        return np.nanmean(best_performances), indices_to_update

    def _increment_plateau_counter(self, val_performance: Union[npt.NDArray, float], **kwargs) -> int:
        """
        Increment the early stopping plateau counter.
        
        Args:
            val_performance (Union[npt.NDArray, float]): Validation performance by class or overall.
            **kwargs:
                plateau_counter (int): Current plateau count.
                
        Returns:
            int: plateau_counter
        """
        plateau_counter = kwargs.get('plateau_counter')

        # if in class independent mode then each class must have its own val performances
        if self.class_independent:
            val_performance, indices_to_update = self._class_independent_evaluate_step(
                val_performance)

        # check if the val performance is better
        improvement_condition = self._check_improvement(val_performance)

        if improvement_condition:
            self.logger.info(f"CURRENT VALIDATION PERFORMANCE {val_performance} "
                             f"IMPROVED VS BEST VAL PERFORMANCE {self.best_val_performance}.")

            plateau_counter = 0
            if not self.class_independent:
                # best states have already been saved by class in class_independent_evaluate_step
                self.save_best_state()
            self.best_val_performance = val_performance
        else:
            plateau_counter += 1

        return plateau_counter

    def _validation_step(self, **kwargs):
        """
        Perform validation step and increment early stopping plateau counter.

        Args:
            **kwargs:
                val_dl (DataLoader): PyTorch DataLoader with validation data.
                evaluation_metric (sklearn.metrics): Scikit-learn performance metric function.
                multi_class (bool): Use multi-class evaluation approach.

        Returns:
            int: plateau_counter
        """
        epoch = kwargs.get('epoch')

        self.logger.info(f"VALIDATION EVALUATION - EPOCH [{epoch}]: \n")

        predictions, targets = self.model.predict(kwargs.get('val_dl'))
        val_performance = self.model.evaluate(
            predictions=predictions,
            labels=targets,
            class_independent=self.class_independent,
            output_metric=kwargs.get('evaluation_metric'),
            multi_class=kwargs.get('multi_class')
        )

        plateau_counter = self._increment_plateau_counter(val_performance=val_performance, **kwargs)

        return plateau_counter

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

        return features, target

    def _process_batches(
            self,
            dl: DataLoader,
            optimizer,
            **kwargs,
    ) -> Tuple[float, int]:
        """
        Perform batched epoch.

        Args:
            dl (DataLoader): the training dataloader
                Format:
                    DataLoader must return batches with the keys:
                        'features': [BATCH_SIZE, N_FEATURES]
                        'target': [BATCH_SIZE, N_TARGETS]
            optimizer (Optimizer): pytorch optimizer

        Returns:
            Tuple[float, dict, int]: total loss, kwargs, total steps taken
        """
        total_loss = 0
        total_steps = kwargs.get('total_steps')
        for i, batch in enumerate(dl):

            # [BATCH_SIZE, N_FEATURES]
            features = batch['features']
            # [BATCH_SIZE, N_TARGETS]
            target = batch['target'].squeeze()
            target = target.reshape(features.size(0), -1)

            if self.USE_CUDA:
                features = features.cuda()
                target = target.cuda()
            elif self.USE_MPS:
                features = features.to('mps')
                target = target.to('mps')

            # data augmentation
            if self.augment is not None:
                features, target = self._augment_data(features, target)

            # [BATCH_SIZE, N_TARGETS]
            predictions = self.model.rn(features)
            predictions = predictions.squeeze()
            predictions = predictions.reshape(features.size(0), -1)

            # [BATCH_SIZE, N_TARGETS]
            loss = self.loss_func(predictions, target.float())

            # add l1 loss
            if self.l1_lambda > 0.0:
                l1_norm = sum(p.abs().sum() for p in self.model.rn.parameters())
                loss = loss + self.l1_lambda * l1_norm

            loss /= self.accumulation_steps

            if loss.ndim >= 1:
                loss = loss.sum(axis=0)
                loss_agg = loss.mean()
            else:
                loss_agg = loss

            loss_agg.backward()

            if (i + 1) % self.accumulation_steps == 0 or (i + 1 == len(dl)):
                optimizer.step()
                # if using stochastic weight averaging
                if self.model.averaged_rn is not None:
                    self.model.averaged_rn.update_parameters(self.model.rn)
                optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()
                total_steps += 1

            # report aggregated loss if using distributed training
            total_loss += loss

        if isinstance(total_loss, torch.Tensor):
            total_loss = total_loss.detach().cpu().numpy() / len(dl.dataset)

        return total_loss, total_steps

    def train(self, train_dl: DataLoader, val_dl: DataLoader = None, evaluation_metric=None, multi_class=False):
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
        total_steps = 0
        total_loss = 1e12
        plateau_counter = 0

        if val_dl is None:
            val_dl = train_dl

        for epoch in range(self.epochs):

            self.epoch = epoch

            kwargs = {
                'val_dl': val_dl,
                'train_dl': train_dl,
                'optimizer': self.optimizer,
                'epoch': epoch,
                'plateau_counter': plateau_counter,
                'evaluation_metric': evaluation_metric,
                'multi_class': multi_class,
                'output_metric': evaluation_metric,
                'total_loss': total_loss
            }

            if self.early_stopping_plateau_count > 0:
                self.model.rn.eval()
                plateau_counter = self._validation_step(**kwargs)
                self.model.rn.train()

            if plateau_counter >= self.early_stopping_plateau_count > 0:
                self.logger.info(f"STOPPING EARLY.  REACHED PLATEAU COUNT OF {plateau_counter}.")
                break

            total_loss, total_steps = (
                self._process_batches(
                    dl=train_dl,
                    optimizer=self.optimizer,
                    val_dl=val_dl,
                    epoch=epoch,
                    plateau_counter=plateau_counter,
                    total_steps=total_steps,
                    output_metric=evaluation_metric
                ))

            self.logger.info(f"Epoch: {epoch}, Total Loss: {total_loss}")
            epoch += 1

        if self.early_stopping_plateau_count == 0:
            self.save_best_state()


__all__ = [AttnNRNTrainer]
