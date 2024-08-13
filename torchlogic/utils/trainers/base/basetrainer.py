import logging
from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from torchlogic.models.base import ReasoningNetworkModel


class BaseReasoningNetworkDistributedTrainer(object):

    def __init__(
            self,
            model: ReasoningNetworkModel,
            loss_func,
            optimizer,
            scheduler=None,
            epochs: int = 100,
            l1_lambda: float = 1e-5,
            early_stopping_plateau_count: int = 20,
            accumulation_steps: int = 1,
            objective: str = 'maximize'
    ):
        """
        Train a ReasoningNetwork model.

        Args:
            model (ReasoningNetwork): the model to be trained.
                Format:
                    This model must implement the following methods:
                        predict(DataLoader): returns Tuple[pd.DataFrame, pd.DataFrame]: prediction values, target values
                        evaluate(DataLoader): returns float: evaluation metric
            loss_func (Loss): One of pytorch's loss classes
            optimizer (Optimizer): One of pytorch's optimizer classes
            scheduler (Scheduler): One of pytorch's scheduler classes
            class_weights (torch.Tensor): tensor of class weights.
            epochs (int): Maximum number of epochs to train for
            l1_lambda (float): L1 regularization weight
            early_stopping_plateau_count (int): Number of epochs without improvement to end training.
            accumulation_steps (int): Number of optimization steps to perform before backward operation.
            validation_steps: Optional.  Number of optimization steps to perform evaluation after.
            objective (str): 'maximize' or 'minimize'.  Default is 'maximize'.
        """
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.l1_lambda = l1_lambda
        self.early_stopping_plateau_count = early_stopping_plateau_count
        self.accumulation_steps = accumulation_steps
        self.objective = objective

        self.epoch = 0
        if self.objective == 'maximize':
            self.best_val_performance = 0
            self.best_train_performance = 1e12
        elif self.objective == 'minimize':
            self.best_val_performance = 1e12
            self.best_train_performance = 1e12

        self.initialized_optimizer = None
        self.was_pruned = False

        self.USE_CUDA = torch.cuda.is_available()
        self.USE_MPS = torch.backends.mps.is_available()
        if hasattr(self.model, 'rn'):
            self.USE_DATA_PARALLEL = isinstance(self.model.rn, torch.nn.DataParallel)
        else:
            self.USE_DATA_PARALLEL = False

        self.logger = logging.getLogger(self.__class__.__name__)

    def _validate_state_dicts(self, model_state_dict_1, model_state_dict_2):
        if len(model_state_dict_1) != len(model_state_dict_2):
            self.logger.info(
                f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
            )
            return False

        # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
        if next(iter(model_state_dict_1.keys())).startswith("module"):
            model_state_dict_1 = {
                k[len("module") + 1:]: v for k, v in model_state_dict_1.items()
            }

        if next(iter(model_state_dict_2.keys())).startswith("module"):
            model_state_dict_2 = {
                k[len("module") + 1:]: v for k, v in model_state_dict_2.items()
            }

        for ((k_1, v_1), (k_2, v_2)) in zip(
                model_state_dict_1.items(), model_state_dict_2.items()
        ):
            if k_1 != k_2:
                self.logger.info(f"Key mismatch: {k_1} vs {k_2}")
                return False
            # convert both to the same CUDA device
            if str(v_1.device) != "cuda:0":
                v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
            if str(v_2.device) != "cuda:0":
                v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

            if not torch.allclose(v_1, v_2):
                self.logger.info(f"Tensor mismatch: {v_1} vs {v_2}")
                return False

        self.logger.info("ALL CHECKS ON STATE DICT PASSED")
        return True

    def save_best_state(self):
        """
        Save the state of the network.
        """
        self.model.best_state['rn'] = deepcopy(self.model.rn)
        self.model.best_state['epoch'] = self.epoch
        # self.model.best_state['initialized_optimizer'] = deepcopy(self.optimizer)
        self.model.best_state['was_pruned'] = self.was_pruned

    def set_best_state(self):
        """
        Set the best state from the last saved.
        """
        self.model.rn = deepcopy(self.model.best_state['rn'])
        self.epoch = self.model.best_state['epoch']
        # self.optimizer = deepcopy(self.model.best_state['initialized_optimizer'])
        self.was_pruned = self.model.best_state['was_pruned']

        return self._validate_state_dicts(self.model.rn.state_dict(), self.model.best_state['rn'].state_dict())

    def _rn_to_cuda(self):
        """
        Send model to cuda.
        """
        # to cuda again
        if self.USE_DATA_PARALLEL:
            self.model.rn = torch.nn.DataParallel(self.model.rn.module)
        if self.USE_CUDA:
            self.model.rn = self.model.rn.cuda()
        elif self.USE_MPS:
            self.model.rn = self.model.rn.to('mps')

    def _check_improvement(self, val_performance):
        """
        Check if performances have improved
        """
        if self.objective == 'maximize':
            return val_performance > self.best_val_performance
        elif self.objective == 'minimize':
            return val_performance < self.best_val_performance
        else:
            raise AssertionError("`objective` must be one of 'maximize' or 'minimize'")

    def evaluate_step(self, dl: DataLoader, epoch, plateau_counter, **kwds):
        """
        Perform evaluation and increment counters.

        Args:
            dl (DataLoader): data loader that should be used for evaluation
            epoch (int): current epoch
            plateau_counter (int): plateau_counter for early stopping

        Returns:
            int: plateau_counter
        """
        self.logger.info(f"VALIDATION EVALUATION - EPOCH [{epoch}]: \n")
        predictions, targets = self.model.predict(dl)
        val_performance = self.model.evaluate(
            predictions=predictions,
            labels=targets,
            output_metric=kwds.get('evaluation_metric')
        )
        if self._check_improvement(val_performance):
            self.logger.info(f"CURRENT VALIDATION PERFORMANCE {val_performance} "
                             f"IMPROVED VS BEST VAL PERFORMANCE {self.best_val_performance} "
                             f"UPDATING BEST PARAMETERS.")
            self.best_val_performance = val_performance
            self.save_best_state()
            plateau_counter = 0
        else:
            plateau_counter += 1

        return plateau_counter

    def _validation_step(self, **kwargs):
        plateau_counter = self.evaluate_step(
            dl=kwargs.get('val_dl'), epoch=kwargs.get('epoch'), plateau_counter=kwargs.get('plateau_counter'))
        return {'plateau_counter': plateau_counter}

    def _process_batches(
            self,
            dl: DataLoader,
            optimizer,
            **kwargs,
    ):
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
            Tuple[float, Tensor]: total loss, final batch of features used for sample level explanation
        """
        total_loss = 0
        total_steps = kwargs.get('total_steps')
        for i, batch in enumerate(dl):

            # [BATCH_SIZE, N_FEATURES]
            features = batch['features']
            # [BATCH_SIZE, N_TARGETS]
            target = batch['target'].squeeze(-1)

            if self.USE_CUDA:
                features = features.cuda()
                target = target.cuda()
            elif self.USE_MPS:
                features = features.to('mps')
                target = target.to('mps')

            # [BATCH_SIZE, N_TARGETS]
            predictions = self.model.rn(features)
            if predictions.ndim > 1:
                predictions = predictions.squeeze(-1)
            # [BATCH_SIZE, N_TARGETS]
            loss = self.loss_func(predictions, target.float())

            if loss.ndim >= 1:
                loss = loss.sum(axis=0)
                loss_agg = loss.mean()
            else:
                loss_agg = loss

            # add l1 loss
            if self.l1_lambda > 0.0:
                l1_norm = sum(p.abs().sum() for p in self.model.rn.parameters())
                loss_agg = loss_agg + self.l1_lambda * l1_norm

            loss_agg /= self.accumulation_steps

            loss_agg.backward()

            if (i + 1) % self.accumulation_steps == 0 or (i + 1 == len(dl)):
                optimizer.step()
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
        plateau_counter = 0
        total_steps = 0

        if val_dl is None:
            val_dl = train_dl

        for epoch in range(self.epochs):

            self.epoch = epoch

            if epoch > 0:
                plateau_counter = self.evaluate_step(
                    val_dl,
                    epoch,
                    plateau_counter,
                    evaluation_function=evaluation_metric,
                    multi_class=multi_class
                )

            if plateau_counter >= self.early_stopping_plateau_count:
                self.logger.info(f"STOPPING EARLY.  REACHED PLATEAU COUNT OF {plateau_counter}.")
                break

            total_loss, total_steps = self._process_batches(
                dl=train_dl, optimizer=self.optimizer, val_dl=val_dl, epoch=epoch,
                plateau_counter=plateau_counter, total_steps=total_steps)

            self.logger.info(f"Epoch: {epoch}, Total Loss: {total_loss}")
            epoch += 1


__all__ = [BaseReasoningNetworkDistributedTrainer]
