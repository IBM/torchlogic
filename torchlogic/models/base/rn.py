import os
import joblib
import inspect
import logging
from copy import deepcopy
from typing import Tuple, Union, List
from types import FunctionType

import torch
import numpy as np
import numpy.typing as npt
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from torchlogic.utils import tensor_agg


def _format(x):
    return x.replace("_label", "")


@joblib.delayed
def _compute_metric_func(func, predictions, col, tpn):
    p = predictions.dropna(subset=[tpn])
    t = p[tpn].values
    if not any(t == -1) and np.unique(p[tpn].values).shape[0] > 1:
        return func(p[tpn].values, p[f'{col}_{_format(tpn)}'].values)
    else:
        return np.nan


class ReasoningNetworkModel(object):

    def __init__(self):
        """
        Initialize a ReasoningNetworkModel
        """
        self.best_state = {}
        self.target_names = None
        self.rn = None

        self.USE_CUDA = torch.cuda.is_available()
        self.USE_MPS = torch.backends.mps.is_available()
        self.logger = logging.getLogger(self.__class__.__name__)

    def predict(self, dl: DataLoader) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predict for data given by features

        Args:
            dl (DataLoader): data loader for predictions

        Returns:
            (DataFrame, DataFrame): data frame of predictions, data frame of targets
        """
        all_predictions = []
        all_targets = []
        all_indexes = []
        with torch.no_grad():
            for batch in dl:
                # [BATCH_SIZE, N_FEATURES, 2]
                features = batch['features']
                if 'target' in batch:
                    # [BATCH_SIZE, N_TARGETS]
                    target = batch['target']
                    if target.ndim > 2:
                        target.squeeze()
                    all_targets += [deepcopy(target)]

                if self.USE_CUDA:
                    features = features.cuda()
                elif self.USE_MPS:
                    features = features.to('mps')

                # [BATCH_SIZE, N_TARGETS]
                all_predictions += [self.rn(features)]
                # [BATCH_SIZE]
                all_indexes += [deepcopy(batch['sample_idx'])]

            all_predictions = torch.cat(all_predictions, dim=0).cpu().detach()
            all_indexes = torch.cat(all_indexes, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

        all_predictions = all_predictions.numpy()
        all_indexes = all_indexes.numpy()
        all_targets = all_targets.numpy()

        predictions = pd.DataFrame(
            data=all_predictions,
            index=all_indexes,
            columns=[f"probs_{x.replace('_label', '')}" for x in self.target_names]
        )
        del all_predictions

        all_targets = pd.DataFrame(
            data=all_targets,
            index=all_indexes,
            columns=self.target_names
        )

        return predictions, all_targets

    def evaluate(
            self,
            predictions: pd.DataFrame,
            labels: pd.DataFrame,
            decision_boundary: float = 0.5,
            metrics: List[str] = [],
            output_metric: FunctionType = None,
            class_independent: bool = False,
            multi_class: bool = False
    ) -> Union[float, npt.NDArray]:
        """
        Evaluate predictions from features and labels

        Args:
            predictions (pd.DataFrame): data frame of predictions
            labels (pd.DataFrame): data frame of labels to evaluate against
            decision_boundary (float): (0, 1) decision boundary to use for classification
            metrics (List[str]): list of metrics to use.
                Options:  `AUC`, 'Target mAP', 'Sample mAP', `Base Rate`, `Accuracy`
            output_metric (FunctionType): A function taking 'true values', 'predicted values'
            class_independent (bool): If True, return a list of performances by class
            multi_class (bool): If True, treat as a multi-class problem, else treat as multi-label (or binary).

        Returns:
            Union[DataFrame, Int]: predictions or evaluation score
        """
        if output_metric is None:
            output_metric = roc_auc_score

        def _compute_metric(func, preds, column, metric=None, user=False):
            if not user:
                out = joblib.Parallel(n_jobs=max(1, os.cpu_count() - 1), verbose=0)(
                    _compute_metric_func(
                        func=func,
                        predictions=predictions,
                        col=column,
                        tpn=tn)
                    for tn in self.target_names)
            else:
                target_names = [x for x in self.target_names if not any(preds[x] == -1)]
                preds = preds[
                        [x for x in target_names] +
                        [f'{column}_{_format(x)}' for x in target_names]
                    ]
                out = preds.apply(lambda y: func(y[:len(target_names)], y[len(target_names):]), axis=1)

            if metric is not None:
                return metric(out)
            return np.array([round(x, 3) for x in out])

        predictions = labels.join(predictions)
        if multi_class:
            predictions_max = predictions[[f'probs_{_format(x)}' for x in self.target_names]].max(axis=1)
        else:
            predictions_max = None
        for tpn in self.target_names:
            if multi_class:
                predictions[f'predictions_{_format(tpn)}'] = predictions[f'probs_{_format(tpn)}'] == predictions_max
            else:
                predictions[f'predictions_{_format(tpn)}'] = predictions[f'probs_{_format(tpn)}'] > decision_boundary
            predictions[f'outcome_{_format(tpn)}'] = list(
                map(lambda x, y: x == y, predictions[f'predictions_{_format(tpn)}'], predictions[tpn]))

        # compute user map if multi-label
        if len(self.target_names) > 1 and 'Sample mAP' in metrics:
            mean_ap = _compute_metric(average_precision_score, predictions, "probs", np.nanmean, True)
            user_map = f'Sample mAP: {mean_ap}'
        else:
            user_map = ''

        if 'AUC' in metrics:
            avg_auc = f"Avg. AUC: {_compute_metric(roc_auc_score, predictions, 'probs', np.nanmean)}"
            med_auc = f"Med. AUC: {_compute_metric(roc_auc_score, predictions, 'probs', np.nanmedian)}"
        else:
            avg_auc = ''
            med_auc = ''

        if 'Base Rate' in metrics:
            avg_br = f"Avg. Base Rate: {predictions[self.target_names].mean(axis=1, skipna=True).mean()}"
        else:
            avg_br = ''

        if 'Accuracy' in metrics:
            acc = predictions[[f'outcome_{_format(tpn)}' for tpn in self.target_names]].mean(axis=1, skipna=True).mean()
            avg_acc = f"Avg. Accuracy: {acc}"
        else:
            avg_acc = ''

        if 'Target mAP' in metrics:
            target_map = f"Target mAP: {_compute_metric(average_precision_score, predictions, 'probs', np.nanmean)}"
        else:
            target_map = ''

        if metrics:
            self.logger.info(f"""
                    Predictions Size: {(predictions.shape[0], len(self.target_names))}
                    {avg_br}
                    {avg_acc}
                    {avg_auc}
                    {med_auc}
                    {target_map}
                    {user_map}
                    """)

        if (('y_true' in inspect.signature(output_metric).parameters)
            and ('y_pred' in inspect.signature(output_metric).parameters)
            and ('labels' in inspect.signature(output_metric).parameters)):
            col = 'predictions'
        else:
            col = 'probs'

        if not class_independent:
            return float(_compute_metric(output_metric, predictions, col, np.nanmean))
        return _compute_metric(output_metric, predictions, col, None)


__all__ = [ReasoningNetworkModel]
