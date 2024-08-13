from .danet.DAN_Task import DANetClassifier, DANetRegressor
import argparse
import os
import torch.distributed
import torch.backends.cudnn
from sklearn.metrics import accuracy_score, mean_squared_error
from .danet.lib.utils import normalize_reg_label
from qhoptim.pyt import QHAdam
from .danet.config.default import cfg
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from uuid import uuid4


class DANet(object):

    def __init__(self, max_epochs, patience, lr, layer, base_outdim, k, drop_rate, seed):
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr = lr
        self.layer = layer
        self.base_outdim = base_outdim
        self.k = k
        self.drop_rate = drop_rate

        self.clf = DANetClassifier(
            optimizer_fn=QHAdam,
            optimizer_params=dict(lr=self.lr, weight_decay=1e-5, nus=(0.8, 1.0)),
            scheduler_params=dict(gamma=0.95, step_size=20),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            layer=self.layer,
            base_outdim=self.base_outdim,
            k=self.k,
            drop_rate=self.drop_rate,
            seed=seed
        )

    def fit(self, X_train, y_train, X_valid, y_valid):
        self.clf.fit(
            X_train=X_train, y_train=y_train.ravel(),
            eval_set=[(X_valid, y_valid.ravel())],
            eval_name=['valid'],
            eval_metric=['accuracy'],
            max_epochs=self.max_epochs, patience=self.patience,
            batch_size=8192,
            virtual_batch_size=256,
            logname=f'DANet-{uuid4()}',
            # resume_dir=train_config['resume_dir'],
            n_gpu=1
        )

    def predict(self, X):
        return self.clf.predict(X)
