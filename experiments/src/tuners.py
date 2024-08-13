import os
import copy
import shutil
import uuid
import joblib

import optuna
import numpy as np
import pandas as pd

from optuna.trial import TrialState

from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from torchlogic.models import BanditNRNClassifier
from torchlogic.utils.trainers import BanditNRNTrainer

from aix360.algorithms.rbm import FeatureBinarizerFromTrees
from minepy import cstats

from .datasets import TorchDataset
from .mlp import MlpMultiClassClassifier
from .fttransformer import FTTransfomerMultiClassClassifier

from .nam import NAM
from .danet_model import DANet
from .BRCG import BRCG
from .cofrnet import Cofrnet
# from .difflogic_model import DiffLogic

storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage("./journal.log"),
)


def generate_model(trial, model_name, X_train=None, y_train=None, categorical_idx_counts=None, numeric_idx=None):
    if model_name == "RF":
        # https: // arxiv.org / pdf / 2207.08815.pdf
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        n_estimators = trial.suggest_int("n_estimators", low=10, high=3000, log=True)
        max_depth = trial.suggest_categorical("max_depth", [None, 2, 3, 4])
        max_features = trial.suggest_categorical(
            "max_features", ["sqrt", "sqrt", "log2", None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        min_samples_split = trial.suggest_categorical("min_samples_split", [2, 3])
        min_samples_leaf = trial.suggest_int("min_samples_leaf", low=2, high=50, log=True)
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        min_impurity_decrease = trial.suggest_categorical("min_impurity_decrease", [0.0, 0.01, 0.02, 0.05])

        model = RandomForestClassifier(
            criterion=criterion,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            min_impurity_decrease=min_impurity_decrease,
            random_state=42,
            n_jobs=-1
        )
        return model

    if model_name == "GBT":
        # https: // arxiv.org / pdf / 2207.08815.pdf
        loss = trial.suggest_categorical("loss", ['log_loss', 'exponential'])
        learning_rate = trial.suggest_float("learning_rate", low=0.01, high=10, log=True)
        subsample = trial.suggest_float("subsample", low=0.5, high=1.0)
        n_estimators = trial.suggest_int("n_estimators", low=10, high=1000, log=True)
        criterion = trial.suggest_categorical("criterion", ['friedman_mse', 'squared_error'])
        min_samples_split = trial.suggest_categorical("min_samples_split", [2, 3])
        min_samples_leaf = trial.suggest_int("min_samples_leaf", low=2, high=50, log=True)
        min_impurity_decrease = trial.suggest_categorical("min_impurity_decrease", [0.0, 0.01, 0.02, 0.05])
        max_leaf_nodes = trial.suggest_categorical("max_leaf_nodes", [None, 5, 10, 15])

        model = GradientBoostingClassifier(
            loss=loss,
            learning_rate=learning_rate,
            subsample=subsample,
            criterion=criterion,
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_leaf_nodes=max_leaf_nodes,
            random_state=42,
        )
        return model

    if model_name == "GBT":
        # https: // arxiv.org / pdf / 2207.08815.pdf
        loss = trial.suggest_categorical("loss", ['log_loss', 'exponential'])
        learning_rate = trial.suggest_float("learning_rate", low=0.01, high=10, log=True)
        subsample = trial.suggest_float("subsample", low=0.5, high=1.0)
        n_estimators = trial.suggest_int("n_estimators", low=10, high=1000, log=True)
        criterion = trial.suggest_categorical("criterion", ['friedman_mse', 'squared_error'])
        min_samples_split = trial.suggest_categorical("min_samples_split", [2, 3])
        min_samples_leaf = trial.suggest_int("min_samples_leaf", low=2, high=50, log=True)
        min_impurity_decrease = trial.suggest_categorical("min_impurity_decrease", [0.0, 0.01, 0.02, 0.05])
        max_leaf_nodes = trial.suggest_categorical("max_leaf_nodes", [None, 5, 10, 15])

        model = GradientBoostingClassifier(
            loss=loss,
            learning_rate=learning_rate,
            subsample=subsample,
            criterion=criterion,
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_leaf_nodes=max_leaf_nodes,
            random_state=42,
        )
        return model

    if model_name == "SKLEARN-NN":
        hidden_layer_sizes = trial.suggest_categorical(
            "hidden_layer_sizes", ((10,), (20,), (50,), (10, 10), (20, 20), (50, 50))
        )
        activation = trial.suggest_categorical(
            "activation", ["logistic", "tanh", "relu"]
        )
        alpha = trial.suggest_float(
            "alpha", low=0.000001, high=0.0001
        )  # L2 regularization
        beta_2 = trial.suggest_float("beta_2", low=0.99, high=0.9999)
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            beta_2=beta_2,
            random_state=42
        )
        return model

    if model_name == "PYTORCH-NN-NODA-200":

        layer_size = trial.suggest_categorical("layer_size", [10, 20, 50])
        n_layers = trial.suggest_categorical("n_layers", [1, 2])
        layer_sizes = [layer_size] * n_layers
        activation = trial.suggest_categorical("activation", ["logistic", "tanh", "relu"])
        learning_rate = trial.suggest_float("learning_rate", low=0.001, high=0.1)  # L2 regularization
        swa = trial.suggest_categorical('swa', [True, False])

        ### Weight Decay Regularization
        use_weight_decay = trial.suggest_categorical('use_weight_decay', [True, False])
        if use_weight_decay:
            weight_decay = trial.suggest_float('weight_decay', low=0.00001, high=0.1)
        else:
            weight_decay = 0

        ### Lookahead Optimization
        use_lookahead = trial.suggest_categorical('use_lookahead', [True, False])
        if use_lookahead:
            lookahead_steps = trial.suggest_int('lookahead_steps', low=4, high=15, step=1)
            lookahead_steps_size = trial.suggest_float('lookahead_steps_size', low=0.5, high=0.8)
        else:
            lookahead_steps = 0
            lookahead_steps_size = 0

        ### Data Augmentation
        augment = trial.suggest_categorical('augment', [None])
        if augment is not None:
            augment_alpha = trial.suggest_float('augment_alpha', low=0.0, high=1.0)
        else:
            augment_alpha = 0

        ### Early Stopping
        use_early_stopping = trial.suggest_categorical('use_early_stopping', [True])
        if use_early_stopping:
            early_stopping_plateau_count = trial.suggest_int('early_stopping_plateau_count', low=25, high=50,
                                                             step=1)
        else:
            early_stopping_plateau_count = 0

        # batch norm
        batch_norm = trial.suggest_categorical('batch_norm', [True, False])

        # dropout
        use_dropout = trial.suggest_categorical('use_dropout', [True, False])
        if use_dropout:
            dropout_pct = trial.suggest_float('dropout_pct', low=0.0, high=0.8)
        else:
            dropout_pct = 0

        use_fbt = trial.suggest_categorical('use_fbt', [True, False])
        if use_fbt:
            tree_num = trial.suggest_int("fbt_tree_num", low=2, high=20)
            tree_depth = trial.suggest_int("fbt_tree_depth", low=2, high=10)
            tree_feature_selection = trial.suggest_float("fbt_tree_feature_selection", low=0.3, high=1.0)
            thresh_round = trial.suggest_int("fbt_thresh_round", low=2, high=6)
            fbt = FeatureBinarizerFromTrees(
                treeNum=tree_num,
                treeDepth=tree_depth,
                treeFeatureSelection=tree_feature_selection,
                threshRound=thresh_round,
                randomState=0
            )
            X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
            numeric_columns = X_train.columns[X_train.nunique() > 2]
            categorical_columns = X_train.columns[X_train.nunique() <= 2]
            X_train_numeric = X_train[numeric_columns]
            X_train_categorical = X_train[categorical_columns]
            X_train_numeric = fbt.fit_transform(X_train_numeric, y_train)
            X_train = np.hstack([X_train_numeric.values, X_train_categorical.values])
        else:
            fbt = None

        # Scheduler parameters
        t_0 = trial.suggest_int('t_0', low=2, high=10, step=1)
        t_mult = trial.suggest_int('t_mult', low=1, high=5, step=1)

        # reproducibility
        np.random.seed(42)
        torch.random.manual_seed(42)

        model = MlpMultiClassClassifier(
            input_size=X_train.shape[1],
            layer_sizes=layer_sizes,
            output_size=y_train.shape[1],
            learning_rate=learning_rate,
            n_epochs=200,
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
            t_mult=t_mult,
            fbt=fbt
        )

        return model

    if model_name == "FTT-NODA":

        attn_dropout = trial.suggest_float("attn_dropout", low=0.0, high=0.5)
        ff_dropout = trial.suggest_float("ff_dropout", low=0.0, high=0.5)
        depth = trial.suggest_categorical("depth", [5, 6, 7])
        heads = trial.suggest_categorical("heads", [7, 8, 9])
        learning_rate = trial.suggest_float("learning_rate", low=0.001, high=0.1)  # L2 regularization

        ### Weight Decay Regularization
        use_weight_decay = trial.suggest_categorical('use_weight_decay', [True, False])
        if use_weight_decay:
            weight_decay = trial.suggest_float('weight_decay', low=0.00001, high=0.1)
        else:
            weight_decay = 0

        ### Lookahead Optimization
        use_lookahead = trial.suggest_categorical('use_lookahead', [True, False])
        if use_lookahead:
            lookahead_steps = trial.suggest_int('lookahead_steps', low=4, high=15, step=1)
            lookahead_steps_size = trial.suggest_float('lookahead_steps_size', low=0.5, high=0.8)
        else:
            lookahead_steps = 0
            lookahead_steps_size = 0

        ### Data Augmentation
        augment = trial.suggest_categorical('augment', [None])
        if augment is not None:
            augment_alpha = trial.suggest_float('augment_alpha', low=0.0, high=1.0)
        else:
            augment_alpha = 0

        ### Early Stopping
        use_early_stopping = trial.suggest_categorical('use_early_stopping', [True])
        if use_early_stopping:
            early_stopping_plateau_count = trial.suggest_int('early_stopping_plateau_count', low=25, high=50,
                                                             step=1)
        else:
            early_stopping_plateau_count = 0

        use_fbt = trial.suggest_categorical('use_fbt', [False])
        if use_fbt:
            tree_num = trial.suggest_int("fbt_tree_num", low=2, high=20)
            tree_depth = trial.suggest_int("fbt_tree_depth", low=2, high=10)
            tree_feature_selection = trial.suggest_float("fbt_tree_feature_selection", low=0.3, high=1.0)
            thresh_round = trial.suggest_int("fbt_thresh_round", low=2, high=6)
            fbt = FeatureBinarizerFromTrees(
                treeNum=tree_num,
                treeDepth=tree_depth,
                treeFeatureSelection=tree_feature_selection,
                threshRound=thresh_round,
                randomState=0
            )
            X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
            numeric_columns = X_train.columns[X_train.nunique() > 2]
            categorical_columns = X_train.columns[X_train.nunique() <= 2]
            X_train_numeric = X_train[numeric_columns]
            X_train_categorical = X_train[categorical_columns]
            X_train_numeric = fbt.fit_transform(X_train_numeric, y_train)
            X_train = np.hstack([X_train_numeric.values, X_train_categorical.values])
        else:
            fbt = None

        # Scheduler parameters
        t_0 = trial.suggest_int('t_0', low=2, high=10, step=1)
        t_mult = trial.suggest_int('t_mult', low=1, high=5, step=1)

        # reproducibility
        np.random.seed(42)
        torch.random.manual_seed(42)

        model = FTTransfomerMultiClassClassifier(
            categories=categorical_idx_counts,
            num_continuous=len(numeric_idx),
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            depth=depth,
            heads=heads,
            output_size=y_train.shape[1],
            learning_rate=learning_rate,
            n_epochs=50,
            weight_decay=weight_decay,
            early_stopping_plateau_count=early_stopping_plateau_count,
            lookahead_steps=lookahead_steps,
            lookahead_steps_size=lookahead_steps_size,
            augment=augment,
            augment_alpha=augment_alpha,
            t_0=t_0,
            t_mult=t_mult,
            fbt=fbt
        )

        return model

    if model_name == "XGB":
        # https: // arxiv.org / pdf / 2207.08815.pdf
        max_depth = trial.suggest_int("max_depth", low=1, high=11, step=1)
        n_estimators = trial.suggest_categorical("n_estimators", [100, 200, 6000])
        min_child_weight = trial.suggest_float("min_child_weight", low=1, high=100, log=True)
        subsample = trial.suggest_float("subsample", low=0.5, high=1.0)
        eta = trial.suggest_float("eta", low=0.00001, high=0.1, log=True)  # learning rate
        colsample_bylevel = trial.suggest_float("colsample_bylevel", low=0.5, high=1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", low=0.5, high=1.0)
        gamma = trial.suggest_float("gamma", low=0.00000001, high=7, log=True)
        reg_lambda = trial.suggest_float("lreg_lambda", low=1, high=4, log=True)
        reg_alpha = trial.suggest_float("reg_alpha", low=0.00000001, high=100, log=True)
        objective = "binary:logistic"
        eval_metric = "auc"

        # reproducibility
        np.random.seed(42)
        torch.random.manual_seed(42)

        model = XGBClassifier(
            eta=eta,
            gamma=gamma,
            objective=objective,
            eval_metric=eval_metric,
            max_depth=max_depth,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bylevel=colsample_bylevel,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha
        )
        return model

    if model_name == "NAM-HO":

        learning_rate = trial.suggest_float('learning_rate', low=0.001, high=0.1)
        output_regularization = trial.suggest_float('output_regularization', low=0.001, high=0.1)
        l2_regularization = trial.suggest_float('l2_regularization', low=0.000001, high=0.0001)
        dropout = trial.suggest_categorical('dropout', [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        feature_dropout = trial.suggest_categorical('feature_dropout', [0, 0.05, 0.1, 0.2])

        tf_seed = 42
        training_epochs = 1000
        batch_size = 1024
        decay_rate = 0.995
        num_basis_functions = 1000
        units_multiplier = 2
        shallow = False
        early_stopping_epochs = 60

        model = NAM(
            training_epochs=training_epochs,
            learning_rate=learning_rate,
            output_regularization=output_regularization,
            l2_regularization=l2_regularization,
            batch_size=batch_size,
            decay_rate=decay_rate,
            dropout=dropout,
            tf_seed=tf_seed,
            feature_dropout=feature_dropout,
            num_basis_functions=num_basis_functions,
            units_multiplier=units_multiplier,
            shallow=shallow,
            early_stopping_epochs=early_stopping_epochs
        )

        return model

    if model_name == "DANet-200":
        patience = trial.suggest_int('patience', low=50, high=200, step=10)
        lr = trial.suggest_categorical('lr', [0.008, 0.02])
        layer = trial.suggest_categorical('layer', [8, 20, 32])
        base_outdim = trial.suggest_categorical('base_outdim', [64, 96])
        k = trial.suggest_categorical('k', [5, 8])
        drop_rate = trial.suggest_categorical('drop_rate', 0)

        max_epochs = 200
        seed = 42
        model = DANet(
            max_epochs=max_epochs,
            patience=patience,
            lr=lr,
            layer=layer,
            base_outdim=base_outdim,
            k=k,
            drop_rate=drop_rate,
            seed=seed
        )

        return model

    if model_name == 'BRCG':
        lambda0 = trial.suggest_float('lambda0', low=0.0001, high=0.01)
        lambda1 = trial.suggest_float('lambda1', low=0.0001, high=0.01)
        cnf = trial.suggest_categorical('cnf', [True, False])
        k = trial.suggest_int('k', low=5, high=15)
        d = trial.suggest_int('d', low=5, high=15)
        b = trial.suggest_int('b', low=5, high=15)

        tree_num = trial.suggest_int("fbt_tree_num", low=2, high=20)
        tree_depth = trial.suggest_int("fbt_tree_depth", low=2, high=10)
        tree_feature_selection = trial.suggest_float("fbt_tree_feature_selection", low=0.3, high=1.0)
        thresh_round = trial.suggest_int("fbt_thresh_round", low=2, high=6)
        fbt = FeatureBinarizerFromTrees(
            treeNum=tree_num,
            treeDepth=tree_depth,
            treeFeatureSelection=tree_feature_selection,
            threshRound=thresh_round,
            randomState=0
        )
        X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
        numeric_columns = X_train.columns[X_train.nunique() > 2]
        categorical_columns = X_train.columns[X_train.nunique() <= 2]
        X_train_numeric = X_train[numeric_columns]
        X_train_categorical = X_train[categorical_columns]
        X_train_numeric = fbt.fit_transform(X_train_numeric, y_train)
        X_train = np.hstack([X_train_numeric.values, X_train_categorical.values])

        iterMax = 100
        timeMax = 120
        eps = 1e-06
        solver = 'ECOS'

        model = BRCG(
            lambda0=lambda0,
            lambda1=lambda1,
            CNF=cnf,
            K=k,
            D=d,
            B=b,
            iterMax=iterMax,
            timeMax=timeMax,
            eps=eps,
            solver=solver,
            fbt=fbt
        )

        return model

    if model_name == 'Cofrnet':
        network_depth = trial.suggest_int('network_depth', low=2, high=50, step=1)
        variant = trial.suggest_categorical(
            'variant', 0)
        lr = trial.suggest_float('lr', low=0.0001, high=0.01)
        momentum = trial.suggest_float('momentum', low=0.85, high=0.95)
        early_stopping_plateau_count = trial.suggest_int('early_stopping_plateau_count', low=20, high=200, step=20)
        weight_decay = trial.suggest_float('weight_decay', low=0.00001, high=0.1)

        epochs = 200
        model = Cofrnet(
            network_depth=network_depth,
            variant=variant,
            input_size=X_train.shape[1],
            output_size=2 if y_train.shape[1] == 1 else y_train.shape[1],
            lr=lr,
            momentum=momentum,
            epochs=epochs,
            weight_decay=weight_decay,
            early_stopping_plateau_count=early_stopping_plateau_count
        )
        return model

    elif model_name == "Difflogic":
        num_neurons = trial.suggest_int('num_neurons', low=int(np.ceil(X_train.shape[1])), high=int(np.ceil(X_train.shape[1])) * 4, step=2)
        num_layers = trial.suggest_int('num_layers', low=2, high=8)
        tau = trial.suggest_categorical('tau', [1, 1/0.3, 1/0.1, 1/0.03, 1/0.01])
        learning_rate = trial.suggest_float('learning_rate', low=0.001, high=0.1)
        training_bit_count = trial.suggest_categorical('training_bit_count', [16, 32, 64])
        connections = trial.suggest_categorical('connections', ['unique', 'random'])
        grad_factor = trial.suggest_float('grad_factor', low=0.001, high=0.5)

        seed = 42
        batch_size = 128
        num_iterations = 200
        model = DiffLogic(
            input_dim=X_train.shape[1],
            class_count=y_train.shape[1],
            num_neurons=num_neurons,
            num_layers=num_layers,
            tau=tau,
            seed=seed,
            batch_size=batch_size,
            learning_rate=learning_rate,
            training_bit_count=training_bit_count,
            implementation='cuda' if torch.cuda.is_available() else 'python',
            num_iterations=num_iterations,
            connections=connections,
            grad_factor=grad_factor,
            eval_freq=20
        )

        return model


class BaselineTuner:
    def __init__(
        self,
        n_trials,
        feature_names,
        target_values,
        X_train_folds,
        X_val_folds,
        y_train_folds,
        y_val_folds,
        evaluation_metric,
        openml_id,
        random_state_generator,
        categorical_idx=None,
        categorical_idx_counts=None,
        numeric_idx=None
    ):
        self.best_model = None
        self.best_val_performance = 0.0

        self.n_trials = n_trials
        self.target_values = target_values
        self.X_train_folds = X_train_folds
        self.X_val_folds = X_val_folds
        self.y_train_folds = y_train_folds
        self.y_val_folds = y_val_folds
        self.evaluation_metric = evaluation_metric
        self.openml_id = openml_id
        self.random_state_generator = random_state_generator
        self.categorical_idx = categorical_idx
        self.categorical_idx_counts = categorical_idx_counts
        self.numeric_idx = numeric_idx

    def _objective(self, trial, model_name):
        # exits study once n_trials is reached even when using journal storage
        if len(trial.study.get_trials(states=[TrialState.COMPLETE])) >= self.n_trials:
            trial.study.stop()
            return

        model = generate_model(trial, model_name, self.X_train_folds[0], self.y_train_folds[0],
                               categorical_idx_counts=self.categorical_idx_counts, numeric_idx=self.numeric_idx)

        val_scores = []
        for i, (X_train, X_val, y_train, y_val) in enumerate(zip(
                self.X_train_folds, self.X_val_folds, self.y_train_folds, self.y_val_folds)):

            if model_name == 'PYTORCH-NN-NODA-200':

                if model.fbt is not None:
                    X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
                    X_val = pd.DataFrame(X_val, columns=[f'feature_{i}' for i in range(X_val.shape[1])])

                    numeric_columns = X_train.columns[X_train.nunique() > 2]
                    categorical_columns = X_train.columns[X_train.nunique() <= 2]
                    X_train_numeric = X_train[numeric_columns]
                    X_val_numeric = X_val[numeric_columns]
                    X_train_categorical = X_train[categorical_columns]
                    X_val_categorical = X_val[categorical_columns]

                    X_train_numeric = model.fbt.transform(X_train_numeric)
                    X_val_numeric = model.fbt.transform(X_val_numeric)

                    X_train = np.hstack([X_train_numeric.values, X_train_categorical.values])
                    X_val = np.hstack([X_val_numeric.values, X_val_categorical.values])

                train_dataset = TorchDataset(X=X_train, y=y_train)
                val_dataset = TorchDataset(X=X_val, y=y_val)

                val_dl = DataLoader(
                    val_dataset,
                    batch_size=min(128, len(val_dataset)),
                    generator=self.random_state_generator,
                    drop_last=False,
                    shuffle=False,
                    num_workers=os.cpu_count() - 1,
                    pin_memory=True,
                    persistent_workers=True,
                )

                if model.early_stopping_plateau_count > 0:
                    train_size = len(train_dataset)
                    indices = list(range(train_size))
                    np.random.seed(0)
                    np.random.shuffle(indices)
                    train_holdout_split_index = int(np.floor(0.2 * train_size))
                    train_idx, train_holdout_idx = indices[train_holdout_split_index:], indices[
                                                                                        :train_holdout_split_index]
                    train_sampler = SubsetRandomSampler(train_idx)

                    train_dl = DataLoader(
                        train_dataset,
                        batch_size=min(128, len(train_dataset)),
                        generator=self.random_state_generator,
                        drop_last=False,
                        sampler=train_sampler,
                        num_workers=os.cpu_count() - 1,
                        pin_memory=True,
                        persistent_workers=True,
                    )
                    train_holdout_dl = DataLoader(
                        train_dataset,
                        batch_size=min(128, len(train_dataset)),
                        generator=self.random_state_generator,
                        drop_last=False,
                        sampler=train_holdout_idx,
                        num_workers=os.cpu_count() - 1,
                        pin_memory=True,
                        persistent_workers=True,
                    )
                    fold_model = copy.copy(model)
                    if model.USE_CUDA:
                        fold_model.model.cuda()
                    fold_model.fit(train_dl, train_holdout_dl)
                else:
                    train_dl = DataLoader(
                        train_dataset,
                        batch_size=min(128, len(train_dataset)),
                        generator=self.random_state_generator,
                        drop_last=False,
                        shuffle=True,
                        num_workers=os.cpu_count() - 1,
                        pin_memory=True,
                        persistent_workers=True,
                    )
                    fold_model = copy.copy(model)
                    if model.USE_CUDA:
                        fold_model.model.cuda()
                    fold_model.fit(train_dl)

                fold_model = copy.copy(model)
                if model.USE_CUDA:
                    fold_model.model.cuda()
                fold_model.fit(train_dl, val_dl)
                y_val_pred, _ = fold_model.predict(val_dl)

                del train_dataset, val_dataset, train_dl, val_dl

            elif model_name == 'FTT-NODA':

                if model.fbt is not None:
                    X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
                    X_val = pd.DataFrame(X_val, columns=[f'feature_{i}' for i in range(X_val.shape[1])])

                    numeric_columns = X_train.columns[X_train.nunique() > 2]
                    categorical_columns = X_train.columns[X_train.nunique() <= 2]
                    X_train_numeric = X_train[numeric_columns]
                    X_val_numeric = X_val[numeric_columns]
                    X_train_categorical = X_train[categorical_columns]
                    X_val_categorical = X_val[categorical_columns]

                    X_train_numeric = model.fbt.transform(X_train_numeric)
                    X_val_numeric = model.fbt.transform(X_val_numeric)

                    X_train = np.hstack([X_train_numeric.values, X_train_categorical.values])
                    X_val = np.hstack([X_val_numeric.values, X_val_categorical.values])

                train_dataset = TorchDataset(X=X_train, y=y_train,
                                             categorical_idxs=self.categorical_idx, numeric_idxs=self.numeric_idx)
                val_dataset = TorchDataset(X=X_val, y=y_val,
                                           categorical_idxs=self.categorical_idx, numeric_idxs=self.numeric_idx)

                val_dl = DataLoader(
                    val_dataset,
                    batch_size=min(128, len(val_dataset)),
                    generator=self.random_state_generator,
                    drop_last=False,
                    shuffle=False,
                    num_workers=os.cpu_count() - 1,
                    pin_memory=True,
                    persistent_workers=True,
                )

                if model.early_stopping_plateau_count > 0:
                    train_size = len(train_dataset)
                    indices = list(range(train_size))
                    np.random.seed(0)
                    np.random.shuffle(indices)
                    train_holdout_split_index = int(np.floor(0.2 * train_size))
                    train_idx, train_holdout_idx = indices[train_holdout_split_index:], indices[
                                                                                        :train_holdout_split_index]
                    train_sampler = SubsetRandomSampler(train_idx)

                    train_dl = DataLoader(
                        train_dataset,
                        batch_size=min(128, len(train_dataset)),
                        generator=self.random_state_generator,
                        drop_last=False,
                        sampler=train_sampler,
                        num_workers=os.cpu_count() - 1,
                        pin_memory=True,
                        persistent_workers=True,
                    )
                    train_holdout_dl = DataLoader(
                        train_dataset,
                        batch_size=min(128, len(train_dataset)),
                        generator=self.random_state_generator,
                        drop_last=False,
                        sampler=train_holdout_idx,
                        num_workers=os.cpu_count() - 1,
                        pin_memory=True,
                        persistent_workers=True,
                    )
                    fold_model = copy.copy(model)
                    if model.USE_CUDA:
                        fold_model.model.cuda()
                    fold_model.fit(train_dl, train_holdout_dl)
                else:
                    train_dl = DataLoader(
                        train_dataset,
                        batch_size=min(128, len(train_dataset)),
                        generator=self.random_state_generator,
                        drop_last=False,
                        shuffle=True,
                        num_workers=os.cpu_count() - 1,
                        pin_memory=True,
                        persistent_workers=True,
                    )
                    fold_model = copy.copy(model)
                    if model.USE_CUDA:
                        fold_model.model.cuda()
                    fold_model.fit(train_dl)

                fold_model = copy.copy(model)
                if model.USE_CUDA:
                    fold_model.model.cuda()
                fold_model.fit(train_dl, val_dl)
                y_val_pred, _ = fold_model.predict(val_dl)

                del train_dataset, val_dataset, train_dl, val_dl

            elif model_name == "NAM-HO":
                train_size = len(X_train)
                indices = list(range(train_size))
                np.random.seed(0)
                np.random.shuffle(indices)
                train_holdout_split_index = int(np.floor(0.2 * train_size))
                train_idx, train_holdout_idx = indices[train_holdout_split_index:], indices[:train_holdout_split_index]
                X_holdout = X_train[train_holdout_idx]
                y_holdout = y_train[train_holdout_idx]
                X_train = X_train[train_idx]
                y_train = y_train[train_idx]

                validation_performance = model.testing(X_train, y_train, X_holdout, y_holdout, X_val, y_val)
                fold_model = copy.copy(model)

                # cleanup
                shutil.rmtree(model.FLAGS['logdir'])

            elif model_name == "DANet-200":
                train_size = len(X_train)
                indices = list(range(train_size))
                np.random.seed(0)
                np.random.shuffle(indices)
                train_holdout_split_index = int(np.floor(0.2 * train_size))
                train_idx, train_holdout_idx = indices[train_holdout_split_index:], indices[:train_holdout_split_index]
                X_holdout = X_train[train_holdout_idx]
                y_holdout = y_train[train_holdout_idx]
                X_train = X_train[train_idx]
                y_train = y_train[train_idx]

                model.fit(X_train, y_train, X_holdout, y_holdout)
                y_val_pred = model.predict(X_val)
                fold_model = copy.copy(model)

                # cleanup
                shutil.rmtree(model.clf.log.log_dir)

            elif model_name == "BRCG":
                X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
                X_val = pd.DataFrame(X_val, columns=[f'feature_{i}' for i in range(X_val.shape[1])])

                numeric_columns = X_train.columns[X_train.nunique() > 2]
                categorical_columns = X_train.columns[X_train.nunique() <= 2]
                X_train_numeric = X_train[numeric_columns]
                X_val_numeric = X_val[numeric_columns]
                X_train_categorical = X_train[categorical_columns]
                X_val_categorical = X_val[categorical_columns]

                X_train_numeric = model.fbt.transform(X_train_numeric)
                X_val_numeric = model.fbt.transform(X_val_numeric)

                X_train = np.hstack([X_train_numeric.values, X_train_categorical.values])
                X_val = np.hstack([X_val_numeric.values, X_val_categorical.values])

                fold_model = copy.copy(model)
                fold_model.fit(X_train, y_train)
                y_val_pred = fold_model.predict(X_val)

            elif model_name == "Cofrnet":

                train_size = len(X_train)
                indices = list(range(train_size))
                np.random.seed(0)
                np.random.shuffle(indices)
                train_holdout_split_index = int(np.floor(0.2 * train_size))
                train_idx, train_holdout_idx = indices[train_holdout_split_index:], indices[:train_holdout_split_index]
                X_holdout = X_train[train_holdout_idx]
                y_holdout = y_train[train_holdout_idx]
                X_train = X_train[train_idx]
                y_train = y_train[train_idx]

                try:
                    fold_model = copy.copy(model)
                    fold_model.train(X_train, y_train, X_holdout, y_holdout)
                    if y_val.shape[1] == 1:
                        y_val_pred = model.predict(X_val)[:, -1]
                    else:
                        y_val_pred = model.predict(X_val)
                except RuntimeError as e:
                    if str(e).find("Function CustomizedLinearFunctionBackward returned an invalid gradient") > -1:
                        return None
                    else:
                        raise e
                except ValueError as e:
                    if str(e).find("Input contains NaN.") > -1 or str(e).find("Input contains infinity or a value too large for") > -1:
                        return None
                    else:
                        raise e

            elif model_name == "Difflogic":
                train_dataset = TorchDataset(X=X_train, y=y_train)
                val_dataset = TorchDataset(X=X_val, y=y_val)

                val_dl = DataLoader(
                    val_dataset,
                    batch_size=min(128, len(val_dataset)),
                    generator=self.random_state_generator,
                    drop_last=False,
                    shuffle=False,
                    num_workers=os.cpu_count() - 1,
                    pin_memory=True,
                    persistent_workers=True,
                )

                train_size = len(train_dataset)
                indices = list(range(train_size))
                np.random.seed(0)
                np.random.shuffle(indices)
                train_holdout_split_index = int(np.floor(0.2 * train_size))
                train_idx, train_holdout_idx = indices[train_holdout_split_index:], indices[
                                                                                    :train_holdout_split_index]
                train_sampler = SubsetRandomSampler(train_idx)

                train_dl = DataLoader(
                    train_dataset,
                    batch_size=min(128, len(train_dataset)),
                    generator=self.random_state_generator,
                    drop_last=False,
                    sampler=train_sampler,
                    num_workers=os.cpu_count() - 1,
                    pin_memory=True,
                    persistent_workers=True,
                )
                train_holdout_dl = DataLoader(
                    train_dataset,
                    batch_size=min(128, len(train_dataset)),
                    generator=self.random_state_generator,
                    drop_last=False,
                    sampler=train_holdout_idx,
                    num_workers=os.cpu_count() - 1,
                    pin_memory=True,
                    persistent_workers=True,
                )
                fold_model = copy.copy(model)
                try:
                    fold_model.train(train_dl, train_holdout_dl)
                except AssertionError as e:
                    return None
                y_val_pred = fold_model.predict(val_dl).cpu()

            else:
                fold_model = clone(model)
                fold_model.fit(X_train, y_train)
                y_val_pred = fold_model.predict(X_val)

            if model_name != "NAM-HO":
                val_performance = roc_auc_score(y_val, y_val_pred, multi_class='ovo', average='micro')
                val_scores += [val_performance]
            else:
                val_scores += [validation_performance]

        val_performance = np.mean(val_scores)
        if val_performance > self.best_val_performance:
            self.best_val_performance = val_performance
            if model_name not in ["FTT-NODA", "PYTORCH-NN-NODA-200", "DANet-200",
                                  "Cofrnet", "Difflogic"]:
                self.best_model = copy.deepcopy(fold_model)
            else:
                self.best_model = fold_model

        return val_performance

    def tune(self, model_name):
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=42)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            storage=storage,
            study_name=f"{model_name}-{self.openml_id}-auc-fix",
            load_if_exists=True
        )
        study.optimize(
            lambda trial: self._objective(trial, model_name), n_trials=self.n_trials
        )
        joblib.dump(study, f"{model_name}-{self.openml_id}-auc-fix-study.pkl")
        return study.best_params


class BanditRRNNODATuner:
    def __init__(
        self,
        n_trials,
        feature_names,
        target_values,
        X_train_folds,
        X_val_folds,
        y_train_folds,
        y_val_folds,
        evaluation_metric,
        random_state_generator,
        openml_id,
    ):
        self.best_model = None
        self.best_lnn_val_performance = 0.0

        self.n_trials = n_trials
        self.feature_names = [f"the {x} is" for x in feature_names]
        self.target_values = target_values
        self.X_train_folds = X_train_folds
        self.X_val_folds = X_val_folds
        self.y_train_folds = y_train_folds
        self.y_val_folds = y_val_folds
        self.evaluation_metric = evaluation_metric
        self.random_state_generator = random_state_generator
        self.openml_id = openml_id

        self.multi_class = len(self.target_values) > 2

        if self.multi_class:
            self.target_names = [str(x) + "_label" for x in self.target_values]
            # self.loss_func = nn.CrossEntropyLoss()  # can't use class independent training.
            # I think this performs worse even though BCELoss treats each class independently
            # create a one-vs-rest multi-class estimator
            self.loss_func = nn.BCELoss()
        else:
            self.target_names = ["class_label"]
            self.loss_func = nn.BCELoss()

    def _objective(self, trial):

        # exits study once n_trials is reached even when using journal storage
        if len(trial.study.get_trials(states=[TrialState.COMPLETE])) >= self.n_trials:
            trial.study.stop()
            return

        # set parameters
        # RRN hyper-parameters

        # layer sizes
        layer_sizes = trial.suggest_int("layer_sizes", low=2, high=30, step=1)
        n_layers = trial.suggest_int("n_layers", low=1, high=6, step=1)
        layer_sizes = [layer_sizes] * n_layers

        # n features internal
        n_selected_features_internal = trial.suggest_int(
            "n_selected_features_internal", low=2, high=min(min(layer_sizes), 10)
        )

        # n features output
        n_selected_features_output = trial.suggest_int(
            "n_selected_features_output", low=2, high=min(layer_sizes[-1], 10)
        )

        perform_prune_plateau_count = trial.suggest_int('perform_prune_plateau_count', low=1, high=8)
        perform_prune_quantile = trial.suggest_float('perform_prune_quantile', low=0.05, high=0.9)
        increase_prune_plateau_count = trial.suggest_int('increase_prune_plateau_count', low=0, high=20)
        increase_prune_plateau_count_plateau_count = trial.suggest_int('increase_prune_plateau_count_plateau_count',
                                                                       low=10, high=30)
        ucb_scale = trial.suggest_float('ucb_scale', low=1.0, high=2.0)
        normal_form = trial.suggest_categorical('normal_form', ['dnf', 'cnf'])
        prune_strategy = trial.suggest_categorical('prune_strategy', ['class', 'logic', 'logic_class'])
        delta = trial.suggest_float('delta', low=1.0, high=12.0)
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        swa = trial.suggest_categorical('swa', [True, False])
        add_negations = trial.suggest_categorical('add_negations', [True, False])
        weight_init = trial.suggest_float('weight_init', low=0.01, high=1.0)

        ## Optimizer Parameters

        ### Learning Rate
        learning_rate = trial.suggest_float('learning_rate', low=0.0001, high=0.15)

        ### L1 Regularization
        use_l1 = trial.suggest_categorical('use_l1', [False])
        if use_l1:
            l1_lambda = trial.suggest_float('l1_lambda', low=0.00001, high=0.1)
        else:
            l1_lambda = 0

        ### Weight Decay Regularization
        use_weight_decay = trial.suggest_categorical('use_weight_decay', [True, False])
        if use_weight_decay:
            weight_decay = trial.suggest_float('weight_decay', low=0.00001, high=0.1)
        else:
            weight_decay = 0

        ### Lookahead Optimization
        use_lookahead = trial.suggest_categorical('use_lookahead', [True, False])
        if use_lookahead:
            lookahead_steps = trial.suggest_int('lookahead_steps', low=4, high=15, step=1)
            lookahead_steps_size = trial.suggest_float('lookahead_steps_size', low=0.5, high=0.8)
        else:
            lookahead_steps = 0
            lookahead_steps_size = 0

        ### Data Augmentation
        augment = trial.suggest_categorical('augment', [None])
        if augment is not None:
            augment_alpha = trial.suggest_float('augment_alpha', low=0.0, high=1.0)
        else:
            augment_alpha = 0

        ### Tree Binarization
        if add_negations:
            use_fbt = False
        else:
            use_fbt = trial.suggest_categorical('use_fbt', [True, False])
        if use_fbt:
            tree_num = trial.suggest_int("fbt_tree_num", low=2, high=20)
            tree_depth = trial.suggest_int("fbt_tree_depth", low=2, high=10)
            tree_feature_selection = trial.suggest_float("fbt_tree_feature_selection", low=0.3, high=1.0)
            thresh_round = trial.suggest_int("fbt_thresh_round", low=2, high=6)
        else:
            self.fbt = None

        ### Early Stopping
        use_early_stopping = trial.suggest_categorical('use_early_stopping', [True])
        if use_early_stopping:
            early_stopping_plateau_count = trial.suggest_int('early_stopping_plateau_count', low=25, high=50, step=1)
        else:
            early_stopping_plateau_count = 0

        ## Scheduler parameters
        t_0 = trial.suggest_int('T_0', low=2, high=10, step=1)
        t_mult = trial.suggest_int('T_mult', low=1, high=5, step=1)

        val_scores = []
        for i, (X_train, X_val, y_train, y_val) in enumerate(zip(
                self.X_train_folds, self.X_val_folds, self.y_train_folds, self.y_val_folds)):

            if use_fbt:
                X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
                X_val = pd.DataFrame(X_val, columns=[f'feature_{i}' for i in range(X_val.shape[1])])

                numeric_columns = X_train.columns[X_train.nunique() > 2]
                categorical_columns = X_train.columns[X_train.nunique() <= 2]
                X_train_numeric = X_train[numeric_columns]
                X_val_numeric = X_val[numeric_columns]
                X_train_categorical = X_train[categorical_columns]
                X_val_categorical = X_val[categorical_columns]

                if i == 0:
                    self.fbt = FeatureBinarizerFromTrees(
                        treeNum=tree_num,
                        treeDepth=tree_depth,
                        treeFeatureSelection=tree_feature_selection,
                        threshRound=thresh_round,
                        randomState=0
                    )
                    X_train_numeric = self.fbt.fit_transform(X_train_numeric, y_train)
                else:
                    X_train_numeric = self.fbt.transform(X_train_numeric)
                X_val_numeric = self.fbt.transform(X_val_numeric)

                X_train_numeric.columns = list(map(
                    lambda x: str(x).replace("(", "").replace(")", "").replace(" ", "_")
                    .replace("/", "_").replace(",", "").replace("'", ""),
                    X_train_numeric.columns.to_flat_index()))
                X_val_numeric.columns = list(map(
                    lambda x: str(x).replace("(", "").replace(")", "").replace(" ", "_")
                    .replace("/", "_").replace(",", "").replace("'", ""),
                    X_val_numeric.columns.to_flat_index()))

                feature_names = list(X_train_numeric.columns) + list(categorical_columns)

                X_train = np.hstack([X_train_numeric.values, X_train_categorical.values])
                X_val = np.hstack([X_val_numeric.values, X_val_categorical.values])
            else:
                feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

            # feature binarization can result in fewer features than the original dataset
            # reselect the input features value on the first fold only
            # feature names from the original dataset doesn't match the size of the transformed data
            if i == 0:
                # n features input
                high = min(len(feature_names), 12)
                n_selected_features_input = trial.suggest_int(
                    "n_selected_features_input", low=2, high=high
                )

            assert len(feature_names) == X_train.shape[1] == X_val.shape[1], "feature names didn't match data shape"

            train_dataset = TorchDataset(X=X_train, y=y_train)
            val_dataset = TorchDataset(X=X_val, y=y_val)

            val_dl = DataLoader(
                train_dataset,
                batch_size=min(128, len(val_dataset)),
                generator=self.random_state_generator,
                drop_last=False,
                shuffle=False,
                num_workers=os.cpu_count() - 1,
                pin_memory=True,
                persistent_workers=True,
            )

            # init model

            # reproducibility
            np.random.seed(42)
            torch.random.manual_seed(42)

            # initial bandit policy
            mic_c_policy, _ = cstats(X_train.T, y_train.T, alpha=9, c=5, est="mic_e")
            mic_c_policy = torch.tensor(mic_c_policy.T)

            model = BanditNRNClassifier(
                target_names=self.target_names if len(self.target_names) > 2 else ['positive_class'],
                feature_names=feature_names,
                input_size=X_train.shape[1],
                output_size=len(self.target_names) if len(self.target_values) > 2 else 1,
                layer_sizes=layer_sizes,
                n_selected_features_input=n_selected_features_input,
                n_selected_features_internal=n_selected_features_internal,
                n_selected_features_output=n_selected_features_output,
                perform_prune_quantile=perform_prune_quantile,
                ucb_scale=ucb_scale,
                normal_form=normal_form,
                delta=delta,
                prune_strategy=prune_strategy,
                bootstrap=bootstrap,
                swa=swa,
                add_negations=add_negations,
                weight_init=weight_init,
                policy_init=mic_c_policy
            )

            epochs = 200
            accumulation_steps = 1
            optimizer = optim.AdamW(model.rn.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult)
            trainer = BanditNRNTrainer(
                model=model,
                loss_func=self.loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=epochs,
                accumulation_steps=accumulation_steps,
                l1_lambda=l1_lambda,
                early_stopping_plateau_count=early_stopping_plateau_count,
                perform_prune_plateau_count=perform_prune_plateau_count,
                increase_prune_plateau_count=increase_prune_plateau_count,
                increase_prune_plateau_count_plateau_count=increase_prune_plateau_count_plateau_count,
                lookahead_steps=lookahead_steps,
                lookahead_steps_size=lookahead_steps_size,
                augment=augment,
                augment_alpha=augment_alpha,
                class_independent=self.multi_class
            )

            if use_early_stopping:
                train_size = len(train_dataset)
                indices = list(range(train_size))
                np.random.seed(0)
                np.random.shuffle(indices)
                train_holdout_split_index = int(np.floor(0.2 * train_size))
                train_idx, train_holdout_idx = indices[train_holdout_split_index:], indices[:train_holdout_split_index]
                train_sampler = SubsetRandomSampler(train_idx)

                train_dl = DataLoader(
                    train_dataset,
                    batch_size=min(128, len(train_dataset)),
                    generator=self.random_state_generator,
                    drop_last=False,
                    sampler=train_sampler,
                    num_workers=os.cpu_count() - 1,
                    pin_memory=True,
                    persistent_workers=True,
                )
                train_holdout_dl = DataLoader(
                    train_dataset,
                    batch_size=min(128, len(train_dataset)),
                    generator=self.random_state_generator,
                    drop_last=False,
                    sampler=train_holdout_idx,
                    num_workers=os.cpu_count() - 1,
                    pin_memory=True,
                    persistent_workers=True,
                )
                trainer.train(
                    train_dl,
                    train_holdout_dl,
                    evaluation_metric=self.evaluation_metric,
                    multi_class=self.multi_class,
                )
            else:
                train_dl = DataLoader(
                    train_dataset,
                    batch_size=min(128, len(train_dataset)),
                    generator=self.random_state_generator,
                    drop_last=False,
                    shuffle=True,
                    num_workers=os.cpu_count() - 1,
                    pin_memory=True,
                    persistent_workers=True,
                )
                trainer.train(
                    train_dl,
                    evaluation_metric=self.evaluation_metric,
                    multi_class=self.multi_class,
                )
            trainer.set_best_state()

            predictions, targets = trainer.model.predict(val_dl)
            lnn_val_performance = roc_auc_score(targets, predictions, multi_class='ovo', average='micro')

            val_scores += [lnn_val_performance]

        k_fold_val_performance = np.mean(val_scores)
        if k_fold_val_performance > self.best_lnn_val_performance:
            self.best_lnn_val_performance = k_fold_val_performance
            self.best_model = copy.copy(trainer.model)
            self.best_model.rn = copy.deepcopy(trainer.model.rn)
            self.best_fbt = copy.deepcopy(self.fbt)

        del train_dataset, val_dataset, train_dl, val_dl

        return k_fold_val_performance

    def tune(self):
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=42)
        study = optuna.create_study(
            study_name=f"BRRN-NODA-{self.openml_id}-mutual-info-auc-fix",
            direction="maximize",
            sampler=sampler,
            storage=storage,
            load_if_exists=True,  # this argument helps in resuming the experiments
        )
        study.optimize(self._objective, n_trials=self.n_trials)
        joblib.dump(study, f"BRRN-NODA-{self.openml_id}-study-mutual-info-auc-fix.pkl")
        return study.best_params
