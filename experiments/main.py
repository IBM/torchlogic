import os
import argparse
import warnings
import random
import shutil

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler
torch.multiprocessing.set_sharing_strategy('file_system')

from torchlogic.models import BanditNRNClassifier, AttnNRNClassifier
from torchlogic.utils.trainers import BanditNRNTrainer, AttnNRNTrainer

from aix360.algorithms.rbm import FeatureBinarizerFromTrees
from minepy import cstats

from src.encoders import FeatureEncoder
from src.tuners import BanditRRNTuner, BaselineTuner, AttnNRNTuner, BanditRRNNODATuner
from src.datasets import OpemlMLDataset, TorchDataset
from src.mlp import MlpMultiClassClassifier
from src.fttransformer import FTTransfomerMultiClassClassifier
from src.nam import NAM
from src.danet_model import DANet
from src.BRCG import BRCG
from src.cofrnet import Cofrnet
# from src.difflogic_model import DiffLogic

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# https://arxiv.org/pdf/2207.08815.pdf
BENCHMARK_DATASETS = [44120, 44121, 44122, 44123, 44124, 44125, 44126, 44127, 44128, 44129, 44130, 44131,
                      44089, 44090, 44091, 44156, 44157, 44158, 44159, 44160, 44161, 44162]

pd.set_option("display.max_columns", 100)


def generate_model(model_name, best_params, random_state):
    if model_name == "RF":
        model = RandomForestClassifier(
            criterion=best_params['criterion'],
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            max_features=best_params['max_features'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            bootstrap=best_params['bootstrap'],
            min_impurity_decrease=best_params['min_impurity_decrease'],
            random_state=random_state,
            n_jobs=-1
        )
        return model

    if model_name == "GBT":
        model = GradientBoostingClassifier(
            loss=best_params['loss'],
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            criterion=best_params['criterion'],
            n_estimators=best_params['n_estimators'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            min_impurity_decrease=best_params['min_impurity_decrease'],
            max_leaf_nodes=best_params['max_leaf_nodes'],
            random_state=random_state
        )
        return model

    if model_name == "GBT":
        model = GradientBoostingClassifier(
            loss=best_params['loss'],
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            criterion=best_params['criterion'],
            n_estimators=best_params['n_estimators'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            min_impurity_decrease=best_params['min_impurity_decrease'],
            max_leaf_nodes=best_params['max_leaf_nodes'],
            random_state=random_state
        )
        return model

    if model_name == "XGB":
        objective = "binary:logistic"
        # objective = "multi:softprob"
        eval_metric = "auc"

        model = XGBClassifier(
            eta=best_params['eta'],
            gamma=best_params['gamma'],
            objective=objective,
            eval_metric=eval_metric,
            max_depth=best_params['max_depth'],
            n_estimators=best_params['n_estimators'],
            min_child_weight=best_params['min_child_weight'],
            subsample=best_params['subsample'],
            colsample_bylevel=best_params['colsample_bylevel'],
            colsample_bytree=best_params['colsample_bytree'],
            reg_lambda=best_params['lreg_lambda'],
            reg_alpha=best_params['reg_alpha']
        )
        return model


def bandit_rrn_noda_main(
        args,
        X_train_val,
        X_train_folds,
        X_val_folds,
        X_test,
        y_train_val,
        y_train_folds,
        y_val_folds,
        y_test
):
    g = torch.Generator()
    g.manual_seed(args.random_state)

    tuner = BanditRRNNODATuner(
        args.runs,
        dataset.feature_names,
        dataset.target_values,
        X_train_folds,
        X_val_folds,
        y_train_folds,
        y_val_folds,
        roc_auc_score,
        g,
        args.openml_id
    )
    best_params = tuner.tune()

    if 'use_fbt' in best_params and best_params['use_fbt']:
        fbt = FeatureBinarizerFromTrees(
            treeNum=best_params['fbt_tree_num'],
            treeDepth=best_params['fbt_tree_depth'],
            treeFeatureSelection=best_params['fbt_tree_feature_selection'],
            threshRound=best_params['fbt_thresh_round'],
            randomState=0
        )
        # fbt = tuner.best_fbt

        X_train_val = pd.DataFrame(X_train_val, columns=[f'feature_{i}' for i in range(X_train_val.shape[1])])
        X_test = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
        X_train_fold = pd.DataFrame(X_train_folds[0], columns=[f'feature_{i}' for i in range(X_train_folds[0].shape[1])])

        numeric_columns = X_train_fold.columns[X_train_fold.nunique() > 2]
        categorical_columns = X_train_fold.columns[X_train_fold.nunique() <= 2]
        X_train_val_numeric = X_train_val[numeric_columns]
        X_test_numeric = X_test[numeric_columns]
        X_train_val_categorical = X_train_val[categorical_columns]
        X_test_categorical = X_test[categorical_columns]

        fbt.fit(X_train_fold[numeric_columns], y_train_folds[0])
        X_train_val_numeric = fbt.transform(X_train_val_numeric)
        X_test_numeric = fbt.transform(X_test_numeric)

        X_train_val_numeric.columns = list(map(
            lambda x: str(x).replace("(", "").replace(")", "").replace(" ", "_")
            .replace("/", "_").replace(",", "").replace("'", ""),
            X_train_val_numeric.columns.to_flat_index()))
        X_test_numeric.columns = list(map(
            lambda x: str(x).replace("(", "").replace(")", "").replace(" ", "_")
            .replace("/", "_").replace(",", "").replace("'", ""),
            X_test_numeric.columns.to_flat_index()))

        feature_names = list(X_train_val_numeric.columns) + list(categorical_columns)

        X_train_val = np.hstack([X_train_val_numeric.values, X_train_val_categorical.values])
        X_test = np.hstack([X_test_numeric.values, X_test_categorical.values])

    else:
        feature_names = [f"feature_{i}" for i in range(X_train_val.shape[1])]

    assert len(feature_names) == X_train_val.shape[1] == X_test.shape[1], "feature names didn't match data shape"

    # build test dataset
    test_dataset = TorchDataset(X=X_test, y=y_test)
    test_dl = DataLoader(
        test_dataset,
        batch_size=128,
        generator=g,
        drop_last=False,
        shuffle=False,
        num_workers=os.cpu_count() - 1,
        pin_memory=True,
        persistent_workers=True,
    )

    score = 0.0

    # refit model on all data
    train_dataset = TorchDataset(X=X_train_val, y=y_train_val)

    # layer sizes
    layer_sizes = best_params['layer_sizes']
    n_layers = best_params['n_layers']
    layer_sizes = [layer_sizes] * n_layers

    # init model
    # reproducibility
    np.random.seed(args.random_state)
    torch.random.manual_seed(args.random_state)

    # initial bandit policy
    mic_c_policy, _ = cstats(X_train_val.T, y_train_val.T, alpha=9, c=5, est="mic_e")
    mic_c_policy = torch.tensor(mic_c_policy.T)

    model = BanditNRNClassifier(
        target_names=tuner.target_names if len(tuner.target_names) > 2 else ['positive_class'],
        feature_names=feature_names,
        input_size=X_train_val.shape[1],
        output_size=len(tuner.target_names) if len(tuner.target_values) > 2 else 1,
        layer_sizes=layer_sizes,
        n_selected_features_input=best_params['n_selected_features_input'],
        n_selected_features_internal=best_params['n_selected_features_internal'],
        n_selected_features_output=best_params['n_selected_features_output'],
        perform_prune_quantile=best_params['perform_prune_quantile'],
        ucb_scale=best_params['ucb_scale'],
        normal_form=best_params['normal_form'],
        delta=best_params['delta'],
        prune_strategy=best_params['prune_strategy'],
        bootstrap=best_params['bootstrap'],
        swa=best_params['swa'],
        add_negations=best_params['add_negations'],
        weight_init=best_params['weight_init'],
        policy_init=mic_c_policy
    )

    # trainable parameters from the NRN and 2 * feature count for mean and standard deviation
    # required for bandit (per class but all problems are binary classificaiton meaning only one dist is needed)
    print("Number of trainable parameters",
          sum([x.numel() for x in model.rn.parameters() if x.requires_grad]) + len(feature_names) * 2)

    epochs = 200
    accumulation_steps = 1
    optimizer = optim.AdamW(
        model.rn.parameters(), lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'] if best_params['use_weight_decay'] else 0)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=best_params['T_0'], T_mult=best_params['T_mult'])
    trainer = BanditNRNTrainer(
        model=model,
        loss_func=tuner.loss_func,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        accumulation_steps=accumulation_steps,
        l1_lambda=best_params['l1_lambda'] if best_params['use_l1'] else 0,
        early_stopping_plateau_count=best_params['early_stopping_plateau_count']
        if best_params['use_early_stopping'] else 0,
        perform_prune_plateau_count=best_params['perform_prune_plateau_count'],
        increase_prune_plateau_count=best_params['increase_prune_plateau_count'],
        increase_prune_plateau_count_plateau_count=best_params['increase_prune_plateau_count_plateau_count'],
        lookahead_steps=best_params['lookahead_steps'] if best_params['use_lookahead'] else 0,
        lookahead_steps_size=best_params['lookahead_steps_size'] if best_params['use_lookahead'] else 0,
        augment=best_params['augment'],
        augment_alpha=best_params['augment_alpha'] if best_params['augment'] is not None else 0,
        class_independent=tuner.multi_class
    )

    if best_params['use_early_stopping']:
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
            generator=g,
            drop_last=False,
            sampler=train_sampler,
            num_workers=os.cpu_count() - 1,
            pin_memory=True,
            persistent_workers=True,
        )
        train_holdout_dl = DataLoader(
            train_dataset,
            batch_size=min(128, len(train_dataset)),
            generator=g,
            drop_last=False,
            sampler=train_holdout_idx,
            num_workers=os.cpu_count() - 1,
            pin_memory=True,
            persistent_workers=True,
        )
        trainer.train(
            train_dl,
            train_holdout_dl,
            evaluation_metric=tuner.evaluation_metric,
            multi_class=tuner.multi_class,
        )
    else:
        train_dl = DataLoader(
            train_dataset,
            batch_size=min(128, len(train_dataset)),
            generator=g,
            drop_last=False,
            shuffle=True,
            num_workers=os.cpu_count() - 1,
            pin_memory=True,
            persistent_workers=True,
        )
        trainer.train(
            train_dl,
            evaluation_metric=tuner.evaluation_metric,
            multi_class=tuner.multi_class,
        )
    trainer.set_best_state()

    refit_predictions, targets = trainer.model.predict(test_dl)
    refit_score = roc_auc_score(targets, refit_predictions, multi_class='ovo', average='micro')

    return score, refit_score


def baseline_main(
        args,
        X_train_val,
        X_train_folds,
        X_val_folds,
        X_test,
        y_train_val,
        y_train_folds,
        y_val_folds,
        y_test,
        categorical_idx=None,
        categorical_idx_counts=None,
        numeric_idx=None
):
    g = torch.Generator()
    g.manual_seed(args.random_state)

    tuner = BaselineTuner(
        args.runs,
        dataset.feature_names,
        dataset.target_values,
        X_train_folds,
        X_val_folds,
        y_train_folds,
        y_val_folds,
        roc_auc_score,
        args.openml_id,
        g,
        categorical_idx,
        categorical_idx_counts,
        numeric_idx
    )
    best_params = tuner.tune(args.model_name)

    if args.model_name == 'PYTORCH-NN' or args.model_name == 'PYTORCH-NN-NODA-200':

        if 'use_fbt' in best_params and best_params['use_fbt']:
            fbt = FeatureBinarizerFromTrees(
                treeNum=best_params['fbt_tree_num'],
                treeDepth=best_params['fbt_tree_depth'],
                treeFeatureSelection=best_params['fbt_tree_feature_selection'],
                threshRound=best_params['fbt_thresh_round'],
                randomState=0
            )
            # fbt = tuner.best_model.fbt

            X_train_val = pd.DataFrame(X_train_val, columns=[f'feature_{i}' for i in range(X_train_val.shape[1])])
            X_test = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
            X_train_fold = pd.DataFrame(X_train_folds[0],
                                        columns=[f'feature_{i}' for i in range(X_train_folds[0].shape[1])])

            numeric_columns = X_train_fold.columns[X_train_fold.nunique() > 2]
            categorical_columns = X_train_fold.columns[X_train_fold.nunique() <= 2]
            X_train_val_numeric = X_train_val[numeric_columns]
            X_test_numeric = X_test[numeric_columns]
            X_train_val_categorical = X_train_val[categorical_columns]
            X_test_categorical = X_test[categorical_columns]

            fbt.fit(X_train_fold[numeric_columns], y_train_folds[0])
            X_train_val_numeric = fbt.transform(X_train_val_numeric)
            X_test_numeric = fbt.transform(X_test_numeric)

            X_train_val = np.hstack([X_train_val_numeric.values, X_train_val_categorical.values])
            X_test = np.hstack([X_test_numeric.values, X_test_categorical.values])

        # reproducibility
        np.random.seed(args.random_state)
        torch.random.manual_seed(args.random_state)

        model = MlpMultiClassClassifier(
            input_size=X_test.shape[1],
            layer_sizes=[best_params['layer_size']] * best_params['n_layers'],
            output_size=y_test.shape[1],
            learning_rate=best_params['learning_rate'],
            n_epochs=200,
            weight_decay=best_params['weight_decay'] if 'weight_decay' in best_params else 0.0,
            early_stopping_plateau_count=best_params['early_stopping_plateau_count']
            if 'early_stopping_plateau_count' in best_params else 0,
            activation=best_params['activation'],
            batch_norm=best_params['batch_norm'],
            dropout_pct=best_params['dropout_pct'] if 'dropout_pct' in best_params else 0.0,
            lookahead_steps=best_params['lookahead_steps'] if 'lookahead_steps' in best_params else 0,
            lookahead_steps_size=best_params['lookahead_steps_size'] if 'lookahead_step_size' in best_params else 0.0,
            augment=best_params['augment'] if 'augment' in best_params else None,
            augment_alpha=best_params['augment_alpha'] if 'augment_alpha' in best_params else 0.0,
            swa=best_params['swa'],
            t_0=best_params['t_0'],
            t_mult=best_params['t_mult']
        )

        # build test dataset
        test_dataset = TorchDataset(X=X_test, y=y_test)
        test_dl = DataLoader(
            test_dataset,
            batch_size=128,
            generator=g,
            drop_last=False,
            shuffle=False,
            num_workers=os.cpu_count() - 1,
            pin_memory=True,
            persistent_workers=True,
        )

        # predictions, _ = tuner.best_model.predict(test_dl)

        # refit model on all data
        train_dataset = TorchDataset(X=X_train_val, y=y_train_val)

        if best_params['use_early_stopping']:
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
                generator=g,
                drop_last=False,
                sampler=train_sampler,
                num_workers=os.cpu_count() - 1,
                pin_memory=True,
                persistent_workers=True,
            )
            train_holdout_dl = DataLoader(
                train_dataset,
                batch_size=min(128, len(train_dataset)),
                generator=g,
                drop_last=False,
                sampler=train_holdout_idx,
                num_workers=os.cpu_count() - 1,
                pin_memory=True,
                persistent_workers=True,
            )
            model.fit(train_dl, train_holdout_dl)
        else:
            train_dl = DataLoader(
                train_dataset,
                batch_size=min(128, len(train_dataset)),
                generator=g,
                drop_last=False,
                shuffle=True,
                num_workers=os.cpu_count() - 1,
                pin_memory=True,
                persistent_workers=True,
            )
            model.fit(train_dl)

        refit_predictions, _ = model.predict(test_dl)

    elif args.model_name == 'FTT-NODA':

        if 'use_fbt' in best_params and best_params['use_fbt']:
            fbt = FeatureBinarizerFromTrees(
                treeNum=best_params['fbt_tree_num'],
                treeDepth=best_params['fbt_tree_depth'],
                treeFeatureSelection=best_params['fbt_tree_feature_selection'],
                threshRound=best_params['fbt_thresh_round'],
                randomState=0
            )
            # fbt = tuner.best_model.fbt

            X_train_val = pd.DataFrame(X_train_val, columns=[f'feature_{i}' for i in range(X_train_val.shape[1])])
            X_test = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
            X_train_fold = pd.DataFrame(X_train_folds[0],
                                        columns=[f'feature_{i}' for i in range(X_train_folds[0].shape[1])])

            numeric_columns = X_train_fold.columns[X_train_fold.nunique() > 2]
            categorical_columns = X_train_fold.columns[X_train_fold.nunique() <= 2]
            X_train_val_numeric = X_train_val[numeric_columns]
            X_test_numeric = X_test[numeric_columns]
            X_train_val_categorical = X_train_val[categorical_columns]
            X_test_categorical = X_test[categorical_columns]

            fbt.fit(X_train_fold[numeric_columns], y_train_folds[0])
            X_train_val_numeric = fbt.transform(X_train_val_numeric)
            X_test_numeric = fbt.transform(X_test_numeric)

            X_train_val = np.hstack([X_train_val_numeric.values, X_train_val_categorical.values])
            X_test = np.hstack([X_test_numeric.values, X_test_categorical.values])

        # reproducibility
        np.random.seed(args.random_state)
        torch.random.manual_seed(args.random_state)
        torch.manual_seed(args.random_state)
        torch.cuda.manual_seed(args.random_state)
        random.seed(args.random_state)

        model = FTTransfomerMultiClassClassifier(
            categories=categorical_idx_counts,
            num_continuous=len(numeric_idx),
            attn_dropout=best_params['attn_dropout'],
            ff_dropout=best_params['ff_dropout'],
            depth=best_params['depth'],
            heads=best_params['heads'],
            output_size=y_test.shape[1],
            learning_rate=best_params['learning_rate'],
            n_epochs=50,
            weight_decay=best_params['weight_decay'] if 'weight_decay' in best_params else 0.0,
            early_stopping_plateau_count=best_params['early_stopping_plateau_count']
            if 'early_stopping_plateau_count' in best_params else 0,
            lookahead_steps=best_params['lookahead_steps'] if 'lookahead_steps' in best_params else 0,
            lookahead_steps_size=best_params['lookahead_steps_size'] if 'lookahead_step_size' in best_params else 0.0,
            augment=best_params['augment'] if 'augment' in best_params else None,
            augment_alpha=best_params['augment_alpha'] if 'augment_alpha' in best_params else 0.0,
            t_0=best_params['t_0'],
            t_mult=best_params['t_mult']
        )

        # build test dataset
        test_dataset = TorchDataset(X=X_test, y=y_test,
                                    categorical_idxs=categorical_idx, numeric_idxs=numeric_idx)
        test_dl = DataLoader(
            test_dataset,
            batch_size=128,
            generator=g,
            drop_last=False,
            shuffle=False,
            num_workers=os.cpu_count() - 1,
            pin_memory=True,
            persistent_workers=True,
        )

        # predictions, _ = tuner.best_model.predict(test_dl)

        # refit model on all data
        train_dataset = TorchDataset(X=X_train_val, y=y_train_val,
                                     categorical_idxs=categorical_idx, numeric_idxs=numeric_idx)

        if best_params['use_early_stopping']:
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
                generator=g,
                drop_last=False,
                sampler=train_sampler,
                num_workers=os.cpu_count() - 1,
                pin_memory=True,
                persistent_workers=True,
            )
            train_holdout_dl = DataLoader(
                train_dataset,
                batch_size=min(128, len(train_dataset)),
                generator=g,
                drop_last=False,
                sampler=train_holdout_idx,
                num_workers=os.cpu_count() - 1,
                pin_memory=True,
                persistent_workers=True,
            )
            model.fit(train_dl, train_holdout_dl)
        else:
            train_dl = DataLoader(
                train_dataset,
                batch_size=min(128, len(train_dataset)),
                generator=g,
                drop_last=False,
                shuffle=True,
                num_workers=os.cpu_count() - 1,
                pin_memory=True,
                persistent_workers=True,
            )
            model.fit(train_dl)

        refit_predictions, _ = model.predict(test_dl)

    elif args.model_name == "NAM-HO":

        # reproducibility
        np.random.seed(args.random_state)
        torch.random.manual_seed(args.random_state)
        torch.manual_seed(args.random_state)
        torch.cuda.manual_seed(args.random_state)
        random.seed(args.random_state)

        tf_seed = args.random_state
        training_epochs = 1000
        batch_size = 1024
        decay_rate = 0.995
        num_basis_functions = 1000
        units_multiplier = 2
        shallow = False
        early_stopping_epochs = 60

        model = NAM(
            training_epochs=training_epochs,
            learning_rate=best_params['learning_rate'],
            output_regularization=best_params['output_regularization'],
            l2_regularization=best_params['l2_regularization'],
            batch_size=batch_size,
            decay_rate=decay_rate,
            dropout=best_params['dropout'],
            tf_seed=tf_seed,
            feature_dropout=best_params['feature_dropout'],
            num_basis_functions=num_basis_functions,
            units_multiplier=units_multiplier,
            shallow=shallow,
            early_stopping_epochs=early_stopping_epochs
        )

        train_size = len(X_train_val)
        indices = list(range(train_size))
        np.random.seed(0)
        np.random.shuffle(indices)
        train_holdout_split_index = int(np.floor(0.2 * train_size))
        train_idx, train_holdout_idx = indices[train_holdout_split_index:], indices[:train_holdout_split_index]
        X_train = X_train_val[train_idx]
        y_train = y_train_val[train_idx]
        X_holdout = X_train_val[train_holdout_idx]
        y_holdout = y_train_val[train_holdout_idx]

        test_performance = model.testing(X_train, y_train, X_holdout, y_holdout, X_test, y_test)

        # cleanup
        # shutil.rmtree(model.FLAGS['logdir'])

    elif args.model_name == "DANet-200":
        seed = args.random_state
        max_epochs = 200

        model = DANet(
            max_epochs=max_epochs,
            patience=best_params['patience'],
            lr=best_params['lr'],
            layer=best_params['layer'],
            base_outdim=best_params['base_outdim'],
            k=best_params['k'],
            drop_rate=best_params['drop_rate'],
            seed=seed
        )

        train_size = len(X_train_val)
        indices = list(range(train_size))
        np.random.seed(0)
        np.random.shuffle(indices)
        train_holdout_split_index = int(np.floor(0.2 * train_size))
        train_idx, train_holdout_idx = indices[train_holdout_split_index:], indices[:train_holdout_split_index]
        X_train = X_train_val[train_idx]
        y_train = y_train_val[train_idx]
        X_holdout = X_train_val[train_holdout_idx]
        y_holdout = y_train_val[train_holdout_idx]

        model.fit(X_train, y_train.ravel(), X_holdout, y_holdout.ravel())
        refit_predictions = model.predict(X_test)

        # cleanup
        shutil.rmtree(model.clf.log.log_dir)

    elif args.model_name == 'BRCG':
        iterMax = 100
        timeMax = 120
        eps = 1e-06
        solver = 'ECOS'

        model = BRCG(
            lambda0=best_params['lambda0'],
            lambda1=best_params['lambda1'],
            CNF=best_params['cnf'],
            K=best_params['k'],
            D=best_params['d'],
            B=best_params['b'],
            iterMax=iterMax,
            timeMax=timeMax,
            eps=eps,
            solver=solver
        )

        fbt = FeatureBinarizerFromTrees(
            treeNum=best_params['fbt_tree_num'],
            treeDepth=best_params['fbt_tree_depth'],
            treeFeatureSelection=best_params['fbt_tree_feature_selection'],
            threshRound=best_params['fbt_thresh_round'],
            randomState=0
        )

        X_train_val = pd.DataFrame(X_train_val, columns=[f'feature_{i}' for i in range(X_train_val.shape[1])])
        X_test = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
        X_train_fold = pd.DataFrame(X_train_folds[0],
                                    columns=[f'feature_{i}' for i in range(X_train_folds[0].shape[1])])

        numeric_columns = X_train_fold.columns[X_train_fold.nunique() > 2]
        categorical_columns = X_train_fold.columns[X_train_fold.nunique() <= 2]
        X_train_val_numeric = X_train_val[numeric_columns]
        X_test_numeric = X_test[numeric_columns]
        X_train_val_categorical = X_train_val[categorical_columns]
        X_test_categorical = X_test[categorical_columns]

        fbt.fit(X_train_fold[numeric_columns], y_train_folds[0])
        X_train_val_numeric = fbt.transform(X_train_val_numeric)
        X_test_numeric = fbt.transform(X_test_numeric)

        X_train_val = np.hstack([X_train_val_numeric.values, X_train_val_categorical.values])
        X_test = np.hstack([X_test_numeric.values, X_test_categorical.values])

        model.fit(X_train_val, y_train_val)
        refit_predictions = model.predict(X_test)

    elif args.model_name == "Cofrnet":
        # reproducibility
        np.random.seed(args.random_state)
        torch.random.manual_seed(args.random_state)
        torch.manual_seed(args.random_state)
        torch.cuda.manual_seed(args.random_state)
        random.seed(args.random_state)

        epochs = 200
        model = Cofrnet(
            network_depth=best_params['network_depth'],
            variant=best_params['variant'],
            input_size=X_train_val.shape[1],
            output_size=2 if y_train_val.shape[1] == 1 else y_train_val.shape[1],
            lr=best_params['lr'],
            momentum=best_params['momentum'],
            epochs=epochs,
            early_stopping_plateau_count=best_params['early_stopping_plateau_count'],
            weight_decay=best_params['weight_decay']
        )

        train_size = len(X_train_val)
        indices = list(range(train_size))
        np.random.seed(0)
        np.random.shuffle(indices)
        train_holdout_split_index = int(np.floor(0.2 * train_size))
        train_idx, train_holdout_idx = indices[train_holdout_split_index:], indices[:train_holdout_split_index]
        X_train = X_train_val[train_idx]
        y_train = y_train_val[train_idx]
        X_holdout = X_train_val[train_holdout_idx]
        y_holdout = y_train_val[train_holdout_idx]

        model.train(X_train, y_train, X_holdout, y_holdout)
        if y_train_val.shape[1] == 1:
            refit_predictions = model.predict(X_test)[:, -1]
        else:
            refit_predictions = model.predict(X_test)

    elif args.model_name == "Difflogic":
        # reproducibility
        np.random.seed(args.random_state)
        torch.random.manual_seed(args.random_state)
        torch.manual_seed(args.random_state)
        torch.cuda.manual_seed(args.random_state)
        random.seed(args.random_state)

        seed = args.random_state
        batch_size = 128
        num_iterations = 200
        model = DiffLogic(
            input_dim=X_train_val.shape[1],
            class_count=y_train_val.shape[1],
            num_neurons=best_params['num_neurons'],
            num_layers=best_params['num_layers'],
            tau=best_params['tau'],
            seed=seed,
            batch_size=batch_size,
            learning_rate=best_params['learning_rate'],
            training_bit_count=best_params['training_bit_count'],
            implementation='cuda' if torch.cuda.is_available() else 'python',
            num_iterations=num_iterations,
            connections=best_params['connections'],
            grad_factor=best_params['grad_factor'],
            eval_freq=20
        )

        # build test dataset
        test_dataset = TorchDataset(X=X_test, y=y_test)
        test_dl = DataLoader(
            test_dataset,
            batch_size=128,
            generator=g,
            drop_last=False,
            shuffle=False,
            num_workers=os.cpu_count() - 1,
            pin_memory=True,
            persistent_workers=True,
        )

        # refit model on all data
        train_dataset = TorchDataset(X=X_train_val, y=y_train_val)

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
            generator=g,
            drop_last=False,
            sampler=train_sampler,
            num_workers=os.cpu_count() - 1,
            pin_memory=True,
            persistent_workers=True,
        )
        train_holdout_dl = DataLoader(
            train_dataset,
            batch_size=min(128, len(train_dataset)),
            generator=g,
            drop_last=False,
            sampler=train_holdout_idx,
            num_workers=os.cpu_count() - 1,
            pin_memory=True,
            persistent_workers=True,
        )
        model.train(train_dl, train_holdout_dl)
        refit_predictions = model.predict(test_dl).cpu()

    else:
        # reproducibility
        np.random.seed(args.random_state)

        best_model = generate_model(args.model_name, best_params, args.random_state)
        best_model.fit(X_train_val, y_train_val)
        refit_predictions = best_model.predict(X_test)

    targets = y_test

    if args.model_name != "NAM-HO":
        return 0.0, roc_auc_score(targets, refit_predictions, multi_class='ovo', average='micro')
    return 0.0, test_performance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--openml_id", type=int)
    parser.add_argument("-t", "--train_size", type=float, default=0.6)
    parser.add_argument("-e", "--test_size", type=float, default=0.5)
    parser.add_argument("-s", "--random_state", type=int, default=42)
    parser.add_argument("-r", "--runs", type=int, default=400)
    parser.add_argument("-m", "--model_name", type=str, default="BanditRRNNODA")
    args = parser.parse_args()

    dataset = OpemlMLDataset(
        args.openml_id, args.train_size, args.test_size, args.random_state
    )

    if args.model_name != 'FTT-NODA':
        feature_encoder = FeatureEncoder(
            numerical_features=dataset.numerical_features,
            categorical_features=dataset.categorical_features,
            ordinal_features=dataset.ordinal_features,
            label_encode_categorical=False
        )
    else:
        feature_encoder = FeatureEncoder(
            numerical_features=dataset.numerical_features,
            categorical_features=dataset.categorical_features,
            ordinal_features=dataset.ordinal_features,
            label_encode_categorical=True
        )

    if args.model_name != 'FTT-NODA':
        X_train_val = feature_encoder.fit_transform(dataset.X_train_val)
        X_train_folds = [feature_encoder.transform(df) for df in dataset.X_train_folds]
        X_val_folds = [feature_encoder.transform(df) for df in dataset.X_val_folds]
        X_test = feature_encoder.transform(dataset.X_test)
    else:
        X_train_val = dataset.X_train_val
        X_train_folds = [df for df in dataset.X_train_folds]
        X_val_folds = [df for df in dataset.X_val_folds]
        X_test = dataset.X_test

    mms = MinMaxScaler()
    qt = QuantileTransformer(random_state=55688, output_distribution='normal')

    if args.model_name not in ['FTT-NODA', 'DANet', 'DANet-200']:
        X_train_val = mms.fit_transform(X_train_val)
        X_train_folds = [mms.transform(df) for df in X_train_folds]
        X_val_folds = [mms.transform(df) for df in X_val_folds]
        X_test = mms.transform(X_test)
    elif args.model_name in ['DANet', 'DANet-200']:
        X_train_val = qt.fit_transform(X_train_val)
        X_train_folds = [qt.transform(df) for df in X_train_folds]
        X_val_folds = [qt.transform(df) for df in X_val_folds]
        X_test = qt.transform(X_test)
    else:
        # custom processing for specific datasets and the FTT algorithm because of how FTT handles categorical data
        if args.openml_id == 44158:
            X_train_val['Var218'] = X_train_val['Var218'].astype('category').cat.codes
            new_x_train_folds = []
            for df in X_train_folds:
                df['Var218'] = df['Var218'].astype('category').cat.codes
                new_x_train_folds += [df]
            X_train_folds = new_x_train_folds
            new_x_val_folds = []
            for df in X_val_folds:
                df['Var218'] = df['Var218'].astype('category').cat.codes
                new_x_val_folds += [df]
            X_val_folds = new_x_val_folds
            X_test['Var218'] = X_test['Var218'].astype('category').cat.codes
        elif args.openml_id == 44161:
            for feat in dataset.ordinal_features:
                X_train_val[feat] = X_train_val[feat].astype('category').cat.codes
                new_x_train_folds = []
                for df in X_train_folds:
                    df[feat] = df[feat].astype('category').cat.codes
                    new_x_train_folds += [df]
                X_train_folds = new_x_train_folds
                new_x_val_folds = []
                for df in X_val_folds:
                    df[feat] = df[feat].astype('category').cat.codes
                    new_x_val_folds += [df]
                X_val_folds = new_x_val_folds
                X_test[feat] = X_test[feat].astype('category').cat.codes

        X_train_val = X_train_val.astype(float).values
        X_train_folds = [df.astype(float).values for df in X_train_folds]
        X_val_folds = [df.astype(float).values for df in X_val_folds]
        X_test = X_test.astype(float).values

    y_train_val = dataset.y_train_val
    y_train_folds = dataset.y_train_folds
    y_val_folds = dataset.y_val_folds
    y_test = dataset.y_test

    print(
        f"Train Size: {X_train_folds[0].shape}\n"
        f"Validation Size: {X_val_folds[0].shape}"
        f"\nTest Size{X_test.shape}"
    )

    if args.model_name == "BanditRRNNODA":
        score, refit_score = bandit_rrn_noda_main(
            args, X_train_val, X_train_folds, X_val_folds, X_test,
            y_train_val, y_train_folds, y_val_folds, y_test)
    else:
        score, refit_score = baseline_main(args, X_train_val, X_train_folds, X_val_folds, X_test,
                                           y_train_val, y_train_folds, y_val_folds, y_test,
                                           dataset.non_numeric_idxs, dataset.non_numeric_idxs_counts,
                                           dataset.numeric_idxs)

    results = pd.DataFrame.from_dict(
        {
            "dataset": [dataset.dataset.name],
            "model": [args.model_name],
            "roc_auc_score": [score],
            "refit_roc_auc_score": [refit_score],
            "openml_id": [args.openml_id],
            "train_size": [args.train_size],
            "test_size": [args.test_size],
            "random_state": [args.random_state],
            "runs": [args.runs],
            "model_name": [args.model_name],
            "n_ensembles": [args.n_ensembles]
        }
    )

    print(results)

    out_file = f"{args.model_name}_{dataset.dataset.name}_{args.openml_id}_attn_nrn_NAM_ish_big.csv"

    if os.path.isfile(out_file):
        results = pd.concat([pd.read_csv(out_file), results])

    results.to_csv(out_file, index=False)
