import datetime
import os
import copy
import json
import argparse
import warnings
import random
import shutil

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler
torch.multiprocessing.set_sharing_strategy('file_system')

from torchlogic.models import BanditNRNClassifier
from torchlogic.utils.trainers import BanditNRNTrainer

from aix360.algorithms.rbm import FeatureBinarizerFromTrees
from src.tuners import BanditRRNNODATuner
from src.datasets import TorchDataset
from minepy import cstats

from src.encoders import FeatureEncoder
from src.datasets import OpemlMLDataset


def set_seeds():
    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)


set_seeds()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# https://arxiv.org/pdf/2207.08815.pdf
BENCHMARK_DATASETS = [44120, 44121, 44122, 44123, 44124, 44125, 44126, 44127, 44128, 44129, 44130, 44131,
                      44089, 44090, 44091, 44156, 44157, 44158, 44159, 44160, 44161, 44162]

pd.set_option("display.max_columns", 100)


def incr_addition(m, best_params, X_train_val, X_test, y_test, feature_importances, start_score):
    # feature_importances is List[Tuple[str,float]]
    results = []
    features = []

    prev_score = start_score

    X_test_orig = X_test.copy()

    X_test_shuffled = X_test.copy()
    X_test_shuffled = X_test_shuffled.sample(frac=1)

    for feature, importance in feature_importances:
        features.append(feature)
        X_test_shuffled[feature] = X_test[feature].copy().to_numpy()
        set_seeds()

        g = torch.Generator()
        g.manual_seed(args.random_state)

        if 'use_fbt' in best_params and best_params['use_fbt']:
            fbt = FeatureBinarizerFromTrees(
                treeNum=best_params['fbt_tree_num'],
                treeDepth=best_params['fbt_tree_depth'],
                treeFeatureSelection=best_params['fbt_tree_feature_selection'],
                threshRound=best_params['fbt_thresh_round'],
                randomState=0
            )
            # fbt = tuner.best_fbt

            X_train_val = pd.DataFrame(X_train_val, columns=X_test_orig.columns)
            X_test = pd.DataFrame(X_test, columns=X_test_orig.columns)
            X_train_fold = pd.DataFrame(X_train_folds[0],
                                        columns=X_test_orig.columns)

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
        test_dataset = TorchDataset(X=X_test_shuffled.values, y=y_test)
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

        y_pred, y_test_out = m.predict(test_dl)
        score = roc_auc_score(y_test_out, y_pred, multi_class="ovo", average="micro")

        if prev_score:
            diff = np.abs(prev_score - score)
            ratio = score / prev_score
        else:
            diff = 0.0
            ratio = 1.0

        prev_score = score

        results.append(
            {
                "features_added": copy.copy(features),
                "addition": feature,
                "importance": float(importance),
                "roc_auc_score": float(score),
                "diff": float(diff),
                "ratio": float(ratio),
            }
        )
    return results


def incr_deletion(m, best_params, X_train_val, X_test, y_test, feature_importances, start_score):
    # feature_importances is List[Tuple[str,float]]
    results = []
    features = []

    prev_score = start_score

    X_test_orig = X_test.copy()

    X_test_del = X_test.copy()

    for feature, importance in feature_importances:
        features.append(feature)
        X_test_del[feature] = X_test_del[feature].sample(frac=1).to_numpy()
        set_seeds()

        g = torch.Generator()
        g.manual_seed(args.random_state)

        if 'use_fbt' in best_params and best_params['use_fbt']:
            fbt = FeatureBinarizerFromTrees(
                treeNum=best_params['fbt_tree_num'],
                treeDepth=best_params['fbt_tree_depth'],
                treeFeatureSelection=best_params['fbt_tree_feature_selection'],
                threshRound=best_params['fbt_thresh_round'],
                randomState=0
            )
            # fbt = tuner.best_fbt

            X_train_val = pd.DataFrame(X_train_val, columns=X_test_orig.columns)
            X_test = pd.DataFrame(X_test, columns=X_test_orig.columns)
            X_train_fold = pd.DataFrame(X_train_folds[0],
                                        columns=X_test_orig.columns)

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
        test_dataset = TorchDataset(X=X_test_del.values, y=y_test)
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

        y_pred, y_test_out = m.predict(test_dl)

        score = roc_auc_score(y_test_out, y_pred, multi_class="ovo", average="micro")

        if prev_score:
            diff = np.abs(prev_score - score)
            ratio = score / prev_score
        else:
            diff = 0.0
            ratio = 1.0

        prev_score = score

        results.append(
            {
                "features_deleted":  copy.copy(features),
                "deletion": feature,
                "importance": float(importance),
                "roc_auc_score": float(score),
                "diff": float(diff),
                "ratio": float(ratio),
            }
        )
    return results


def single_deletion(m, best_params, X_train_val, X_test, y_test, feature_importances, start_score):
    # feature_importances is List[Tuple[str,float]]
    results = []
    features = []

    X_test_orig = X_test.copy()
    X_train_val_orig = X_train_val.copy()
    print("COL", X_test_orig.columns)

    for feature, importance in feature_importances:
        X_test_sd = X_test_orig.copy()
        X_test_sd[feature] = X_test_sd[feature].sample(frac=1).to_numpy()
        X_train_val = X_train_val_orig.copy()
        set_seeds()

        g = torch.Generator()
        g.manual_seed(args.random_state)

        if 'use_fbt' in best_params and best_params['use_fbt']:
            fbt = FeatureBinarizerFromTrees(
                treeNum=best_params['fbt_tree_num'],
                treeDepth=best_params['fbt_tree_depth'],
                treeFeatureSelection=best_params['fbt_tree_feature_selection'],
                threshRound=best_params['fbt_thresh_round'],
                randomState=0
            )
            # fbt = tuner.best_fbt

            X_train_val = pd.DataFrame(X_train_val, columns=[x.capitalize() for x in X_test_orig.columns])
            X_test = pd.DataFrame(X_test_sd, columns=[x.capitalize() for x in X_test_orig.columns])
            X_train_fold = pd.DataFrame(X_train_folds[0],
                                        columns=[x.capitalize() for x in X_test_orig.columns])

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

        y_pred, y_test_out = m.predict(test_dl)
        score = roc_auc_score(y_test_out, y_pred, multi_class="ovo", average="micro")

        diff = np.abs(start_score - score)
        ratio = score / start_score

        results.append(
            {
                "deletion": feature,
                "importance": float(importance),
                "roc_auc_score": float(score),
                "diff": float(diff),
                "ratio": float(ratio),
            }
        )
    return results


def bandit_rrn_noda_main(
        args,
        X_train_val,
        X_train_folds,
        X_val_folds,
        X_test,
        y_train_val,
        y_train_folds,
        y_val_folds,
        y_test,
        importance_type
):
    g = torch.Generator()
    g.manual_seed(args.random_state)

    X_test_df = X_test.copy()
    X_train_val_df = X_train_val.copy()
    print(X_test_df.columns)

    X_test = X_test.astype(float).values

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

        X_train_val = pd.DataFrame(X_train_val, columns=[x.capitalize() for x in X_test_df.columns])
        X_test = pd.DataFrame(X_test, columns=[x.capitalize() for x in X_test_df.columns])
        X_train_fold = pd.DataFrame(X_train_folds[0], columns=[x.capitalize() for x in X_test_df.columns])

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

    # predictions, targets = tuner.best_model.predict(test_dl)
    # score = tuner.best_model.evaluate(
    #     predictions=predictions,
    #     labels=targets,
    #     output_metric=roc_auc_score,
    #     multi_class=tuner.multi_class,
    # )
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
    # refit_score = trainer.model.evaluate(
    #     predictions=refit_predictions,
    #     labels=targets,
    #     output_metric=roc_auc_score,
    #     multi_class=tuner.multi_class,
    # )
    score = roc_auc_score(targets, refit_predictions, multi_class='ovo', average='micro')

    s = datetime.datetime.now()
    feature_importances_dict = trainer.model.get_feature_importances(
        x=test_dataset[0]['features'].unsqueeze(0),
        feature_importances_type=importance_type
    )
    e = datetime.datetime.now()
    feature_importance_compute_time = e - s

    feature_importances = list(feature_importances_dict.values())
    feature_importance_feature_names = list(feature_importances_dict.keys())

    feature_importances_df = pd.DataFrame(
        {'feature_names': feature_importance_feature_names, 'importance': feature_importances})
    feature_importances_df['orig_feature_name'] = feature_importances_df['feature_names'].apply(
        lambda x: x.split("_>")[0] if '_>' in x else x.split("_<")[0])
    feature_importances_df = feature_importances_df.groupby('orig_feature_name')['importance'].sum().reset_index()
    feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)
    sorted_feature_names = feature_importances_df['orig_feature_name'].tolist()
    sorted_importances = feature_importances_df['importance'].tolist()

    #### PERFORM INCREMENTAL DELETION ANALYSIS FROM HERE
    # sorted_feature_names = [x for _, x in sorted(zip(feature_importances, feature_importance_feature_names), reverse=True)]
    # sorted_feature_indexes = [x for _, x in sorted(zip(feature_importances, np.arange(len(feature_importance_feature_names))), reverse=True)]
    # sorted_importances = sorted(feature_importances, reverse=True)
    # sorted_indices = feature_importances.argsort()[::-1]
    # sorted_feature_names = feature_names[sorted_indices]
    # sorted_importances = feature_importances[sorted_indices]
    # zip_fi = list(zip(sorted_feature_names, sorted_feature_indexes, sorted_importances))
    zip_fi = list(zip(sorted_feature_names, sorted_importances))

    nn_model = trainer.model

    # X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # addition_results = incr_addition(nn_model, best_params, X_train_val, X_test_df, y_test, zip_fi, score)
    # deletion_results = incr_deletion(nn_model, best_params, X_train_val, X_test_df, y_test, zip_fi, score)
    single_deletion_results = single_deletion(nn_model, best_params, X_train_val_df, X_test_df, y_test, zip_fi, score)

    results = {
        "openml_id": openml_id,
        "roc_auc_score": float(score),
        "feature_importances": [{"feature": f, "importance": float(i)} for f, i in zip_fi],
        # "incr_addition": addition_results,
        # "incr_deletion": deletion_results,
        "single_deletion": single_deletion_results
    }

    print(json.dumps(results, indent=2))

    df = pd.DataFrame(results['single_deletion'])
    df['feature_importance_compute_time'] = feature_importance_compute_time.seconds

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--openml_id", type=int)
    parser.add_argument("-t", "--train_size", type=float, default=0.6)
    parser.add_argument("-e", "--test_size", type=float, default=0.5)
    parser.add_argument("-s", "--random_state", type=int, default=42)
    parser.add_argument("-r", "--runs", type=int, default=400)
    parser.add_argument("-m", "--model_name", type=str, default="BanditRRNNODA")
    args = parser.parse_args()

    for importance_type in ['weight', 'weight_proportion']:
        all_results = []
        for openml_id in BENCHMARK_DATASETS:

            if os.path.isfile(f'./plots/RNRN_single_deletion_{importance_type}_type.csv'):
                current_results = pd.read_csv(f'./plots/RNRN_single_deletion_{importance_type}_type.csv')
                if openml_id in current_results['openml_id'].tolist():
                    print("skipping", openml_id)
                    continue

            args.__dict__.update({'openml_id': openml_id})

            dataset = OpemlMLDataset(
                openml_id, args.train_size, args.test_size, args.random_state
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
                X_test_columns = dataset.X_test.columns
                X_test = feature_encoder.transform(dataset.X_test)
            else:
                X_train_val = dataset.X_train_val
                X_train_folds = [df for df in dataset.X_train_folds]
                X_val_folds = [df for df in dataset.X_val_folds]
                X_test = dataset.X_test

            mms = MinMaxScaler()
            qt = QuantileTransformer(random_state=55688, output_distribution='normal')

            if args.model_name not in ['FTT-NODA', 'DANet', 'DANet-200']:
                column_names = list(X_train_val.columns)
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
                if openml_id == 44158:
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
                elif openml_id == 44161:
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
                X_test = X_test #.astype(float).values

            y_train_val = dataset.y_train_val
            y_train_folds = dataset.y_train_folds
            y_val_folds = dataset.y_val_folds
            y_test = dataset.y_test

            X_test = pd.DataFrame(X_test, columns=[x.capitalize().replace(" ", "_").replace("(", "").replace(")", "") for x in column_names])

            print(
                f"Train Size: {X_train_folds[0].shape}\n"
                f"Validation Size: {X_val_folds[0].shape}"
                f"\nTest Size{X_test.shape}"
            )

            if args.model_name == "BanditRRNNODA":
                single_deletion_results = bandit_rrn_noda_main(args, X_train_val, X_train_folds, X_val_folds, X_test,
                                                   y_train_val, y_train_folds, y_val_folds, y_test, importance_type)
                single_deletion_results['openml_id'] = openml_id
                if os.path.isfile(f'./plots/RNRN_single_deletion_{importance_type}_type.csv'):
                    single_deletion_results.to_csv(f'./plots/RNRN_single_deletion_{importance_type}_type.csv', index=False, header=False, mode='a+')
                else:
                    single_deletion_results.to_csv(f'./plots/RNRN_single_deletion_{importance_type}_type.csv', index=False)
            else:
                raise ValueError("Can only explain RNRN with this script")



# import pandas as pd
# import numpy as np
# rf_del = pd.read_csv('path/to-results/nam_ccc_single_deletion.csv')
# print("OVERALL", rf_del.groupby('openml_id')[['importance', 'diff']].corr(method='pearson')['diff'].loc[:, 'importance'].mean())
# for x in rf_del.groupby('openml_id')[['importance', 'diff']].corr(method='pearson')['diff'].loc[:, 'importance'].values:
#     print(x)
# for x in rf_del[['feature_importance_compute_time', 'openml_id']].drop_duplicates().sort_values('openml_id')['feature_importance_compute_time'].tolist():
#     print(x)


# def aggregate_results(glob_path):
#     ordering = [31, 334, 1067, 1494, 1043, 1486, 54, 11, 14, 40536, 1468, 29, 46, 22, 4538, 4134, 470, 1493, 1459, 41027,
#                 7, 40981, 934, 846, 10, 51, 25, 41159, 1169, 41143, 1567, 41147]
#     files = glob.glob(glob_path)
#     dfs = []
#     for file in files:
#         df = pd.read_csv(file)
#         df = df.drop_duplicates()
#         df = df[df['random_state'].isin([1, 2, 3, 4, 5])]
#         dfs += [df]
#     df = pd.concat(dfs)
#     df.set_index('openml_id', inplace=True)
#     df = df.reset_index()
#     assert df.shape == df.drop_duplicates(['openml_id', 'model', 'random_state']).shape
#     df = pd.pivot_table(df, index='openml_id', columns=['model', 'random_state'], values='refit_roc_auc_score')
#     df = df.loc[ordering]
#     return df
# res = aggregate_results("./ccc_results/RF*")