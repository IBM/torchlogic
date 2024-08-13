import os
import copy
import json
import argparse
import warnings
import random
import shutil
import datetime

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

from src.encoders import FeatureEncoder
from src.tuners import BaselineTuner
from src.datasets import OpemlMLDataset

import tensorflow.compat.v2 as tf
import tensorflow
tf.enable_v2_behavior()

from  src.neural_additive_models import graph_builder
import os.path as osp


def set_seeds():
    tf.random.set_seed(0)
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


def incr_addition(m, X_test, y_test, feature_importances, start_score):
    # feature_importances is List[Tuple[str,float]]
    results = []
    features = []

    prev_score = start_score

    X_test_shuffled = X_test.copy()
    X_test_shuffled = X_test_shuffled.sample(frac=1)

    for feature, importance in feature_importances:
        features.append(feature)
        X_test_shuffled[feature] = X_test[feature].copy().to_numpy()
        set_seeds()
        y_pred = m.predict(X_test_shuffled.values)
        score = roc_auc_score(y_test, y_pred, multi_class="ovo", average="micro")

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


def incr_deletion(m, X_test, y_test, feature_importances, start_score):
    # feature_importances is List[Tuple[str,float]]
    results = []
    features = []

    prev_score = start_score

    X_test_del = X_test.copy()

    for feature, importance in feature_importances:
        features.append(feature)
        X_test_del[feature] = X_test_del[feature].sample(frac=1).to_numpy()
        set_seeds()
        y_pred = m.predict(X_test_del.values)

        score = roc_auc_score(y_test, y_pred, multi_class="ovo", average="micro")

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


def single_deletion(m, X_test, y_test, feature_importances, start_score):
    # feature_importances is List[Tuple[str,float]]
    results = []
    features = []

    for feature, importance in feature_importances:
        X_test_sd = X_test.copy()
        X_test_sd[feature] = X_test_sd[feature].sample(frac=1).to_numpy()
        set_seeds()
        y_pred = m.predict(X_test_sd.values)
        score = roc_auc_score(y_test, y_pred, multi_class="ovo", average="micro")

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
        numeric_idx=None,
        column_names=None,
        openml_id=None
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

    X_test_df = X_test.copy()
    X_test = X_test.astype(float).values

    if args.model_name == "RF":
        # reproducibility
        np.random.seed(args.random_state)

        # predictions = tuner.best_model.predict(X_test)
        # TODO: you need to recreate the model based on the best parameters instead of cloning because
        #  when using journal storage teh best model in the tuner class is likely not the best model overall
        best_model = generate_model(args.model_name, best_params, args.random_state)
        best_model.fit(X_train_val, y_train_val)
        refit_predictions = best_model.predict(X_test)

    else:
        raise AssertionError("Can only explain RF")

    targets = y_test
    score = roc_auc_score(targets, refit_predictions, multi_class='ovo', average='micro')

    #### PERFORM INCREMENTAL DELETION ANALYSIS FROM HERE
    s = datetime.datetime.now()
    feature_importances = best_model.feature_importances_
    e = datetime.datetime.now()
    feature_importance_compute_time = e - s
    feature_names = np.array(X_test_df.columns)

    sorted_indices = feature_importances.argsort()[::-1]
    sorted_feature_names = feature_names[sorted_indices]
    sorted_importances = feature_importances[sorted_indices]
    zip_fi = list(zip(sorted_feature_names, sorted_importances))

    addition_results = incr_addition(best_model, X_test_df, y_test, zip_fi, score)
    deletion_results = incr_deletion(best_model, X_test_df, y_test, zip_fi, score)
    single_deletion_results = single_deletion(best_model, X_test_df, y_test, zip_fi, score)

    results = {
        "openml_id": openml_id,
        "roc_auc_score": float(score),
        "feature_importances": [{"feature": f, "importance": float(i)} for f, i in zip_fi],
        "incr_addition": addition_results,
        "incr_deletion": deletion_results,
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
    parser.add_argument("-m", "--model_name", type=str, default="RF")
    args = parser.parse_args()

    all_results = []
    for openml_id in BENCHMARK_DATASETS:

        if os.path.isfile('./plots/RF_single_deletion.csv'):
            current_results = pd.read_csv('./plots/RF_single_deletion.csv')
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

        X_test = pd.DataFrame(X_test, columns=column_names)

        print(
            f"Train Size: {X_train_folds[0].shape}\n"
            f"Validation Size: {X_val_folds[0].shape}"
            f"\nTest Size{X_test.shape}"
        )

        if args.model_name == "RF":
            single_deletion_results = baseline_main(args, X_train_val, X_train_folds, X_val_folds, X_test,
                                               y_train_val, y_train_folds, y_val_folds, y_test,
                                               dataset.non_numeric_idxs, dataset.non_numeric_idxs_counts,
                                               dataset.numeric_idxs, column_names, openml_id)
            single_deletion_results['openml_id'] = openml_id
            if os.path.isfile('./plots/RF_single_deletion.csv'):
                single_deletion_results.to_csv('./plots/RF_single_deletion.csv', index=False, header=False, mode='a+')
            else:
                single_deletion_results.to_csv('./plots/RF_single_deletion.csv', index=False)
        else:
            raise ValueError("Can only explain RF with this script")
