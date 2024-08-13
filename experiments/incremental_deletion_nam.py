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
from src.nam import NAM

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


def partition(lst, batch_size):
    lst_len = len(lst)
    index = 0
    while index < lst_len:
        x = lst[index: batch_size + index]
        index += batch_size
        yield x


def generate_predictions(gen, m):
    y_pred = []
    while True:
        try:
            x = next(gen)
            pred = m(x).numpy()
            y_pred.extend(pred)
        except:
            break
    return y_pred


def get_test_predictions(m, x_test, batch_size=256):
    batch_size = min(batch_size, x_test.shape[0])
    generator = partition(x_test, batch_size)
    return generate_predictions(generator, m)


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
        y_pred = get_test_predictions(m, X_test_shuffled.values)
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
        y_pred = get_test_predictions(m, X_test_del.values)

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
        y_pred = get_test_predictions(m, X_test_sd.values)
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

    if args.model_name == "NAM-HO":

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

        tf.compat.v1.reset_default_graph()
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

        score = model.testing(X_train, y_train, X_holdout, y_holdout, X_test, y_test)

    else:
        raise AssertionError("Can only explain NAM-HO")

    # Generate explanations
    s = datetime.datetime.now()
    tf.compat.v1.reset_default_graph()
    nn_model = graph_builder.create_nam_model(
        x_train=X_train,
        dropout=best_params['dropout'],
        feature_dropout=best_params['feature_dropout'],
        units_multiplier=units_multiplier,
        num_basis_functions=num_basis_functions,
        activation='exu',
        trainable=False,
        shallow=False,
        name_scope='model_0')

    _ = nn_model(X_train[:1])
    nn_model.summary()

    # @title Restore checkpoint
    logdir = model.FLAGS['logdir']
    ckpt_dir = osp.join(logdir, 'model_0', 'best_checkpoint')
    ckpt_files = sorted(tf.io.gfile.listdir(ckpt_dir))
    ckpt = osp.join(ckpt_dir, ckpt_files[0].split('.data')[0])
    ckpt_reader = tf.train.load_checkpoint(ckpt)
    print(ckpt_reader.get_variable_to_dtype_map())

    for var in nn_model.variables:
        print(var.name)
        tensor_name = var.name.split(':', 1)[0].replace('nam', 'model_0/nam')
        value = ckpt_reader.get_tensor(tensor_name)
        var.assign(value)

    # @title Helper functions for generating predictions

    def partition(lst, batch_size):
        lst_len = len(lst)
        index = 0
        while index < lst_len:
            x = lst[index: batch_size + index]
            index += batch_size
            yield x

    def generate_predictions(gen, nn_model):
        y_pred = []
        while True:
            try:
                x = next(gen)
                pred = nn_model(x).numpy()
                y_pred.extend(pred)
            except:
                break
        return y_pred

    def get_test_predictions(nn_model, x_test, batch_size=256):
        batch_size = min(batch_size, x_test.shape[0])
        generator = partition(x_test, batch_size)
        return generate_predictions(generator, nn_model)

    def get_feature_predictions_old(nn_model, features, batch_size=256):
        """Get feature predictions for unique values for each feature."""
        unique_feature_pred, unique_feature_gen = [], []
        for i, feature in enumerate(features):
            batch_size = min(batch_size, feature.shape[0])
            generator = partition(feature, batch_size)
            feature_pred = lambda x: nn_model.feature_nns[i](
                x, training=nn_model._false)  # pylint: disable=protected-access
            unique_feature_gen.append(generator)
            unique_feature_pred.append(feature_pred)

        feature_predictions = [
            generate_predictions(generator, feature_pred) for
            feature_pred, generator in zip(unique_feature_pred, unique_feature_gen)
        ]
        feature_predictions = [np.array(x) for x in feature_predictions]
        return feature_predictions

    def get_feature_predictions(nn_model, x_data):
        feature_predictions = []
        unique_features = compute_features(x_data)
        for c, i in enumerate(unique_features):
            f_preds = nn_model.feature_nns[c](i, training=nn_model._false)
            feature_predictions.append(f_preds)
        return feature_predictions

    def compute_features(x_data):
        # x_data, _, _ = data_utils.load_dataset(dataset_name)
        single_features = np.split(x_data, x_data.shape[1], axis=1)
        unique_features = [np.unique(f, axis=0) for f in single_features]
        return unique_features

    # @title Dataset helpers

    def load_col_min_max(x_data, column_names):
        x = pd.DataFrame(x_data, columns=column_names)
        col_min_max = {}
        for col in x:
            unique_vals = x[col].unique()
            col_min_max[col] = (np.min(unique_vals), np.max(unique_vals))
        return col_min_max

    def inverse_min_max_scaler(x, min_val, max_val):
        return (x + 1) / 2 * (max_val - min_val) + min_val

    # @title Calculate individual feature and test predictions

    test_predictions = get_test_predictions(nn_model, X_test)
    unique_features = compute_features(X_train_val)
    feature_predictions = get_feature_predictions(nn_model, X_train_val)

    print("Ytest", y_test)

    test_metric = graph_builder.calculate_metric(
        y_test, test_predictions, regression=False)
    metric_str = 'AUROC'
    print(f'{metric_str}: {test_metric}')

    # @title Individual arrays for each Dataset feature

    NUM_FEATURES = X_train_val.shape[1]
    SINGLE_FEATURES = np.split(X_train_val, NUM_FEATURES, axis=1)
    UNIQUE_FEATURES = [np.unique(x, axis=0) for x in SINGLE_FEATURES]

    column_names = column_names
    col_min_max = load_col_min_max(X_train_val, column_names)

    SINGLE_FEATURES_ORIGINAL = {}
    UNIQUE_FEATURES_ORIGINAL = {}
    for i, col in enumerate(column_names):
        min_val, max_val = col_min_max[col]
        UNIQUE_FEATURES_ORIGINAL[col] = inverse_min_max_scaler(
            UNIQUE_FEATURES[i][:, 0], min_val, max_val)
        SINGLE_FEATURES_ORIGINAL[col] = inverse_min_max_scaler(
            SINGLE_FEATURES[i][:, 0], min_val, max_val)

    avg_hist_data = {col: predictions for col, predictions in zip(column_names, feature_predictions)}

    correct_y_prediction_index = 2
    for i, (label, prediction) in enumerate(zip(y_test, test_predictions)):
        if label == prediction and label == 1:
            correct_y_prediction_index = i
            print(label, prediction, "CORRECT")
            break

    # @title Calculate the mean prediction

    ALL_INDICES = {}
    MEAN_PRED = {}

    for i, col in enumerate(column_names):
        x_i = X_test[:, i]
        ALL_INDICES[col] = np.searchsorted(UNIQUE_FEATURES[i][:, 0], x_i, 'left')

    for col in column_names:
        avg_hist_list = []
        for i in ALL_INDICES[col]:
            try:
                avg_hist_list += [avg_hist_data[col][i]]
            except Exception as e:
                print(e)

        mean_pred_list = []
        for i in ALL_INDICES[col]:
            try:
                mean_pred_list += [avg_hist_data[col][i]]
            except Exception as e:
                print(e)
        MEAN_PRED[col] = np.mean(mean_pred_list)
        # MEAN_PRED[col] = np.mean([avg_hist_data[col][i] for i in ALL_INDICES[col]])

    # @title Helpers for MEAN feature importance

    def compute_mean_feature_importance(avg_hist_data):
        mean_abs_score = {}
        for k in avg_hist_data:
            mean_abs_score[k] = np.mean(np.abs(avg_hist_data[k] - MEAN_PRED[k]))
        x1, x2 = zip(*mean_abs_score.items())
        return x1, x2

    feature_names, feature_importances = compute_mean_feature_importance(avg_hist_data)
    print(feature_names, feature_importances)
    e = datetime.datetime.now()
    feature_importance_compute_time = e - s

    #### PERFORM INCREMENTAL DELETION ANALYSIS FROM HERE
    sorted_feature_names = [x for _, x in sorted(zip(feature_importances, feature_names), reverse=True)]
    sorted_feature_indexes = [x for _, x in sorted(zip(feature_importances, np.arange(len(feature_names))), reverse=True)]
    sorted_importances = sorted(feature_importances, reverse=True)
    # sorted_indices = feature_importances.argsort()[::-1]
    # sorted_feature_names = feature_names[sorted_indices]
    # sorted_importances = feature_importances[sorted_indices]
    # zip_fi = list(zip(sorted_feature_names, sorted_feature_indexes, sorted_importances))
    zip_fi = list(zip(sorted_feature_names, sorted_importances))

    addition_results = incr_addition(nn_model, X_test_df, y_test, zip_fi, score)
    deletion_results = incr_deletion(nn_model, X_test_df, y_test, zip_fi, score)
    single_deletion_results = single_deletion(nn_model, X_test_df, y_test, zip_fi, score)

    results = {
        "openml_id": openml_id,
        "roc_auc_score": float(score),
        "feature_importances": [{"feature": f, "importance": float(i)} for f, i in zip_fi],
        "incr_addition": addition_results,
        "incr_deletion": deletion_results,
        "single_deletion": single_deletion_results,
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
    parser.add_argument("-m", "--model_name", type=str, default="NAM-HO")
    args = parser.parse_args()

    all_results = []
    for openml_id in BENCHMARK_DATASETS:

        if os.path.isfile('./plots/nam_single_deletion.csv'):
            current_results = pd.read_csv('./plots/nam_single_deletion.csv')
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

        if args.model_name == "NAM-HO":
            single_deletion_results = baseline_main(args, X_train_val, X_train_folds, X_val_folds, X_test,
                                               y_train_val, y_train_folds, y_val_folds, y_test,
                                               dataset.non_numeric_idxs, dataset.non_numeric_idxs_counts,
                                               dataset.numeric_idxs, column_names, openml_id)
            single_deletion_results['openml_id'] = openml_id
            if os.path.isfile('./plots/nam_single_deletion.csv'):
                single_deletion_results.to_csv('./plots/nam_single_deletion.csv', index=False, header=False, mode='a+')
            else:
                single_deletion_results.to_csv('./plots/nam_single_deletion.csv', index=False)
        else:
            raise ValueError("Can only explain NAM-HO with this script")
