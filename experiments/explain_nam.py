import argparse
import warnings
import random

import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from .src.encoders import FeatureEncoder
from .src.tuners import BaselineTuner
from .src.datasets import OpemlMLDataset
from .src.nam import NAM

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

from  src.neural_additive_models import graph_builder
import os.path as osp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# https://arxiv.org/pdf/2207.08815.pdf
BENCHMARK_DATASETS = [44120, 44121, 44122, 44123, 44124, 44125, 44126, 44127, 44128, 44129, 44130, 44131,
                      44089, 44090, 44091, 44156, 44157, 44158, 44159, 44160, 44161, 44162]

pd.set_option("display.max_columns", 100)


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
        column_names=None
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

        # test_performance = model.testing(X_train, y_train, X_holdout, y_holdout, X_test, y_test)

        # cleanup
        # shutil.rmtree(model.FLAGS['logdir'])

    else:
        raise AssertionError("Can only explain NAM-HO")

    # Generate explanations

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
    logdir = './logs/98aa95ca-8002-4164-bca1-2f4c8ef98fc8'
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
    print(ALL_INDICES)

    print(avg_hist_data)

    for col in column_names:
        print(col)
        avg_hist_list = []
        for i in ALL_INDICES[col]:
            try:
                avg_hist_list += [avg_hist_data[col][i]]
            except Exception as e:
                print(e)
        MEAN_PRED[col] = np.mean(avg_hist_list)
    print(MEAN_PRED)

    # @title Helpers for MEAN feature importance

    def compute_mean_feature_importance(avg_hist_data):
        mean_abs_score = {}
        for k in avg_hist_data:
            mean_abs_score[k] = np.mean(avg_hist_data[k][correct_y_prediction_index] - MEAN_PRED[k])
        x1, x2 = zip(*mean_abs_score.items())
        return x1, x2

    x1, x2 = compute_mean_feature_importance(avg_hist_data)
    cols = list(column_names)

    def plot_mean_feature_importance(x1, x2, width=0.3):
        fig = plt.figure(figsize=(5, 4))
        ind = np.arange(len(x1))  # the x locations for the groups
        x1_indices = np.argsort(x2)
        cols_here = [cols[i] for i in x1_indices]
        # x1_here = [x12[i] for i in x1_indices]
        x2_here = [x2[i] for i in x1_indices]

        plt.bar(ind, x2_here, width, label='NAMs')
        # plt.bar(ind+width, x1_here, width, label='EBMs')
        plt.xticks(ind + width / 2, cols_here, rotation=90, fontsize='large')
        plt.ylabel('Score', fontsize='large')
        plt.legend(loc='lower right', fontsize='large')
        plt.title(f'Single Sample Importance', fontsize='large')
        plt.tight_layout()
        # plt.savefig("/your/path/here.png")
        return fig

    fig = plot_mean_feature_importance(x1, x2)

    # @title Plotting Helper Functions

    CATEGORICAL_NAMES = []

    def shade_by_density_blocks(hist_data, num_rows, num_cols,
                                n_blocks=5, color=[0.9, 0.5, 0.5],
                                feature_to_use=None):
        hist_data_pairs = list(hist_data.items())
        hist_data_pairs.sort(key=lambda x: x[0])
        min_y = np.min([np.min(a[1]) for a in hist_data_pairs])
        max_y = np.max([np.max(a[1]) for a in hist_data_pairs])
        min_max_dif = max_y - min_y
        min_y = min_y - 0.01 * min_max_dif
        max_y = max_y + 0.01 * min_max_dif

        if feature_to_use:
            hist_data_pairs = [v for v in hist_data_pairs if v[0] in feature_to_use]

        for i, (name, pred) in enumerate(hist_data_pairs):

            # unique_x_data, single_feature_data, pred = data
            unique_x_data = UNIQUE_FEATURES_ORIGINAL[name]
            single_feature_data = SINGLE_FEATURES_ORIGINAL[name]
            ax = plt.subplot(num_rows, num_cols, i + 1)
            min_x = np.min(unique_x_data)
            max_x = np.max(unique_x_data)
            x_n_blocks = min(n_blocks, len(unique_x_data))
            if name in CATEGORICAL_NAMES:
                min_x -= 0.5
                max_x += 0.5
            segments = (max_x - min_x) / x_n_blocks
            density = np.histogram(single_feature_data, bins=x_n_blocks)
            normed_density = density[0] / np.max(density[0])
            rect_params = []
            for p in range(x_n_blocks):
                start_x = min_x + segments * p
                end_x = min_x + segments * (p + 1)
                d = min(1.0, 0.01 + normed_density[p])
                rect_params.append((d, start_x, end_x))

            for param in rect_params:
                alpha, start_x, end_x = param
                rect = patches.Rectangle((start_x, min_y - 1), end_x - start_x,
                                         max_y - min_y + 1, linewidth=0.01,
                                         edgecolor=color, facecolor=color, alpha=alpha)
                ax.add_patch(rect)

    COL_NAMES = {}
    COL_NAMES['Housing'] = {x: x for x in column_names}
    dataset_name = 'Housing'
    FEATURE_LABEL_MAPPING = {}
    FEATURE_LABEL_MAPPING['Housing'] = {}

    def plot_all_hist(hist_data, num_rows, num_cols, color_base,
                      linewidth=3.0, min_y=None, max_y=None, alpha=1.0,
                      feature_to_use=None):
        hist_data_pairs = list(hist_data.items())
        hist_data_pairs.sort(key=lambda x: x[0])
        if min_y is None:
            min_y = np.min([np.min(a) for _, a in hist_data_pairs])
        if max_y is None:
            max_y = np.max([np.max(a) for _, a in hist_data_pairs])
        min_max_dif = max_y - min_y
        min_y = min_y - 0.01 * min_max_dif
        max_y = max_y + 0.01 * min_max_dif
        col_mapping = COL_NAMES[dataset_name]
        feature_mapping = FEATURE_LABEL_MAPPING[dataset_name]

        total_mean_bias = 0

        if feature_to_use:
            hist_data_pairs = [v for v in hist_data_pairs if v[0] in feature_to_use]

        for i, (name, pred) in enumerate(hist_data_pairs):
            mean_pred = MEAN_PRED[name]  # np.mean(pred)
            total_mean_bias += mean_pred
            unique_x_data = UNIQUE_FEATURES_ORIGINAL[name]
            plt.subplot(num_rows, num_cols, i + 1)

            if name in CATEGORICAL_NAMES:
                unique_x_data = np.round(unique_x_data, decimals=1)
                if len(unique_x_data) <= 2:
                    step_loc = "mid"
                else:
                    step_loc = "post"
                unique_plot_data = np.array(unique_x_data) - 0.5
                unique_plot_data[-1] += 1
                plt.step(unique_plot_data, pred - mean_pred, color=color_base,
                         linewidth=linewidth, where=step_loc, alpha=alpha)

                if name in feature_mapping:
                    labels, rot = feature_mapping[name]
                else:
                    labels = unique_x_data
                    rot = None
                plt.xticks(unique_x_data, labels=labels, fontsize='x-large', rotation=rot)
            else:
                plt.plot(unique_x_data, pred - mean_pred, color=color_base,
                         linewidth=linewidth, alpha=alpha)
                plt.xticks(fontsize='x-large')

            plt.ylim(min_y, max_y)
            plt.yticks(fontsize='x-large')
            min_x = np.min(unique_x_data)
            max_x = np.max(unique_x_data)
            if name in CATEGORICAL_NAMES:
                min_x -= 0.5
                max_x += 0.5
            plt.xlim(min_x, max_x)
            if i % num_cols == 0:
                plt.ylabel('House Price Contribution', fontsize='x-large')
            plt.xlabel(col_mapping[name], fontsize='x-large')
        return min_y, max_y

    # Generate Plots

    COLORS = [[0.9, 0.4, 0.5], [0.5, 0.9, 0.4], [0.4, 0.5, 0.9], [0.9, 0.5, 0.9]]
    NUM_COLS = 4  # @param {'type': 'integer'}
    N_BLOCKS = 20  # @param

    MIN_Y = None
    MAX_Y = None

    NUM_ROWS = int(np.ceil(NUM_FEATURES / NUM_COLS))
    fig = plt.figure(num=None, figsize=(NUM_COLS * 4.5, NUM_ROWS * 4.5),
                     facecolor='w', edgecolor='k')

    MIN_Y, MAX_Y = plot_all_hist(avg_hist_data, NUM_ROWS, NUM_COLS, COLORS[2],
                                 min_y=MIN_Y, max_y=MAX_Y, feature_to_use=column_names)
    shade_by_density_blocks(avg_hist_data, NUM_ROWS, NUM_COLS, n_blocks=N_BLOCKS, feature_to_use=column_names)

    # This is for plotting individual plots when there are multiple models
    """
    for pred in feature_predictions:
      model_hist = {col: pred[0, i] for i, col in enumerate(column_names)}
      plot_all_hist(model_hist, NUM_ROWS, NUM_COLS,
                    color_base=[0.3, 0.4, 0.9, 0.2], alpha=0.06,
                    linewidth=0.1, min_y=MIN_Y, max_y=MAX_Y, feature_to_use=features)
    """
    plt.subplots_adjust(hspace=0.23)
    plt.savefig("your/path/here.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--openml_id", type=int)
    parser.add_argument("-t", "--train_size", type=float, default=0.6)
    parser.add_argument("-e", "--test_size", type=float, default=0.5)
    parser.add_argument("-s", "--random_state", type=int, default=42)
    parser.add_argument("-r", "--runs", type=int, default=400)
    parser.add_argument("-m", "--model_name", type=str, default="BanditRRN")
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

    if args.model_name == "NAM-HO":
        baseline_main(args, X_train_val, X_train_folds, X_val_folds, X_test,
                                           y_train_val, y_train_folds, y_val_folds, y_test,
                                           dataset.non_numeric_idxs, dataset.non_numeric_idxs_counts,
                                           dataset.numeric_idxs, column_names)
    else:
        raise ValueError("Can only explain NAM-HO with this script")
