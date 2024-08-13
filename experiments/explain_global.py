import os
import joblib
import argparse
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler
torch.multiprocessing.set_sharing_strategy('file_system')

from torchlogic.models import BanditNRNClassifier
from torchlogic.utils.trainers import BanditNRNTrainer

from aix360.algorithms.rbm import FeatureBinarizerFromTrees
from minepy import cstats

from src.encoders import FeatureEncoder
from src.datasets import OpemlMLDataset, TorchDataset

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# https://arxiv.org/pdf/2207.08815.pdf
BENCHMARK_DATASETS = [44120, 44121, 44122, 44123, 44124, 44125, 44126, 44127, 44128, 44129, 44130, 44131,
                      44089, 44090, 44091, 44156, 44157, 44158, 44159, 44160, 44161, 44162]

pd.set_option("display.max_columns", 100)

def format_binarized_feature_names(binarized_column_names):
    return list(
        map(
            lambda x: str(x).replace("(", "").replace(")", "")
            .replace("/", " ").replace(",", "").replace("'", "")
            .replace(" >= ", " greater than or equal to ").replace(" < ", " less than ")
            .replace(" <= ", " less than or equal to ").replace(" > ", " greater than "),
            binarized_column_names
        )
    )

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

    study = joblib.load(args.study_path)
    best_params = study.best_params

    if 'use_fbt' in best_params and best_params['use_fbt']:
        fbt = FeatureBinarizerFromTrees(
            treeNum=best_params['fbt_tree_num'],
            treeDepth=best_params['fbt_tree_depth'],
            treeFeatureSelection=best_params['fbt_tree_feature_selection'],
            threshRound=best_params['fbt_thresh_round'],
            randomState=0
        )

        X_train_fold = X_train_folds[0]

        column_rename = {
            'MedInc': 'the median income',
            'HouseAge': 'the house age',
            'AveRooms': 'the average number of rooms',
            'AveBedrms': 'the average number of bedrooms',
            'Population': 'the population',
            'AveOccup': 'the average occupancy',
            'Latitude': 'the latitude',
            'Longitude': 'the longitude'
        }
        X_train_val.rename(columns=column_rename, inplace=True)
        X_test.rename(columns=column_rename, inplace=True)
        X_train_fold.rename(columns=column_rename, inplace=True)

        X_train_val.columns = [f"{x} was" for x in X_train_val.columns]
        X_test.columns = [f"{x} was" for x in X_test.columns]
        X_train_fold.columns = [f"{x} was" for x in X_train_fold.columns]

        numeric_columns = X_train_fold.columns[X_train_fold.nunique() > 2]
        categorical_columns = X_train_fold.columns[X_train_fold.nunique() <= 2]
        X_train_val_numeric = X_train_val[numeric_columns]
        X_test_numeric = X_test[numeric_columns]
        X_train_val_categorical = X_train_val[categorical_columns]
        X_test_categorical = X_test[categorical_columns]

        fbt.fit(X_train_fold[numeric_columns], y_train_folds[0])
        X_train_val_numeric = fbt.transform(X_train_val_numeric)
        X_test_numeric = fbt.transform(X_test_numeric)

        X_train_val_numeric.columns = format_binarized_feature_names(X_train_val_numeric.columns.to_flat_index())
        X_test_numeric.columns = format_binarized_feature_names(X_test_numeric.columns.to_flat_index())

        feature_names = list(X_train_val_numeric.columns) + list(categorical_columns)

        min_max_features_dict = {
            col: {'min': X_test.iloc[:, i].min(), 'max': X_test.iloc[:, i].max()}
            for i, col in enumerate(X_test.columns)
        }

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

    multi_class = len(dataset.target_values) > 2
    if multi_class:
        target_names = [str(x) + " class" for x in dataset.target_values]
    else:
        target_names = ["positive"]

    model = BanditNRNClassifier(
        target_names=target_names,
        feature_names=feature_names,
        input_size=X_train_val.shape[1],
        output_size=len(target_names) if len(dataset.target_values) > 2 else 1,
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
        policy_init=mic_c_policy,
        logits=False
    )

    epochs = 200
    accumulation_steps = 1
    optimizer = optim.AdamW(
        model.rn.parameters(), lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'] if best_params['use_weight_decay'] else 0)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=best_params['T_0'], T_mult=best_params['T_mult'])
    trainer = BanditNRNTrainer(
        model=model,
        loss_func=nn.BCELoss(),
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
        class_independent=multi_class
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
            evaluation_metric=roc_auc_score,
            multi_class=multi_class,
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
            evaluation_metric=roc_auc_score,
            multi_class=multi_class,
        )
    trainer.set_best_state()

    predictions, targets = trainer.model.predict(test_dl)

    print("TEST AUC", roc_auc_score(targets, predictions, multi_class='ovo', average='micro'))

    # COLLECT EXPLANATIONS
    raw_prediction_decision_threshold = np.median(predictions.values.ravel())
    raw_prediction_seventy_five_percentile = np.quantile(predictions.values.ravel(), 0.6)
    raw_prediction_twenty_five_percentile = np.quantile(predictions.values.ravel(), 0.4)


    simple_sample_explain_75 = trainer.model.explain(
        required_output_thresholds=torch.tensor(raw_prediction_seventy_five_percentile).float(),
        quantile=1.0,
        target_names=['Positive'],
        explain_type='both',
        print_type='logical',
        ignore_uninformative=True,
        rounding_precision=3,
        show_bounds=False if 'use_fbt' not in best_params else not best_params['use_fbt'],
        # show_bounds=True,
        decision_boundary=raw_prediction_decision_threshold,
        simplify=True,
        exclusions=None,
        # min_max_feature_dict=min_max_features_dict
    )

    simple_sample_explain_t_75 = trainer.model.explain(
        required_output_thresholds=torch.tensor(raw_prediction_seventy_five_percentile).float(),
        quantile=0.1,
        target_names=['Positive'],
        explain_type='both',
        print_type='logical',
        ignore_uninformative=True,
        rounding_precision=3,
        show_bounds=False if 'use_fbt' not in best_params else not best_params['use_fbt'],
        # show_bounds=True,
        decision_boundary=raw_prediction_decision_threshold,
        simplify=True,
        exclusions=None,
        # min_max_feature_dict=min_max_features_dict
    )

    simple_sample_explain_25 = trainer.model.explain(
        required_output_thresholds=torch.tensor(raw_prediction_twenty_five_percentile).float(),
        quantile=1.0,
        target_names=['Positive'],
        explain_type='both',
        print_type='logical',
        ignore_uninformative=True,
        rounding_precision=3,
        show_bounds=False if 'use_fbt' not in best_params else not best_params['use_fbt'],
        # show_bounds=True,
        decision_boundary=raw_prediction_decision_threshold,
        simplify=True,
        exclusions=None,
        # min_max_feature_dict=min_max_features_dict
    )

    simple_sample_explain_t_25 = trainer.model.explain(
        required_output_thresholds=torch.tensor(raw_prediction_twenty_five_percentile).float(),
        quantile=0.1,
        target_names=['Positive'],
        explain_type='both',
        print_type='logical',
        ignore_uninformative=True,
        rounding_precision=3,
        show_bounds=False if 'use_fbt' not in best_params else not best_params['use_fbt'],
        # show_bounds=True,
        decision_boundary=raw_prediction_decision_threshold,
        simplify=True,
        exclusions=None,
        # min_max_feature_dict=min_max_features_dict
    )

    results = pd.DataFrame({'simple_sample_explain_pos': [simple_sample_explain_75],
                            'simple_sample_explain_neg': [simple_sample_explain_25],
                            'simple_sample_explain_pos_t': [simple_sample_explain_t_75],
                            'simple_sample_explain_neg_t': [simple_sample_explain_t_25],
                            })

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--openml_id", type=int)
    parser.add_argument("-t", "--train_size", type=float, default=0.6)
    parser.add_argument("-e", "--test_size", type=float, default=0.5)
    parser.add_argument("-s", "--random_state", type=int, default=42)
    parser.add_argument("-r", "--runs", type=int, default=400)
    parser.add_argument("-p", "--study-path", type=str)
    args = parser.parse_args()

    dataset = OpemlMLDataset(
        args.openml_id, args.train_size, args.test_size, args.random_state
    )

    feature_encoder = FeatureEncoder(
        numerical_features=dataset.numerical_features,
        categorical_features=dataset.categorical_features,
        ordinal_features=dataset.ordinal_features,
        label_encode_categorical=False
    )

    dataset.X.to_csv(f"{args.openml_id}_dataset_for_explanation_analysis.csv")

    X_train_val = feature_encoder.fit_transform(dataset.X_train_val)
    X_train_folds = [feature_encoder.transform(df) for df in dataset.X_train_folds]
    X_val_folds = [feature_encoder.transform(df) for df in dataset.X_val_folds]
    X_test = feature_encoder.transform(dataset.X_test)

    y_train_val = dataset.y_train_val
    y_train_folds = dataset.y_train_folds
    y_val_folds = dataset.y_val_folds
    y_test = dataset.y_test

    print(
        f"Train Size: {X_train_folds[0].shape}\n"
        f"Validation Size: {X_val_folds[0].shape}"
        f"\nTest Size{X_test.shape}"
    )

    results = bandit_rrn_noda_main(args, X_train_val, X_train_folds, X_val_folds, X_test,
                                         y_train_val, y_train_folds, y_val_folds, y_test)

    print(results.head())

    out_file = f"BanditRRN_{dataset.dataset.name}_{args.openml_id}_global_explain_examples.csv"

    if os.path.isfile(out_file):
        results = pd.concat([pd.read_csv(out_file), results])

    results.to_csv(out_file, index=False)
