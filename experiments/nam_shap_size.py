import argparse

import pandas as pd

from src.datasets import OpemlMLDataset

BENCHMARK_DATASETS = [44120, 44121, 44122, 44123, 44124, 44125, 44126, 44127, 44128, 44129, 44130, 44131,
                      44089, 44090, 44091, 44156, 44157, 44158, 44159, 44160, 44161, 44162]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_size", type=float, default=0.6)
    parser.add_argument("-e", "--test_size", type=float, default=0.5)
    parser.add_argument("-s", "--random_state", type=int, default=42)
    args = parser.parse_args()

    openml_ids = []
    explanation_sizes = []
    for openml_id in BENCHMARK_DATASETS:
        dataset = OpemlMLDataset(
            openml_id, args.train_size, args.test_size, args.random_state
        )
        # explanation_size = dataset.X_test.nunique().sum() + dataset.X_test.shape[1]
        explanation_size = dataset.X_test.shape[1]
        openml_ids += [openml_id]
        explanation_sizes += [explanation_size]

    result = pd.DataFrame({'openml_id': openml_ids, 'explanation_size': explanation_sizes})
    result.to_csv('./plots/nam_shap_explanation_sizes.csv', index=False)

    # import pandas as pd
    # import numpy as np
    # import os
    # import glob
    # def calculate_rnrn_explanation_sizes(data_path):
    #     files = glob.glob(os.path.join(data_path, '*auc_fix_explanations_with_percentiles_rnrn_size.csv'))
    #     sizes = []
    #     for file in files:
    #         df = pd.read_csv(file)
    #         df['explanation'] = df['simple_sample_explain_pos'].combine_first(df['simple_sample_explain_neg'])
    #         df['explanation_size'] = df['explanation'].apply(lambda x: x.count('\n\t') if x != 'FAILED' else np.nan)
    #         failed_instances = df['explanation'].str.contains('FAILED').sum()
    #         if failed_instances > 0:
    #             print(f"{failed_instances} FAILED INSTANCES for {file}!!")
    #         sizes += [df['explanation_size'].mean()]
    #     print(np.mean(sizes))
    #     print(pd.DataFrame({'file': files, 'size': sizes}))
    #
    #
    # calculate_rnrn_explanation_sizes('./plots')
