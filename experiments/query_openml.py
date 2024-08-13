import openml


def get_openml_df(max_features, max_classes, min_instances, max_instances):
    openml_df = openml.datasets.list_datasets(output_format="dataframe")
    openml_df = openml_df.query(
        "NumberOfInstancesWithMissingValues == 0 & "
        "NumberOfMissingValues == 0 & "
        "NumberOfClasses > 1 & "
        #'NumberOfClasses <= 30 & '
        "NumberOfSymbolicFeatures == 1 & "
        #'NumberOfInstances > 999 & '
        "NumberOfFeatures >= 2 & "
        "NumberOfNumericFeatures == NumberOfFeatures -1  &"
        "NumberOfClasses <=  " + str(max_classes) + " & "
        "NumberOfFeatures <= " + str(max_features + 1) + " & "
        "NumberOfInstances >= " + str(min_instances) + " & "
        "NumberOfInstances <= " + str(max_instances)
    )

    openml_df = openml_df[
        ["name", "did", "NumberOfClasses", "NumberOfInstances", "NumberOfFeatures"]
    ]

    return openml_df


get_openml_df(100, 20, 160, 1200).to_csv("openml.csv", index=False)
