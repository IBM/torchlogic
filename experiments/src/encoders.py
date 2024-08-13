import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder


class FeatureEncoder:

    def __init__(
            self,
            numerical_features: list,
            categorical_features: list,
            ordinal_features: list,
            label_encode_categorical: bool
    ):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.ordinal_features = ordinal_features
        self.label_encode_categorical = label_encode_categorical
        self.mixed_pipe = None

    def _create_pipeline(self, features: pd.DataFrame):
        n_unique_categories = features[self.categorical_features].nunique().sort_values(ascending=False)
        high_cardinality_features = n_unique_categories[n_unique_categories > 255].index
        low_cardinality_features = n_unique_categories[n_unique_categories <= 255].index

        if not self.label_encode_categorical:
            mixed_encoded_preprocessor = ColumnTransformer(
                [
                    ("numerical", "passthrough", self.numerical_features),
                    (
                        "high_cardinality",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        high_cardinality_features,
                    ),
                    (
                        "low_cardinality",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        low_cardinality_features,
                    ),
                    # (
                    #     "ordinal",
                    #     OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    #     self.ordinal_features,
                    # ),
                    (
                        "ordinal_onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        self.ordinal_features,
                    ),
                ],
                verbose_feature_names_out=False,
            )

            mixed_encoded_preprocessor.set_output(transform='pandas')

        else:
            mixed_encoded_preprocessor = ColumnTransformer(
                [
                    ("numerical", "passthrough", self.numerical_features),
                    (
                        "high_cardinality",
                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        high_cardinality_features,
                    ),
                    (
                        "low_cardinality",
                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        low_cardinality_features,
                    ),
                    # (
                    #     "ordinal",
                    #     OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    #     self.ordinal_features,
                    # ),
                    (
                        "ordinal_onehot",
                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        self.ordinal_features,
                    ),
                ],
                verbose_feature_names_out=False,
            )

        self.mixed_pipe = make_pipeline(
            mixed_encoded_preprocessor,
        )

    def fit_transform(self, features: pd.DataFrame):
        self._create_pipeline(features=features)
        out = self.mixed_pipe.fit_transform(features)
        if not isinstance(out, pd.DataFrame):
            out = pd.DataFrame(out)
        return out

    def fit(self, features: pd.DataFrame):
        self._create_pipeline(features=features)
        out = self.mixed_pipe.fit(features)
        if not isinstance(out, pd.DataFrame):
            out = pd.DataFrame(out)
        return out

    def transform(self, features: pd.DataFrame):
        out = self.mixed_pipe.transform(features)
        if not isinstance(out, pd.DataFrame):
            out = pd.DataFrame(out)
        return out
