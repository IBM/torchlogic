import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class IRISDataset:
    def __init__(self, openml_id, test_size, val_size, random_state):
        mms = MinMaxScaler()
        encoder = LabelEncoder()

        uc = pd.HDFStore("iris.transformed.10class.h5")
        df = uc.select_as_multiple(
            ['df{}'.format(i) for i in range(len(uc.keys()))],
            selector='df0')
        uc.close()

        df = df.drop(["SNAPSHOT_DATE"], axis=1)
        df = df.drop(["CUST_ID"], axis=1)
        df = df.drop(["COUNTRY_CODE"], axis=1)

        target_features = [x for x in list(df.columns) if x.endswith("_label")]
        target_values = target_features
        X = df.drop(target_features, axis=1)
        feature_names=X.columns
        y = df[target_features]

        train_data, test_data = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
        )

        train_data, val_data = train_test_split(
            df,
            test_size=val_size,
            random_state=random_state,
        )

        X_train = train_data.drop(target_features, axis=1)
        X_val = val_data.drop(target_features, axis=1)
        X_test = test_data.drop(target_features, axis=1)

        y_train = train_data[target_features].values
        y_val = val_data[target_features].values
        y_test = test_data[target_features].values

        self.numerical_features = X.columns
        self.categorical_features = []
        self.ordinal_features = []
        
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.target_values = target_values
        self.df = df
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test


__all__ = ["IRISDataset"]
