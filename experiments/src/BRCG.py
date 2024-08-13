import pandas as pd
from aix360.algorithms.rbm import FeatureBinarizerFromTrees
from aix360.algorithms.rbm.boolean_rule_cg import BooleanRuleCG


class BRCG(object):

    def __init__(
            self,
            lambda0,
            lambda1,
            CNF,
            K,
            D,
            B,
            iterMax,
            timeMax,
            eps,
            solver,
            fbt=None
    ):
        self.fbt = fbt
        self.model = BooleanRuleCG(
            lambda0=lambda0,
            lambda1=lambda1,
            CNF=CNF,
            K=K,
            D=D,
            B=B,
            iterMax=iterMax,
            timeMax=timeMax,
            eps=eps,
            solver=solver
        )

    def fit(self, X_train, y_train):
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
        self.model.fit(X_train, y_train.ravel())

    def predict(self, X):
        return self.model.predict(X)
