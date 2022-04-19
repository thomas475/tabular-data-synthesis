"""Additional encoders that can be used as transformers in sklearn.Pipeline.pipeline"""

# Authors: Federico Matteucci <federico.matteucci@kit.edu>
#          Thomas Frank <thomas-frank01@gmx.de>
# License: MIT


import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, default=-1, **kwargs):
        self.default = default
        self.encoding = defaultdict(lambda: defaultdict(lambda: self.default))
        self.inverse_encoding = defaultdict(lambda: defaultdict(lambda: self.default))
        self.cols = None

    def fit(self, X: pd.DataFrame, y=None, **kwargs):
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        X = X.copy()
        return X

    def fit_transform(self, X: pd.DataFrame, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X, y, **kwargs)


class TargetEncoder(Encoder):
    """
    Maps categorical values into the average target associated to them.
    """

    def __init__(self, default=-1, **kwargs):
        super().__init__(default=default, **kwargs)

    def fit(self, X: pd.DataFrame, y, **kwargs):
        X = X.copy()

        # convert column names to a more practical form
        column_titles = range(0, len(X.columns))
        X.columns = column_titles

        # adding target vector
        target_name = len(X.columns)
        X[target_name] = y

        # selecting categorical columns for processing
        self.cols = list(X.select_dtypes(['object']).columns)

        for col in self.cols:
            temp = X.groupby(col)[target_name].mean().to_dict()
            self.encoding[col].update(temp)

        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        X = X.copy()

        # convert column names to a more practical form
        original_column_titles = X.columns
        X.columns = range(0, len(X.columns))

        for col in self.cols:
            X[col] = X[col].apply(lambda cat: self.encoding[col][cat])

        # reverting to original column names
        X.columns = original_column_titles

        return X


class CollapseEncoder(Encoder):
    """
    Every categorical value is mapped to 1.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, X, y=None, **kwargs):
        X = pd.DataFrame(X).copy().reset_index(drop=True)

        # selecting categorical columns for processing
        self.cols = X.infer_objects().select_dtypes(include=['object']).columns

        # drop all categorical columns and add a column instead with the value 1 everywhere
        X = X.drop(columns=self.cols)
        X = pd.concat([X, pd.Series(np.ones(len(X)), index=X.index)], axis=1)
        # X = pd.concat([X, pd.Series(np.ones(len(X)), index=X.index, name='categorical')], axis=1)

        print(X)

        return X
