"""Additional imputers for use in sklearn.pipeline.Pipeline"""

# Authors: Thomas Frank <thomas-frank01@gmx.de>
# License: MIT

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import TransformerMixin


class MostFrequentImputer(TransformerMixin):
    """
    Wrapper for sklearn.impute.SimpleImputer with the "most-frequent" strategy.
    """

    def __init__(self):
        self._imputer = SimpleImputer(strategy='most_frequent')

    def fit(self, X, y=None):
        self._imputer.fit(X, y)
        return self

    def transform(self, X):
        return self._imputer.transform(X)


class DropImputer(TransformerMixin):
    """
    Imputer that drops rows containing np.NaN entries.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if y is None:
            return pd.DataFrame(X).copy().dropna().reset_index(drop=True)
        else:
            X = pd.DataFrame(X).copy()
            original_column_names = X.columns

            # change column names for better usability
            X.columns = range(0, len(X.columns))
            target_column = str(len(X.columns))

            X[target_column] = y

            X = X.dropna().reset_index(drop=True)

            y = X[target_column]
            X = X.drop(columns=[target_column])

            X.columns = original_column_names

            return X, y

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X, y)
