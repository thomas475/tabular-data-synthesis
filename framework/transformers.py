"""Transformer wrappers for SMOTE, GAN and Gibbs sampling methods for use as
   samplers in sklearn.pipeline.Pipeline"""

# Authors: Thomas Frank <thomas-frank01@gmx.de>
# License: MIT
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Labeler(BaseEstimator, TransformerMixin):
    """
    Transformer that labels the dataset by using the submitted already trained
    teacher model.
    """

    def __init__(self, trained_model):
        self._trained_model = trained_model

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X, self._trained_model.predict(X)


class TargetInjector(BaseEstimator, TransformerMixin):
    """
    Append the target column to the dataset.
    """

    def __init__(self):
        return

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        return pd.concat([pd.DataFrame(X).copy(), pd.Series(y).copy()], axis=1, sort=False)

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X, y)


class TargetExtractor(BaseEstimator, TransformerMixin):
    """
    Extract the target column from the end of the dataset.
    """

    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()

        original_column_names = X.columns[:len(X.columns) - 1]
        original_target_column = X.columns[len(X.columns) - 1]
        X.columns = range(0, len(X.columns))
        target_column = len(X.columns) - 1

        y = X[target_column]
        X = X.drop(columns=[target_column])

        X.columns = original_column_names
        y.name = original_target_column

        return X, y


class DatasetCombiner(BaseEstimator, TransformerMixin):
    """
    Transformer that combines the submitted dataset with the current dataset in
    the pipeline.
    """

    def __init__(self, X, y=None):
        self._X = pd.DataFrame(X).copy()
        self._y = pd.Series(y).copy()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        X.columns = self._X.columns

        X_combined = pd.concat([self._X, X], axis=0, sort=False)
        X_combined = X_combined.reset_index(drop=True)

        if self._y is None or y is None:
            return X_combined
        else:
            y = pd.Series(y).copy()
            y.name = self._y.name

            y_combined = pd.concat([self._y, y], axis=0, sort=False)
            y_combined = y_combined.reset_index(drop=True)

            return X_combined, y_combined

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X, y)
