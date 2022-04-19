"""Transformer wrappers for SMOTE, GAN and Gibbs sampling methods for use as
   samplers in sklearn.pipeline.Pipeline"""

# Authors: Thomas Frank <thomas-frank01@gmx.de>
# License: MIT

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class Labeler(TransformerMixin):
    """
    Transformer that labels the dataset by using the submitted already trained
    teacher model.
    """

    def __init__(self, trained_model, ignored_first_n_samples=0):
        self._trained_model = trained_model
        self._ignored_first_n_samples = ignored_first_n_samples

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        if self._ignored_first_n_samples >= len(X):
            return X, y
        else:
            complete_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
            complete_target = pd.Series(y).copy().reset_index(drop=True)

            resampled_dataset = complete_dataset.iloc[self._ignored_first_n_samples:, :]
            original_target = complete_target.iloc[:self._ignored_first_n_samples]

            resampled_target = pd.Series(
                self._trained_model.predict(resampled_dataset)
            ).copy().reset_index(drop=True)

            complete_target = pd.concat([original_target, resampled_target], ignore_index=True)
            complete_target.reset_index(drop=True)

            return complete_dataset, complete_target

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X, y)


class TargetInjector(TransformerMixin):
    """
    Append the target column to the dataset.
    """

    def __init__(self):
        return

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        return pd.concat(
            [
                pd.DataFrame(X).copy().reset_index(drop=True),
                pd.Series(y).copy().reset_index(drop=True)
            ],
            axis=1,
            sort=False
        )

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X, y)


class TargetExtractor(TransformerMixin):
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


class NumericalTargetDiscretizer(TransformerMixin):
    """
    Discretize potentially continuous numerical targets according to the
    discrete targets submitted in the constructor. Each continuous target entry
    is mapped to its closest discrete target value. If a continuous entry is
    exactly between two discrete values it is mapped to the bigger one.
    """

    def __init__(self, y):
        # get unique targets in descending order
        self._discrete_targets = np.flip(np.unique(y), axis=0)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y):
        discretized_y = []
        for entry in y:
            index = np.abs(self._discrete_targets - entry).argmin()
            discretized_y.append(self._discrete_targets[index])

        return X, discretized_y

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X, y)


class DatasetCombiner(TransformerMixin):
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
