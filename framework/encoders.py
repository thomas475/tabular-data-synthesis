"""Additional encoders that can be used as transformers in sklearn.Pipeline.pipeline"""

# Authors: Federico Matteucci <federico.matteucci@kit.edu>
#          Thomas Frank <thomas-frank01@gmx.de>
# License: MIT


import functools

import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin, clone

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from sklearn.model_selection import KFold, StratifiedKFold

from category_encoders import CatBoostEncoder, OneHotEncoder, TargetEncoder

import copy

import warnings
import re
from sklearn.utils.random import check_random_state
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM as bgmm


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


class MultiClassWrapper(BaseEstimator, TransformerMixin):
    """Extend supervised encoders to n-class labels, where n >= 2.

    The label can be numerical (e.g.: 0, 1, 2, 3,...,n), string or categorical (pandas.Categorical).
    The label is first encoded into n-1 binary columns. Subsequently, the inner supervised encoder
    is executed for each binarized label.

    The names of the encoded features are suffixed with underscore and the corresponding class name
    (edge scenarios like 'dog'+'cat_frog' vs. 'dog_cat'+'frog' are not currently handled).

    The implementation is experimental and the API may change in the future.
    The order of the returned features may change in the future.

    Original implementation taken from 'category_encoders.wrapper' and modified.


    Parameters
    ----------

    feature_encoder: Object
        an instance of a supervised encoder.

    """

    def __init__(self, feature_encoder):
        self.feature_encoder = feature_encoder
        self.feature_encoders = {}
        self.label_encoder = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        y = pd.DataFrame(y.copy())
        y.columns = ['target']

        # apply one-hot-encoder on the label
        self.label_encoder = OneHotEncoder(handle_missing='error', handle_unknown='error', cols=['target'],
                                           drop_invariant=True, use_cat_names=True)
        labels = self.label_encoder.fit_transform(y)
        labels.columns = [column[7:] for column in labels.columns]
        labels = labels.iloc[:, 1:]  # drop one label

        # train the feature encoders
        for class_name, label in labels.iteritems():
            self.feature_encoders[class_name] = copy.deepcopy(self.feature_encoder).fit(X, label)

    def transform(self, X: pd.DataFrame):
        # initialization
        encoded = None
        feature_encoder = None
        all_new_features = pd.DataFrame()

        # transform the features
        for class_name, feature_encoder in self.feature_encoders.items():
            encoded = feature_encoder.transform(X)

            # decorate the encoded features with the label class suffix
            new_features = encoded[feature_encoder.cols]
            new_features.columns = [str(column) + '_' + class_name for column in new_features.columns]

            all_new_features = pd.concat((all_new_features, new_features), axis=1)

        # add features that were not encoded
        result = pd.concat((encoded[encoded.columns[~encoded.columns.isin(feature_encoder.cols)]], all_new_features), axis=1)

        return result

    def fit_transform(self, X, y=None, **fit_params):
        # When we are training the feature encoders, we have to use fit_transform() method on the features.

        y = pd.DataFrame(y.copy())
        y.columns = ['target']

        # apply one-hot-encoder on the label
        self.label_encoder = OneHotEncoder(handle_missing='error', handle_unknown='error', cols=['target'],
                                           drop_invariant=True, use_cat_names=True)
        labels = self.label_encoder.fit_transform(y)
        labels.columns = [column[7:] for column in labels.columns]
        labels = labels.iloc[:, 1:]  # drop one label

        # initialization of the feature encoders
        encoded = None
        feature_encoder = None
        all_new_features = pd.DataFrame()

        # fit_transform the feature encoders
        for class_name, label in labels.iteritems():
            feature_encoder = copy.deepcopy(self.feature_encoder)
            encoded = feature_encoder.fit_transform(X, label)

            # decorate the encoded features with the label class suffix
            new_features = encoded[feature_encoder.cols]
            new_features.columns = [str(column) + '_' + class_name for column in new_features.columns]

            all_new_features = pd.concat((all_new_features, new_features), axis=1)
            self.feature_encoders[class_name] = feature_encoder

        # add features that were not encoded
        result = pd.concat((encoded[encoded.columns[~encoded.columns.isin(feature_encoder.cols)]], all_new_features), axis=1)

        return result


class CVEncoder(Encoder):
    """
    Encodes every test fold with an Encoder trained on the training fold.
    Default value of base_encoder is supposed to be np.nan to allow the usage of
    default encoder trained on the whole dataset
    """

    def __init__(self, base_encoder, cols=None, n_splits=5, default=-1, **kwargs):
        self.base_encoder = base_encoder
        self.cols = cols
        self.n_splits = n_splits
        self.cv = KFold(
            n_splits=self.n_splits,
            shuffle=True
        )
        self.default_encoder = self.base_encoder(
            cols=self.cols,
            handle_missing='value',
            handle_unknown='value'
            # handle_missing='return_nan',
            # handle_unknown='return_nan'
        )
        self.fold_encoders = [
            self.base_encoder(
                cols=self.cols,
                handle_missing='return_nan',
                handle_unknown='return_nan'
            ) for _ in range(n_splits)
        ]
        self.default = default
        self.splits = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        # Fit a different encoder on each training fold
        for E, (tr, te) in zip(self.fold_encoders, self.cv.split(X, y)):
            self.splits.append((tr, te))
            E.fit(X.iloc[tr], y.iloc[tr])

        # default encoding
        self.default_encoder.fit(X, y)

        return self

    def transform(self, X: pd.DataFrame, **kwargs):
        """
        Test step: the whole dataset is encoded with the base_encoder
        """
        X = X.copy()

        X = self.default_encoder.transform(X)

        # still missing values?
        X.fillna(self.default, inplace=True)

        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        Training step: each fold is encoded differently
        """
        X = X.copy()
        y = y.copy()

        self.fit(X, y, **kwargs)

        # default values
        X_default = self.default_encoder.transform(X)

        for E, (tr, te) in zip(self.fold_encoders, self.splits):
            X.iloc[te] = E.transform(X.iloc[te])

        # default values handling
        default = X.isna()
        X[default] = X_default[default]

        # still missing values?
        X.fillna(self.default, inplace=True)

        return X

    def __str__(self):
        return f'CV{self.n_splits}{self.base_encoder.__name__}'


class StratifiedCVEncoder(Encoder):
    """
    Encodes every test fold with an Encoder trained on the training fold.
    Default value of base_encoder is supposed to be np.nan to allow the usage of
    default encoder trained on the whole dataset
    """

    def __init__(self, base_encoder, cols=None, n_splits=5, default=-1, **kwargs):
        self.base_encoder = base_encoder
        self.cols = cols
        self.n_splits = n_splits
        self.cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True
        )
        self.default_encoder = self.base_encoder(
            cols=self.cols,
            handle_missing='value',
            handle_unknown='value'
            # handle_missing='return_nan',
            # handle_unknown='return_nan'
        )
        self.fold_encoders = [
            self.base_encoder(
                cols=self.cols,
                handle_missing='return_nan',
                handle_unknown='return_nan'
            ) for _ in range(n_splits)
        ]
        self.default = default
        self.splits = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        # Fit a different encoder on each training fold
        for E, (tr, te) in zip(self.fold_encoders, self.cv.split(X, y)):
            self.splits.append((tr, te))
            E.fit(X.iloc[tr], y.iloc[tr])

        # default encoding
        self.default_encoder.fit(X, y)

        return self

    def transform(self, X: pd.DataFrame, **kwargs):
        """
        Test step: the whole dataset is encoded with the base_encoder
        """
        X = X.copy()

        X = self.default_encoder.transform(X)

        # still missing values?
        X.fillna(self.default, inplace=True)

        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        Training step: each fold is encoded differently
        """
        X = X.copy()
        y = y.copy()

        self.fit(X, y, **kwargs)

        # default values
        X_default = self.default_encoder.transform(X)

        for E, (tr, te) in zip(self.fold_encoders, self.splits):
            X.iloc[te] = E.transform(X.iloc[te])

        # default values handling
        default = X.isna()
        X[default] = X_default[default]

        # still missing values?
        X.fillna(self.default, inplace=True)

        return X

    def __str__(self):
        return f'CV{self.n_splits}{self.base_encoder.__name__}'


class StratifiedCVEncoderOriginal(Encoder):
    """
    Encodes every test fold with an Encoder trained on the training fold.
    Default value of base_encoder is supposed to be np.nan to allow the usage of
    default encoder trained on the whole dataset
    """

    def __init__(self, base_encoder, n_splits=5, random_state=1444, default=-1, **kwargs):
        self.base_encoder = base_encoder
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = StratifiedKFold(
            n_splits=self.n_splits,
            random_state=self.random_state,
            shuffle=True
        )
        self.fold_encoders = [clone(self.base_encoder) for _ in range(n_splits)]
        self.default = default

    def fit(self, X: pd.DataFrame, y, **kwargs):
        self.cols = X.columns

        # Fit a different targetEncoder on each training fold
        self.splits = []
        for E, (tr, te) in zip(self.fold_encoders, self.cv.split(X, y)):
            self.splits.append((tr, te))
            E.fit(X.iloc[tr], y.iloc[tr])

        # default encoding
        self.base_encoder.fit(X, y)

        return self

    def fit_transform(self, X: pd.DataFrame, y=None, **kwargs):
        """
        Training step: each fold is encoded differently
        """
        X = X.copy().astype('object')
        self.fit(X, y, **kwargs)

        # default values
        Xdefault = self.base_encoder.transform(X)

        for E, (tr, te) in zip(self.fold_encoders, self.splits):
            X.iloc[te] = E.transform(X.iloc[te])

        # default values handling
        default = X.isna()
        X[default] = Xdefault[default]

        # still missing values?
        X.fillna(self.default, inplace=True)

        return X

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        """
        Test step: the whole dataset is encoded with the base_encoder
        """

        X = X.copy().astype('object')
        X = self.base_encoder.transform(X, y)

        # missing values?
        X.fillna(self.default, inplace=True)

        return X

    def __str__(self):
        return f'OriginalCV{self.n_splits}{self.base_encoder}'


class CV5TargetEncoder(CVEncoder):
    def __init__(self, cols=None, **kwargs):
        super().__init__(TargetEncoder, cols, 5, -1, **kwargs)


class StratifiedCV5TargetEncoder(StratifiedCVEncoder):
    def __init__(self, cols=None, **kwargs):
        super().__init__(TargetEncoder, cols, 5, -1, **kwargs)


class CV5GLMMEncoder(CVEncoder):
    def __init__(self, cols=None, **kwargs):
        super().__init__(GLMMEncoder, cols, 5, -1, **kwargs)


class StratifiedCV5GLMMEncoder(StratifiedCVEncoder):
    def __init__(self, cols=None, **kwargs):
        super().__init__(GLMMEncoder, cols, 5, -1, **kwargs)


class MultiClassCatBoostEncoder(MultiClassWrapper):
    def __init__(self, cols=None):
        super().__init__(CatBoostEncoder(cols=cols))
        self.cols = cols


class MultiClassGLMMEncoder(MultiClassWrapper):
    def __init__(self, cols=None):
        super().__init__(GLMMEncoder(cols=cols))
        self.cols = cols


class MultiClassStratifiedCV5GLMMEncoder(MultiClassWrapper):
    def __init__(self, cols=None):
        super().__init__(StratifiedCV5GLMMEncoder(cols=cols))
        self.cols = cols


class MultiClassTargetEncoder(MultiClassWrapper):
    def __init__(self, cols=None):
        super().__init__(TargetEncoder(cols=cols))
        self.cols = cols


class MultiClassStratifiedCV5TargetEncoder(MultiClassWrapper):
    def __init__(self, cols=None):
        super().__init__(StratifiedCV5TargetEncoder(cols=cols))
        self.cols = cols


class CollapseEncoder(Encoder):
    """
    Evey categorical value is mapped to 1
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        self.cols = X.columns
        return pd.DataFrame(np.ones(len(X)), index=X.index, columns=['cat'])


class DeepOrdinalEncoder:
    def __init__(self, categorical_columns=None, discrete_target=True):
        self._ordinal_encoder = None
        self._label_encoder = None
        self._original_column_titles = None
        self._original_categorical_column_titles = categorical_columns
        self._original_target_column_title = None
        self._discrete_target = discrete_target

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._original_column_titles = X.columns
        self._ordinal_encoder = OrdinalEncoder()
        self._ordinal_encoder.fit(X[self._original_categorical_column_titles])

        self._original_target_column_title = y.name
        if self._discrete_target:
            self._label_encoder = LabelEncoder()
            self._label_encoder.fit(y)

        return self

    def transform(self, X: pd.DataFrame, y: pd.Series):
        X = X.copy()
        y = y.copy()

        X[self._original_categorical_column_titles] = self._ordinal_encoder.transform(X[self._original_categorical_column_titles])
        X.columns = range(0, len(X.columns))

        if self._discrete_target:
            y = pd.Series(self._label_encoder.transform(y))
        y.name = len(X.columns)
        y.index = X.index

        return X, y

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        self.fit(X, y)
        return self.transform(X, y)

    def transform_column_titles(self, column_titles):
        transformed_column_titles = []

        i = 0
        for column_title in self._original_column_titles:
            if column_title in column_titles:
                transformed_column_titles.append(i)
            i = i + 1

        return transformed_column_titles

    def inverse_transform(self, X: pd.DataFrame, y: pd.Series):
        X = X.copy()
        y = y.copy()

        X.columns = self._original_column_titles
        X[self._original_categorical_column_titles] = self._ordinal_encoder.inverse_transform(
            X[self._original_categorical_column_titles]
        )

        if self._discrete_target:
            y = self._label_encoder.inverse_transform(y.astype(dtype=np.int))
            y = pd.Series(y)
        y.name = self._original_target_column_title
        y.index = X.index

        return X, y


"""Generalized linear mixed model"""

__author__ = 'Jan Motl'


class GLMMEncoder(util.BaseEncoder, util.SupervisedTransformerMixin):
    """Generalized linear mixed model.

    Supported targets: binomial and continuous. For polynomial target support, see PolynomialWrapper.

    This is a supervised encoder similar to TargetEncoder or MEstimateEncoder, but there are some advantages:

        1. Solid statistical theory behind the technique. Mixed effects models are a mature branch of statistics.
        2. No hyper-parameters to tune. The amount of shrinkage is automatically determined through the estimation
        process. In short, the less observations a category has and/or the more the outcome varies for a category
        then the higher the regularization towards "the prior" or "grand mean".
        3. The technique is applicable for both continuous and binomial targets. If the target is continuous,
        the encoder returns regularized difference of the observation's category from the global mean.

    If the target is binomial, the encoder returns regularized log odds per category.

    In comparison to JamesSteinEstimator, this encoder utilizes generalized linear mixed models from statsmodels library.

    Note: This is an alpha implementation. The API of the method may change in the future.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop encoded columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    handle_missing: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which returns 0.
    handle_unknown: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which returns 0.
    randomized: bool,
        adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
    sigma: float
        standard deviation (spread or "width") of the normal distribution.
    binomial_target: bool
        if True, the target must be binomial with values {0, 1} and Binomial mixed model is used.
        If False, the target must be continuous and Linear mixed model is used.
        If None (the default), a heuristic is applied to estimate the target type.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target > 22.5
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = GLMMEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 13 columns):
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    CHAS       506 non-null float64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    RAD        506 non-null float64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(13)
    memory usage: 51.5 KB
    None

    References
    ----------

    .. [1] Data Analysis Using Regression and Multilevel/Hierarchical Models, page 253, from
    https://faculty.psau.edu.sa/filedownload/doc-12-pdf-a1997d0d31f84d13c1cdc44ac39a8f2c-original.pdf

    """
    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.ONE_TO_ONE

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True, handle_unknown='value',
                 handle_missing='value', random_state=None, randomized=False, sigma=0.05, binomial_target=None):
        super().__init__(verbose=verbose, cols=cols, drop_invariant=drop_invariant, return_df=return_df,
                         handle_unknown=handle_unknown, handle_missing=handle_missing)
        self.ordinal_encoder = None
        self.mapping = None
        self.random_state = random_state
        self.randomized = randomized
        self.sigma = sigma
        self.binomial_target = binomial_target

    def _fit(self, X, y, **kwargs):
        y = y.astype(float)

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown='value',
            handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)

        # Training
        self.mapping = self._train(X_ordinal, y)

    def _transform(self, X, y=None):
        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        # Loop over the columns and replace the nominal values with the numbers
        X = self._score(X, y)
        return X

    def _more_tags(self):
        tags = super()._more_tags()
        tags["predict_depends_on_y"] = True
        return tags

    def _train(self, X, y):
        # Initialize the output
        mapping = {}

        # Estimate target type, if necessary
        if self.binomial_target is None:
            if len(y.unique()) <= 2:
                binomial_target = True
            else:
                binomial_target = False
        else:
            binomial_target = self.binomial_target

        # The estimation does not have to converge -> at least converge to the same value.
        original_state = np.random.get_state()
        np.random.seed(2001)

        # Reset random state on completion
        try:
            for switch in self.ordinal_encoder.category_mapping:
                col = switch.get('col')
                values = switch.get('mapping')
                data = self._rename_and_merge(X, y, col)

                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        if binomial_target:
                            # Classification, returns (regularized) log odds per category as stored in vc_mean
                            # Note: md.predict() returns: output = fe_mean + vcp_mean + vc_mean[category]
                            md = bgmm.from_formula('target ~ 1', {'a': '0 + C(feature)'}, data).fit_vb()
                            index_names = [int(float(re.sub(r'C\(feature\)\[(\S+)\]', r'\1', index_name))) for index_name in md.model.vc_names]
                            estimate = pd.Series(md.vc_mean, index=index_names)
                        else:
                            # Regression, returns (regularized) mean deviation of the observation's category from the global mean
                            md = smf.mixedlm('target ~ 1', data, groups=data['feature']).fit()
                            tmp = dict()
                            for key, value in md.random_effects.items():
                                tmp[key] = value[0]
                            estimate = pd.Series(tmp)
                except (np.linalg.LinAlgError, ValueError):
                    # Singular matrix -> just return all zeros
                    estimate = pd.Series(np.zeros(len(values)), index=values)

                # Ignore unique columns. This helps to prevent overfitting on id-like columns
                if len(X[col].unique()) == len(y):
                    estimate[:] = 0

                if self.handle_unknown == 'return_nan':
                    estimate.loc[-1] = np.nan
                elif self.handle_unknown == 'value':
                    estimate.loc[-1] = 0

                if self.handle_missing == 'return_nan':
                    estimate.loc[values.loc[np.nan]] = np.nan
                elif self.handle_missing == 'value':
                    estimate.loc[-2] = 0

                mapping[col] = estimate
        finally:
            np.random.set_state(original_state)

        return mapping

    def _score(self, X, y):
        for col in self.cols:
            # Score the column
            X[col] = X[col].map(self.mapping[col])

            # Randomization is meaningful only for training data -> we do it only if y is present
            if self.randomized and y is not None:
                random_state_generator = check_random_state(self.random_state)
                X[col] = (X[col] * random_state_generator.normal(1., self.sigma, X[col].shape[0]))

        return X

    def _rename_and_merge(self, X, y, col):
        """
        Statsmodels requires:
            1) unique column names
            2) non-numeric columns names
        Solution: internally rename the columns.
        """
        merged = pd.DataFrame()
        merged['feature'] = X[col]
        merged['target'] = y

        return merged


# class CollapseEncoder(Encoder):
#     """
#     Every categorical value is mapped to 1.
#     """
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def transform(self, X, y=None, **kwargs):
#         X = pd.DataFrame(X).copy().reset_index(drop=True)
#
#         # selecting categorical columns for processing
#         self.cols = X.infer_objects().select_dtypes(include=['object']).columns
#
#         # drop all categorical columns and add a column instead with the value 1 everywhere
#         X = X.drop(columns=self.cols)
#         X = pd.concat([X, pd.Series(np.ones(len(X)), index=X.index)], axis=1)
#         # X = pd.concat([X, pd.Series(np.ones(len(X)), index=X.index, name='categorical')], axis=1)
#
#         return X



# class TargetEncoder(Encoder):
#     """
#     Maps categorical values into the average target associated to them.
#     """
#
#     def __init__(self, default=-1, **kwargs):
#         super().__init__(default=default, **kwargs)
#
#     def fit(self, X: pd.DataFrame, y, **kwargs):
#         X = X.copy()
#
#         # convert column names to a more practical form
#         column_titles = range(0, len(X.columns))
#         X.columns = column_titles
#
#         # adding target vector
#         target_name = len(X.columns)
#         X[target_name] = y
#
#         # selecting categorical columns for processing
#         self.cols = list(X.select_dtypes(['object']).columns)
#
#         for col in self.cols:
#             temp = X.groupby(col)[target_name].mean().to_dict()
#             self.encoding[col].update(temp)
#
#         return self
#
#     def transform(self, X: pd.DataFrame, y=None, **kwargs):
#         X = X.copy()
#
#         # convert column names to a more practical form
#         original_column_titles = X.columns
#         X.columns = range(0, len(X.columns))
#
#         for col in self.cols:
#             X[col] = X[col].apply(lambda cat: self.encoding[col][cat])
#
#         # reverting to original column names
#         X.columns = original_column_titles
#
#         return X
