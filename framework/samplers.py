"""Transformer wrappers for SMOTE, GAN and Gibbs sampling methods for use as
   samplers in sklearn.pipeline.Pipeline"""

# Authors: Thomas Frank <thomas-frank01@gmx.de>
# License: MIT

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from imblearn.over_sampling import SMOTE
from ydata_synthetic.synthesizers.regular import CGAN
from ydata_synthetic.synthesizers.regular import DRAGAN
from ydata_synthetic.synthesizers.regular import VanilllaGAN
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

from framework.racog import RACOG

import warnings
import tensorflow as tf
import random


class ProportionalSampler(BaseEstimator, TransformerMixin):
    """
    Proportional sampler base class. Requires the target value in each step.
    """

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        return self

    def fit_transform(self, X, y, **fit_params):
        return self.fit(X, y, **fit_params).transform(X, y)


class UnlabeledSampler(BaseEstimator, TransformerMixin):
    """
    Unlabeled sampler base class. Doesn't require the target.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X, y)


class ProportionalSMOTESampler(ProportionalSampler):
    """
    Transformer that implements a proportional SMOTE sampling routine.
    SMOTE is usually used for oversampling in imbalanced datasets, but this
    implementation allows us to sample all classes proportionally and their
    ratio remains intact.

    Parameters
    ----------

    sample_multiplication_factor : float
        Determines the relative amount of generated data, i.e. 0 means that no
        data is generated and 1 means that we generate the same number of data
        points as we already have.

    only_sampled : bool, default=False
        Determines whether the original dataset is prepended to the generated
        samples.

    random_state : int, default=None
        Control the randomization of the algorithm.

    k_neighbors : int or object, default=5
        If int, number of nearest neighbours to be used to construct synthetic
        samples. If object, an estimator that inherits from KNeighborsMixin that
        will be used to find the k_neighbors.

    n_jobs : int, default=None
        Number of CPU cores used during the cross-validation loop. None means 1
        unless in a joblib.parallel_backend context. -1 means using all
        processors. See Glossary for more details.

    References
    ----------

    .. [1] https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html

    .. [2] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, “SMOTE:
           synthetic minority over-sampling technique,” Journal of artificial
           intelligence research, 321-357, 2002.

    """

    def __init__(
            self,
            sample_multiplication_factor,
            only_sampled=False,
            random_state=None,
            k_neighbors=5,
            n_jobs=None
    ):
        self.sample_multiplication_factor = sample_multiplication_factor
        self.only_sampled = only_sampled
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        """
        Count number of occurrences of each class and resample proportionally.
        """
        # return an empty dataframe and target or the original samples if the multiplication factor is too small
        if int(self.sample_multiplication_factor * len(X)) < 1:
            if self.only_sampled:
                return pd.DataFrame(columns=pd.DataFrame(X).columns), pd.Series([])
            else:
                return X, y

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns
        if hasattr(y, 'name'):
            original_target_title = y.name

        original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
        original_target = pd.Series(y).copy().reset_index(drop=True)

        # calculate the number of occurrences per class
        unique, counts = np.unique(original_target, return_counts=True)
        occurrences_per_class_dict = dict(zip(unique, counts))

        # set the amount of samples per class we want to have; includes original samples
        sampling_strategy = {}
        for class_name in occurrences_per_class_dict:
            sampling_strategy[class_name] = int(
                (self.sample_multiplication_factor + 1) * occurrences_per_class_dict[class_name]
            )

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            k_neighbors=self.k_neighbors,
            n_jobs=self.n_jobs
        )

        # filter user warning that fires because we do not use SMOTE simply for balancing the dataset
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            resampled_dataset, resampled_target = smote.fit_resample(original_dataset, original_target)
        resampled_dataset = pd.DataFrame(resampled_dataset)
        resampled_target = pd.Series(resampled_target)

        if self.only_sampled:
            # remove original dataset
            resampled_dataset = pd.DataFrame(resampled_dataset.iloc[len(X):, :])
            resampled_dataset = resampled_dataset.reset_index(drop=True)
            resampled_target = pd.Series(resampled_target.iloc[len(X):])
            resampled_target = resampled_target.reset_index(drop=True)

        # restore column titles if available
        if hasattr(X, 'columns'):
            resampled_dataset.columns = original_column_titles
        if hasattr(y, 'name'):
            resampled_target.name = original_target_title

        return resampled_dataset, resampled_target


class UnlabeledSMOTESampler(UnlabeledSampler):
    """
    Transformer that implements a SMOTE sampling routine for unlabeled data.
    SMOTE is usually used for oversampling in imbalanced datasets, but this
    implementation allows us to apply SMOTE on data without classes.

    Parameters
    ----------

    sample_multiplication_factor : float
        Determines the relative amount of generated data, i.e. 0 means that no
        data is generated and 1 means that we generate the same number of data
        points as we already have.

    only_sampled : bool, default=False
        Determines whether the original dataset is prepended to the generated
        samples.

    random_state : int, default=None
        Control the randomization of the algorithm.

    k_neighbors : int or object, default=5
        If int, number of nearest neighbours to be used to construct synthetic
        samples. If object, an estimator that inherits from KNeighborsMixin that
        will be used to find the k_neighbors.

    n_jobs : int, default=None
        Number of CPU cores used during the cross-validation loop. None means 1
        unless in a joblib.parallel_backend context. -1 means using all
        processors. See Glossary for more details.

    References
    ----------

    .. [1] https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html

    .. [2] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, “SMOTE:
           synthetic minority over-sampling technique,” Journal of artificial
           intelligence research, 321-357, 2002.

    """

    def __init__(
            self,
            sample_multiplication_factor,
            only_sampled=False,
            random_state=None,
            k_neighbors=5,
            n_jobs=None
    ):
        self.sample_multiplication_factor = sample_multiplication_factor
        self.only_sampled = only_sampled
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        No labels available, so we view the entire set as belonging to one class
        and use SMOTE on the entire dataset. Because SMOTE requires there to be
        at least two classes, we add a dummy label vector and one dummy data
        entry. After we run SMOTE, we remove the dummy entry.
        """
        # return an empty dataframe and target or the original samples if the multiplication factor is too small
        if int(self.sample_multiplication_factor * len(X)) < 1:
            if self.only_sampled:
                return pd.DataFrame(columns=pd.DataFrame(X).columns), pd.Series([])
            else:
                return X, y

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns

        original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)

        # create a dummy target vector [ 0 1 ... 1 ]
        y_dummy = np.append(0, np.full((len(original_dataset), 1), 1))

        # set sampling strategy to ignore the dummy class and resample the rest
        sampling_strategy = {
            0: 1,
            1: int((self.sample_multiplication_factor + 1) * len(original_dataset))
        }

        # insert a dummy entry on index 0
        dummy_dataset = pd.DataFrame(original_dataset).copy()
        dummy_dataset.loc[-1] = [0] * len(dummy_dataset.columns)
        dummy_dataset.index = dummy_dataset.index + 1
        dummy_dataset.sort_index(inplace=True)

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            k_neighbors=self.k_neighbors,
            n_jobs=self.n_jobs
        )

        # filter user warning that fires because we do not use SMOTE simply for balancing the dataset
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            resampled_dummy_dataset, _ = smote.fit_resample(dummy_dataset, y_dummy)
        resampled_dummy_dataset = pd.DataFrame(resampled_dummy_dataset)

        # remove the dummy entry
        resampled_dataset = resampled_dummy_dataset.iloc[1:, :].reset_index(drop=True)

        if self.only_sampled:
            # remove original dataset
            resampled_dataset = pd.DataFrame(resampled_dataset.iloc[len(X):, :])
            resampled_dataset = resampled_dataset.reset_index(drop=True)
            resampled_target = None
        else:
            resampled_target = pd.concat([
                pd.Series(y).copy().reset_index(drop=True),
                pd.Series(
                    np.full(int(self.sample_multiplication_factor * len(X)), np.nan)
                ).copy().reset_index(drop=True)
            ], ignore_index=True)
            resampled_target = pd.Series(resampled_target).reset_index(drop=True)

        # restore column titles if available
        if hasattr(X, 'columns'):
            resampled_dataset.columns = original_column_titles

        return resampled_dataset, resampled_target


class ProportionalRACOGSampler(ProportionalSampler):
    """
    Transformer that implements a proportional RACOG sampling routine. RACOG is
    a Gibbs sampling routine used for oversampling in imbalanced datasets, but
    this implementation allows us to sample all classes proportionally and
    their ratio remains intact.

    Parameters
    ----------

    sample_multiplication_factor : float
        Determines the relative amount of generated data, i.e. 0 means that no
        data is generated and 1 means that we generate the same number of data
        points as we already have.

    only_sampled : bool, default=False
        Determines whether the original dataset is prepended to the generated
        samples.

    burnin : int, default=100
        It determines how many examples generated for a given one are going to
        be discarded firstly.

    lag : int, default=20
        Number of iterations between new generated example for a minority one.

    discretization : 'caim' or 'mdlp', default='caim'
        Method for discretization of continuous variables.

    continuous_distribution : 'normal' or 'laplace', default='normal'
        The distribution used for sampling (reconstruct) continuous variables
        after oversampling.

    n_jobs : int, default=1
        The number of jobs to run in parallel for sampling.

    verbose : int, default=0
        If greater than 0, enable verbose output.

    random_state : int, RandomState instance or None, default=None
        If int, 'random_state' is the seed used by the random number
        generator; If 'RandomState' instance, random_state is the random
        number generator; If 'None', the random number generator is the
        'RandomState' instance used by 'np.random'.

    References
    ----------

    .. [1] B. Das, N. C. Krishnan and D. J. Cook, "RACOG and wRACOG: Two
           Probabilistic Oversampling Techniques," in IEEE Transactions on
           Knowledge and Data Engineering, vol. 27, no. 1, pp. 222-234,
           1 Jan. 2015, doi: 10.1109/TKDE.2014.2324567.

    """

    def __init__(
            self,
            sample_multiplication_factor,
            only_sampled=False,
            burnin=100,
            lag=20,
            discretization='caim',
            continuous_distribution='normal',
            n_jobs=1,
            verbose=0,
            random_state=None
    ):
        self.sample_multiplication_factor = sample_multiplication_factor
        self.only_sampled = only_sampled
        self.burnin = burnin
        self.lag = lag
        self.discretization = discretization
        self.continuous_distribution = continuous_distribution
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        """
        Count number of occurrences of each class and resample proportionally.
        """
        # return an empty dataframe and target or the original samples if the multiplication factor is too small
        if int(self.sample_multiplication_factor * len(X)) < 1:
            if self.only_sampled:
                return pd.DataFrame(columns=pd.DataFrame(X).columns), pd.Series([])
            else:
                return X, y

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns
        if hasattr(y, 'name'):
            original_target_title = y.name

        original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
        original_target = pd.Series(y).copy().reset_index(drop=True)

        # calculate the number of occurrences per class
        unique, counts = np.unique(original_target, return_counts=True)
        occurrences_per_class_dict = dict(zip(unique, counts))

        # set the amount of samples per class we want to have; includes original samples
        sampling_strategy = {}
        for class_name in occurrences_per_class_dict:
            sampling_strategy[class_name] = int(
                (self.sample_multiplication_factor + 1) * occurrences_per_class_dict[class_name]
            )

        racog = RACOG(
            sampling_strategy=sampling_strategy,
            burnin=self.burnin,
            lag=self.lag,
            discretization=self.discretization,
            continuous_distribution=self.continuous_distribution,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
            categorical_features='all',
            only_sampled=self.only_sampled
        )

        # filter user warning that fires because we do not use RACOG simply for balancing the dataset
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            resampled_dataset, resampled_target = racog.fit_resample(original_dataset, original_target)
        resampled_dataset = pd.DataFrame(resampled_dataset)
        resampled_target = pd.Series(resampled_target)

        # restore column titles if available
        if hasattr(X, 'columns'):
            resampled_dataset.columns = original_column_titles
        if hasattr(y, 'name'):
            resampled_target.name = original_target_title

        return resampled_dataset, resampled_target


class UnlabeledRACOGSampler(UnlabeledSampler):
    """
    Transformer that implements a RACOG sampling routine for unlabeled data.
    RACOG is a Gibbs sampling routine used for oversampling in imbalanced
    datasets, but this implementation allows us to apply RACOG on data without
    classes.

    Parameters
    ----------

    sample_multiplication_factor : float
        Determines the relative amount of generated data, i.e. 0 means that no
        data is generated and 1 means that we generate the same number of data
        points as we already have.

    only_sampled : bool, default=False
        Determines whether the original dataset is prepended to the generated
        samples.

    burnin : int, default=100
        It determines how many examples generated for a given one are going to
        be discarded firstly.

    lag : int, default=20
        Number of iterations between new generated example for a minority one.

    discretization : 'caim' or 'mdlp', default='caim'
        Method for discretization of continuous variables.

    continuous_distribution : 'normal' or 'laplace', default='normal'
        The distribution used for sampling (reconstruct) continuous variables
        after oversampling.

    n_jobs : int, default=1
        The number of jobs to run in parallel for sampling.

    verbose : int, default=0
        If greater than 0, enable verbose output.

    random_state : int, RandomState instance or None, default=None
        If int, 'random_state' is the seed used by the random number
        generator; If 'RandomState' instance, random_state is the random
        number generator; If 'None', the random number generator is the
        'RandomState' instance used by 'np.random'.

    References
    ----------

    .. [1] B. Das, N. C. Krishnan and D. J. Cook, "RACOG and wRACOG: Two
           Probabilistic Oversampling Techniques," in IEEE Transactions on
           Knowledge and Data Engineering, vol. 27, no. 1, pp. 222-234,
           1 Jan. 2015, doi: 10.1109/TKDE.2014.2324567.

    """

    def __init__(
            self,
            sample_multiplication_factor,
            only_sampled=False,
            burnin=100,
            lag=20,
            discretization='caim',
            continuous_distribution='normal',
            n_jobs=1,
            verbose=0,
            random_state=None
    ):
        self.sample_multiplication_factor = sample_multiplication_factor
        self.only_sampled = only_sampled
        self.burnin = burnin
        self.lag = lag
        self.discretization = discretization
        self.continuous_distribution = continuous_distribution
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        No labels available, so we view the entire set as belonging to one class
        and use RACOG on the entire dataset. Because RACOG requires there to be
        at least two classes, we add a dummy label vector and one dummy data
        entry that is ignored in the resampling process.
        """
        # return an empty dataframe and target or the original samples if the multiplication factor is too small
        if int(self.sample_multiplication_factor * len(X)) < 1:
            if self.only_sampled:
                return pd.DataFrame(columns=pd.DataFrame(X).columns), pd.Series([])
            else:
                return X, y

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns

        original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)

        # create a dummy target vector [ 0 1 ... 1 ]
        y_dummy = np.append(0, np.full((len(original_dataset), 1), 1))

        # set sampling strategy to ignore the dummy class and resample the rest
        sampling_strategy = {
            0: 1,
            1: int((self.sample_multiplication_factor + 1) * len(original_dataset))
        }

        # insert a dummy entry on index 0
        dummy_dataset = pd.DataFrame(original_dataset).copy()
        dummy_dataset.loc[-1] = [0] * len(dummy_dataset.columns)
        dummy_dataset.index = dummy_dataset.index + 1
        dummy_dataset.sort_index(inplace=True)

        racog = RACOG(
            sampling_strategy=sampling_strategy,
            burnin=self.burnin,
            lag=self.lag,
            discretization=self.discretization,
            continuous_distribution=self.continuous_distribution,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
            categorical_features='all',
            only_sampled=self.only_sampled
        )

        # filter user warning that fires because we do not use RACOG simply for balancing the dataset
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            resampled_dataset, _ = racog.fit_resample(dummy_dataset, y_dummy)
        resampled_dataset = pd.DataFrame(resampled_dataset)

        # remove the dummy entry
        resampled_dataset = resampled_dataset.iloc[1:, :].reset_index(drop=True)

        if self.only_sampled:
            resampled_target = None
        else:
            resampled_target = pd.concat([
                pd.Series(y).copy().reset_index(drop=True),
                pd.Series(
                    np.full(int(self.sample_multiplication_factor * len(X)), np.nan)
                ).copy().reset_index(drop=True)
            ], ignore_index=True)
            resampled_target = pd.Series(resampled_target).reset_index(drop=True)

        # restore column titles if available
        if hasattr(X, 'columns'):
            resampled_dataset.columns = original_column_titles

        return resampled_dataset, resampled_target


class ProportionalVanillaGANSampler(ProportionalSampler):
    """
    Transformer that implements a proportional sampling routine using a vanilla
    GAN implementation. We train and sample from a different vanilla GAN model
    for each class.

    Parameters
    ----------

    sample_multiplication_factor : float
        Determines the relative amount of generated data, i.e. 0 means that no
        data is generated and 1 means that we generate the same number of data
        points as we already have.

    only_sampled : bool, default=False
        Determines whether the original dataset is prepended to the generated
        samples.

    batch_size: int, default=128
        Number of samples used per training step.

    learning_rate: float, default=1e-4
        The learning rate for each training step.

    betas: tuple, default=(0.5, 0.9)
        Initial decay rates of Adam when estimating the first and second
        moments of the gradient.

    noise_dim: int, default=264
        The length of the noise vector per example.

    layers_dim: int, default=128
        The dimension of the networks layers.

    epochs: int, default=300
        Total number of training steps.

    sample_interval: int, default=50
        The interval between samples.

    random_state : int, default=None
        Control the randomization of the algorithm.

    References
    ----------

    .. [1] https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/regular/vanillagan/model.py

    """

    def __init__(
            self,
            sample_multiplication_factor,
            only_sampled=False,
            batch_size=128,
            learning_rate=1e-4,
            betas=(0.5, 0.9),
            noise_dim=264,
            layers_dim=128,
            epochs=300,
            sample_interval=50,
            random_state=None
    ):
        self.sample_multiplication_factor = sample_multiplication_factor
        self.only_sampled = only_sampled
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.betas = betas
        self.noise_dim = noise_dim
        self.layers_dim = layers_dim
        self.epochs = epochs
        self.sample_interval = sample_interval
        self.random_state = random_state

    def fit(self, X, y):
        _set_global_random_state(self.random_state)

        # reset vanilla gan models
        self._vanilla_gan = {}

        original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
        y = pd.Series(y).copy().reset_index(drop=True)

        # change the column titles for easier use
        original_dataset.columns = _convert_list_to_string_list(range(0, len(original_dataset.columns)))
        num_cols = original_dataset.columns.copy().tolist()

        # calculate the number of occurrences per class
        unique, counts = np.unique(y, return_counts=True)
        occurrences_per_class_dict = dict(zip(unique, counts))

        for class_name in occurrences_per_class_dict:
            self._vanilla_gan[class_name] = VanilllaGAN(
                model_parameters=ModelParameters(
                    batch_size=self.batch_size,
                    lr=self.learning_rate,
                    betas=self.betas,
                    noise_dim=self.noise_dim,
                    layers_dim=self.layers_dim
                )
            )

            original_subset = original_dataset.iloc[[x for x in range(0, len(y)) if y[x] == class_name]]

            self._vanilla_gan[class_name].train(
                data=original_subset,
                train_arguments=TrainParameters(
                    epochs=self.epochs,
                    sample_interval=self.sample_interval
                ),
                num_cols=num_cols,
                cat_cols=[]
            )

        return self

    def transform(self, X, y):
        """
        Sample proportionally from each of the vanilla GANs trained on the
        subsets split by class.
        """
        _set_global_random_state(self.random_state)

        # return an empty dataframe and target or the original samples if the multiplication factor is too small
        if int(self.sample_multiplication_factor * len(X)) < 1:
            if self.only_sampled:
                return pd.DataFrame(columns=pd.DataFrame(X).columns), pd.Series([])
            else:
                return X, y

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns
        if hasattr(y, 'name'):
            original_target_title = y.name

        original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
        original_target = pd.Series(y).copy().reset_index(drop=True)

        # calculate the number of occurrences per class
        unique, counts = np.unique(original_target, return_counts=True)
        occurrences_per_class_dict = dict(zip(unique, counts))

        resampled_subsets_per_class = []
        resampled_targets_per_class = []
        for class_name in occurrences_per_class_dict:
            number_of_samples = int(self.sample_multiplication_factor * occurrences_per_class_dict[class_name])

            resampled_subset = self._vanilla_gan[class_name].sample(
                n_samples=number_of_samples,
            )
            resampled_target = np.full((number_of_samples,), class_name).T

            resampled_subset = pd.DataFrame(resampled_subset)
            resampled_target = pd.Series(resampled_target)

            # remove excess generated samples
            resampled_subset = resampled_subset.iloc[:number_of_samples, :]

            resampled_subsets_per_class.append(resampled_subset)
            resampled_targets_per_class.append(resampled_target)

        # join resampled subsets and targets
        resampled_dataset = pd.concat(resampled_subsets_per_class, ignore_index=True)
        resampled_dataset = resampled_dataset.reset_index(drop=True)
        resampled_target = pd.concat(resampled_targets_per_class, ignore_index=True)
        resampled_target = resampled_target.reset_index(drop=True)

        if not self.only_sampled:
            # add original samples and target
            original_dataset.columns = range(len(original_dataset.columns))
            original_target.name = len(original_dataset.columns)
            resampled_dataset.columns = range(len(resampled_dataset.columns))
            resampled_target.name = len(resampled_dataset.columns)

            resampled_dataset = pd.concat([original_dataset, resampled_dataset], ignore_index=True)
            resampled_dataset.reset_index(drop=True)
            resampled_target = pd.concat([original_target, resampled_target], ignore_index=True)
            resampled_target.reset_index(drop=True)

        # restore column titles if available
        if hasattr(X, 'columns'):
            resampled_dataset.columns = original_column_titles
        if hasattr(y, 'name'):
            resampled_target.name = original_target_title

        return resampled_dataset, resampled_target


class UnlabeledVanillaGANSampler(UnlabeledSampler):
    """
    Transformer that implements a sampling routine for a trained vanilla GAN
    model on unlabeled data.

    Parameters
    ----------

    sample_multiplication_factor : float
        Determines the relative amount of generated data, i.e. 0 means that no
        data is generated and 1 means that we generate the same number of data
        points as we already have.

    only_sampled : bool, default=False
        Determines whether the original dataset is prepended to the generated
        samples.

    batch_size: int, default=128
        Number of samples used per training step.

    learning_rate: float, default=1e-4
        The learning rate for each training step.

    betas: tuple, default=(0.5, 0.9)
        Initial decay rates of Adam when estimating the first and second
        moments of the gradient.

    noise_dim: int, default=264
        The length of the noise vector per example.

    layers_dim: int, default=128
        The dimension of the networks layers.

    epochs: int, default=300
        Total number of training steps.

    sample_interval: int, default=50
        The interval between samples.

    random_state : int, default=None
        Control the randomization of the algorithm.

    References
    ----------

    .. [1] https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/regular/vanillagan/model.py

    """

    def __init__(
            self,
            sample_multiplication_factor,
            only_sampled=False,
            batch_size=128,
            learning_rate=1e-4,
            betas=(0.5, 0.9),
            noise_dim=264,
            layers_dim=128,
            epochs=300,
            sample_interval=50,
            random_state=None
    ):
        self.sample_multiplication_factor = sample_multiplication_factor
        self.only_sampled = only_sampled
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.betas = betas
        self.noise_dim = noise_dim
        self.layers_dim = layers_dim
        self.epochs = epochs
        self.sample_interval = sample_interval
        self.random_state = random_state

    def fit(self, X, y=None):
        _set_global_random_state(self.random_state)

        original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
        original_dataset.columns = _convert_list_to_string_list(original_dataset.columns)

        self._vanilla_gan = VanilllaGAN(
            model_parameters=ModelParameters(
                batch_size=self.batch_size,
                lr=self.learning_rate,
                betas=self.betas,
                noise_dim=self.noise_dim,
                layers_dim=self.layers_dim
            )
        )

        self._vanilla_gan.train(
            data=original_dataset,
            train_arguments=TrainParameters(
                epochs=self.epochs,
                sample_interval=self.sample_interval
            ),
            num_cols=original_dataset.columns.tolist(),
            cat_cols=[]
        )

        return self

    def transform(self, X, y=None):
        """
        Runs a vanilla GAN sampling routine trained on the entire dataset.
        """
        _set_global_random_state(self.random_state)

        # return an empty dataframe and target or the original samples if the multiplication factor is too small
        if int(self.sample_multiplication_factor * len(X)) < 1:
            if self.only_sampled:
                return pd.DataFrame(columns=pd.DataFrame(X).columns), pd.Series([])
            else:
                return X, y

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns

        original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)

        # resample the original dataset
        resampled_dataset = self._vanilla_gan.sample(
            n_samples=int(self.sample_multiplication_factor * len(original_dataset))
        )
        resampled_dataset = pd.DataFrame(resampled_dataset)

        # drop excess samples
        resampled_dataset = resampled_dataset.head(
            int(self.sample_multiplication_factor * len(original_dataset))
        )
        resampled_dataset = resampled_dataset.reset_index(drop=True)

        if not self.only_sampled:
            # add original samples
            original_dataset.columns = range(len(original_dataset.columns))
            resampled_dataset.columns = range(len(resampled_dataset.columns))

            resampled_dataset = pd.concat([original_dataset, resampled_dataset], ignore_index=True)
            resampled_dataset.reset_index(drop=True)

            resampled_target = pd.concat([
                pd.Series(y).copy().reset_index(drop=True),
                pd.Series(
                    np.full(int(self.sample_multiplication_factor * len(X)), np.nan)
                ).copy().reset_index(drop=True)
            ], ignore_index=True)
            resampled_target = pd.Series(resampled_target).reset_index(drop=True)
        else:
            resampled_dataset = None

        # restore column titles if available
        if hasattr(X, 'columns'):
            resampled_dataset.columns = original_column_titles

        return resampled_dataset, resampled_target


class ProportionalConditionalGANSampler(ProportionalSampler):
    """
    Transformer that implements a proportional sampling routine using a
    conditional GAN implementation. For each class, we sample a proportional
    amount of samples using the condition vector.

    Parameters
    ----------

    sample_multiplication_factor : float
        Determines the relative amount of generated data, i.e. 0 means that no
        data is generated and 1 means that we generate the same number of data
        points as we already have.

    only_sampled : bool, default=False
        Determines whether the original dataset is prepended to the generated
        samples.

    batch_size: int, default=128
        Number of samples used per training step.

    learning_rate: float, default=1e-4
        The learning rate for each training step.

    betas: tuple, default=(0.5, 0.9)
        Initial decay rates of Adam when estimating the first and second
        moments of the gradient.

    noise_dim: int, default=264
        The length of the noise vector per example.

    layers_dim: int, default=128
        The dimension of the networks layers.

    epochs: int, default=300
        Total number of training steps.

    sample_interval: int, default=50
        The interval between samples.

    random_state : int, default=None
        Control the randomization of the algorithm.

    References
    ----------

    .. [1] https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/regular/cgan/model.py

    """

    def __init__(
            self,
            sample_multiplication_factor,
            only_sampled=False,
            batch_size=128,
            learning_rate=1e-4,
            betas=(0.5, 0.9),
            noise_dim=264,
            layers_dim=128,
            epochs=300,
            sample_interval=50,
            random_state=None
    ):
        self.sample_multiplication_factor = sample_multiplication_factor
        self.only_sampled = only_sampled
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.betas = betas
        self.noise_dim = noise_dim
        self.layers_dim = layers_dim
        self.epochs = epochs
        self.sample_interval = sample_interval
        self.random_state = random_state

    def fit(self, X, y):
        _set_global_random_state(self.random_state)

        # set the number of classes
        unique, _ = np.unique(y, return_counts=True)
        self._cgan = CGAN(
            model_parameters=ModelParameters(
                batch_size=self.batch_size,
                lr=self.learning_rate,
                betas=self.betas,
                noise_dim=self.noise_dim,
                layers_dim=self.layers_dim
            ),
            num_classes=len(unique)
        )

        # change the column titles for easier use
        original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
        y = pd.Series(y).copy().reset_index(drop=True)
        original_dataset.columns = _convert_list_to_string_list(range(0, len(original_dataset.columns)))
        num_cols = original_dataset.columns.copy().tolist()

        # add the target column to the dataset
        target_column_title = str(len(original_dataset.columns))
        original_dataset[target_column_title] = y

        self._cgan.train(
            data=original_dataset,
            label_col=target_column_title,
            train_arguments=TrainParameters(
                epochs=self.epochs,
                sample_interval=self.sample_interval
            ),
            num_cols=num_cols,
            cat_cols=[]
        )

        return self

    def transform(self, X, y):
        """
        Sample a proportional number of samples from the generated conditional
        GAN model.
        """
        _set_global_random_state(self.random_state)

        # return an empty dataframe and target or the original samples if the multiplication factor is too small
        if int(self.sample_multiplication_factor * len(X)) < 1:
            if self.only_sampled:
                return pd.DataFrame(columns=pd.DataFrame(X).columns), pd.Series([])
            else:
                return X, y

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns
        if hasattr(y, 'name'):
            original_target_title = y.name

        original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
        original_target = pd.Series(y).copy().reset_index(drop=True)

        # calculate the number of occurrences per class
        unique, counts = np.unique(original_target, return_counts=True)
        occurrences_per_class_dict = dict(zip(unique, counts))

        # resample the original dataset proportionally for each class
        resampled_subsets_per_class = []
        for class_name in occurrences_per_class_dict:
            number_of_samples = int(self.sample_multiplication_factor * occurrences_per_class_dict[class_name])
            condition = np.array([class_name])
            resampled_subset = self._cgan.sample(
                n_samples=number_of_samples,
                condition=condition
            )
            resampled_subset = pd.DataFrame(resampled_subset)

            # remove excess generated samples
            resampled_subset = resampled_subset.iloc[:number_of_samples, :]

            resampled_subsets_per_class.append(resampled_subset)

        # join resampled subsets into one dataset
        resampled_dataset = pd.concat(resampled_subsets_per_class, ignore_index=True)
        resampled_dataset = pd.DataFrame(resampled_dataset)

        # extract the target column from the generated dataset
        resampled_dataset.columns = range(0, len(resampled_dataset.columns))
        target_column_name = len(resampled_dataset.columns) - 1
        resampled_target = resampled_dataset[target_column_name]
        resampled_dataset = resampled_dataset.drop(columns=[target_column_name])

        # convert target entries to numpy because they are returned as tensorflow tensor
        resampled_target_numpy = []
        for entry in resampled_target:
            resampled_target_numpy.append(entry[0].numpy())
        resampled_target = pd.Series(resampled_target_numpy)

        resampled_dataset = resampled_dataset.reset_index(drop=True)
        resampled_target = resampled_target.reset_index(drop=True)

        if not self.only_sampled:
            # add original samples and target
            original_dataset.columns = range(len(original_dataset.columns))
            original_target.name = len(original_dataset.columns)
            resampled_dataset.columns = range(len(resampled_dataset.columns))
            resampled_target.name = len(resampled_dataset.columns)

            resampled_dataset = pd.concat([original_dataset, resampled_dataset], ignore_index=True)
            resampled_dataset.reset_index(drop=True)
            resampled_target = pd.concat([original_target, resampled_target], ignore_index=True)
            resampled_target.reset_index(drop=True)

        # restore column titles if available
        if hasattr(X, 'columns'):
            resampled_dataset.columns = original_column_titles
        if hasattr(y, 'name'):
            resampled_target.name = original_target_title

        return resampled_dataset, resampled_target


class UnlabeledConditionalGANSampler(UnlabeledSampler):
    """
    Transformer that implements an unlabeled sampling routine using a
    conditional GAN implementation where we set the target vector to be all one
    class.

    Parameters
    ----------

    sample_multiplication_factor : float
        Determines the relative amount of generated data, i.e. 0 means that no
        data is generated and 1 means that we generate the same number of data
        points as we already have.

    only_sampled : bool, default=False
        Determines whether the original dataset is prepended to the generated
        samples.

    batch_size: int, default=128
        Number of samples used per training step.

    learning_rate: float, default=1e-4
        The learning rate for each training step.

    betas: tuple, default=(0.5, 0.9)
        Initial decay rates of Adam when estimating the first and second
        moments of the gradient.

    noise_dim: int, default=264
        The length of the noise vector per example.

    layers_dim: int, default=128
        The dimension of the networks layers.

    epochs: int, default=300
        Total number of training steps.

    sample_interval: int, default=50
        The interval between samples.

    random_state : int, default=None
        Control the randomization of the algorithm.

    References
    ----------

    .. [1] https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/regular/vanillagan/model.py

    """

    def __init__(
            self,
            sample_multiplication_factor,
            only_sampled=False,
            batch_size=128,
            learning_rate=1e-4,
            betas=(0.5, 0.9),
            noise_dim=264,
            layers_dim=128,
            epochs=300,
            sample_interval=50,
            random_state=None
    ):
        self.sample_multiplication_factor = sample_multiplication_factor
        self.only_sampled = only_sampled
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.betas = betas
        self.noise_dim = noise_dim
        self.layers_dim = layers_dim
        self.epochs = epochs
        self.sample_interval = sample_interval
        self.random_state = random_state

    def fit(self, X, y=None):
        _set_global_random_state(self.random_state)

        # change the column titles for easier use
        original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
        original_dataset.columns = _convert_list_to_string_list(range(0, len(original_dataset.columns)))
        num_cols = original_dataset.columns.copy().tolist()

        # add the target column to the dataset
        target_column_title = len(original_dataset.columns)
        original_dataset[target_column_title] = np.full((len(X),), 0).T

        self._cgan = CGAN(
            model_parameters=ModelParameters(
                batch_size=self.batch_size,
                lr=self.learning_rate,
                betas=self.betas,
                noise_dim=self.noise_dim,
                layers_dim=self.layers_dim
            ),
            num_classes=1
        )

        self._cgan.train(
            data=original_dataset,
            label_col=target_column_title,
            train_arguments=TrainParameters(
                epochs=self.epochs,
                sample_interval=self.sample_interval
            ),
            num_cols=num_cols,
            cat_cols=[]
        )

        return self

    def transform(self, X, y=None):
        """
        Sample a proportional number of samples from the generated conditional
        GAN model.
        """
        _set_global_random_state(self.random_state)

        # return an empty dataframe and target or the original samples if the multiplication factor is too small
        if int(self.sample_multiplication_factor * len(X)) < 1:
            if self.only_sampled:
                return pd.DataFrame(columns=pd.DataFrame(X).columns), pd.Series([])
            else:
                return X, y

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns

        original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)

        number_of_samples = int(self.sample_multiplication_factor * len(X))
        condition = np.array([0])
        resampled_dataset = self._cgan.sample(
            n_samples=number_of_samples,
            condition=condition
        )
        resampled_dataset = pd.DataFrame(resampled_dataset)

        # remove excess generated samples
        resampled_dataset = resampled_dataset.iloc[:number_of_samples, :]

        # drop the target column from the generated dataset
        resampled_dataset.columns = range(0, len(resampled_dataset.columns))
        resampled_dataset = resampled_dataset.drop(columns=[len(resampled_dataset.columns) - 1])
        resampled_dataset = resampled_dataset.reset_index(drop=True)

        if not self.only_sampled:
            # add original samples
            original_dataset.columns = range(len(original_dataset.columns))
            resampled_dataset.columns = range(len(resampled_dataset.columns))

            resampled_dataset = pd.concat([original_dataset, resampled_dataset], ignore_index=True)
            resampled_dataset.reset_index(drop=True)

            resampled_target = pd.concat([
                pd.Series(y).copy().reset_index(drop=True),
                pd.Series(
                    np.full(int(self.sample_multiplication_factor * len(X)), np.nan)
                ).copy().reset_index(drop=True)
            ], ignore_index=True)
            resampled_target = pd.Series(resampled_target).reset_index(drop=True)
        else:
            resampled_dataset = None

        # restore column titles if available
        if hasattr(X, 'columns'):
            resampled_dataset.columns = original_column_titles

        return resampled_dataset, resampled_target


class ProportionalDRAGANSampler(ProportionalSampler):
    """
    Transformer that implements a proportional sampling routine using a DRAGAN
    implementation. We train and sample from a different DRAGAN model for each
    class.

    Parameters
    ----------

    sample_multiplication_factor : float
        Determines the relative amount of generated data, i.e. 0 means that no
        data is generated and 1 means that we generate the same number of data
        points as we already have.

    only_sampled : bool, default=False
        Determines whether the original dataset is prepended to the generated
        samples.

    discriminator_updates_per_step : int, default=1
        Determines how many times the discriminator is updated in each training
        step.

    batch_size: int, default=128
        Number of samples used per training step.

    learning_rate: float, default=1e-4
        The learning rate for each training step.

    betas: tuple, default=(0.5, 0.9)
        Initial decay rates of Adam when estimating the first and second
        moments of the gradient.

    noise_dim: int, default=264
        The length of the noise vector per example.

    layers_dim: int, default=128
        The dimension of the networks layers.

    epochs: int, default=300
        Total number of training steps.

    sample_interval: int, default=50
        The interval between samples.

    random_state : int, default=None
        Control the randomization of the algorithm.

    References
    ----------

    .. [1] https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/regular/dragan/model.py

    """

    def __init__(
            self,
            sample_multiplication_factor,
            only_sampled=False,
            discriminator_updates_per_step=1,
            batch_size=128,
            learning_rate=1e-4,
            betas=(0.5, 0.9),
            noise_dim=264,
            layers_dim=128,
            epochs=300,
            sample_interval=50,
            random_state=None
    ):
        self.sample_multiplication_factor = sample_multiplication_factor
        self.only_sampled = only_sampled
        self.discriminator_updates_per_step = discriminator_updates_per_step
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.betas = betas
        self.noise_dim = noise_dim
        self.layers_dim = layers_dim
        self.epochs = epochs
        self.sample_interval = sample_interval
        self.random_state = random_state

    def fit(self, X, y):
        _set_global_random_state(self.random_state)

        # reset dragan models
        self._dragan = {}

        # change the column titles for easier use
        original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
        y = pd.Series(y).copy().reset_index(drop=True)
        original_dataset.columns = _convert_list_to_string_list(range(0, len(original_dataset.columns)))
        num_cols = original_dataset.columns.copy().tolist()

        # calculate the number of occurrences per class
        unique, counts = np.unique(y, return_counts=True)
        occurrences_per_class_dict = dict(zip(unique, counts))

        for class_name in occurrences_per_class_dict:
            self._dragan[class_name] = DRAGAN(
                model_parameters=ModelParameters(
                    batch_size=self.batch_size,
                    lr=self.learning_rate,
                    betas=self.betas,
                    noise_dim=self.noise_dim,
                    layers_dim=self.layers_dim
                ),
                n_discriminator=self.discriminator_updates_per_step
            )

            original_subset = original_dataset.iloc[[x for x in range(0, len(y)) if y[x] == class_name]]

            self._dragan[class_name].train(
                data=original_subset,
                train_arguments=TrainParameters(
                    epochs=self.epochs,
                    sample_interval=self.sample_interval
                ),
                num_cols=num_cols,
                cat_cols=[]
            )

        return self

    def transform(self, X, y):
        """
        Sample proportionally from each of the DRAGANs trained on the subsets
        split by class.
        """
        _set_global_random_state(self.random_state)

        # return an empty dataframe and target or the original samples if the multiplication factor is too small
        if int(self.sample_multiplication_factor * len(X)) < 1:
            if self.only_sampled:
                return pd.DataFrame(columns=pd.DataFrame(X).columns), pd.Series([])
            else:
                return X, y

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns
        if hasattr(y, 'name'):
            original_target_title = y.name

        original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
        original_target = pd.Series(y).copy().reset_index(drop=True)

        # calculate the number of occurrences per class
        unique, counts = np.unique(original_target, return_counts=True)
        occurrences_per_class_dict = dict(zip(unique, counts))

        resampled_subsets_per_class = []
        resampled_targets_per_class = []
        for class_name in occurrences_per_class_dict:
            number_of_samples = int(self.sample_multiplication_factor * occurrences_per_class_dict[class_name])

            resampled_subset = self._dragan[class_name].sample(
                n_samples=number_of_samples,
            )
            resampled_target = np.full((number_of_samples,), class_name).T

            resampled_subset = pd.DataFrame(resampled_subset)
            resampled_target = pd.Series(resampled_target)

            # remove excess generated samples
            resampled_subset = resampled_subset.iloc[:number_of_samples, :]

            resampled_subsets_per_class.append(resampled_subset)
            resampled_targets_per_class.append(resampled_target)

        # join resampled subsets and targets
        resampled_dataset = pd.concat(resampled_subsets_per_class, ignore_index=True)
        resampled_dataset = resampled_dataset.reset_index(drop=True)
        resampled_target = pd.concat(resampled_targets_per_class, ignore_index=True)
        resampled_target = resampled_target.reset_index(drop=True)

        if not self.only_sampled:
            # add original samples and target
            original_dataset.columns = range(len(original_dataset.columns))
            original_target.name = len(original_dataset.columns)
            resampled_dataset.columns = range(len(resampled_dataset.columns))
            resampled_target.name = len(resampled_dataset.columns)

            resampled_dataset = pd.concat([original_dataset, resampled_dataset], ignore_index=True)
            resampled_dataset.reset_index(drop=True)
            resampled_target = pd.concat([original_target, resampled_target], ignore_index=True)
            resampled_target.reset_index(drop=True)

        # restore column titles if available
        if hasattr(X, 'columns'):
            resampled_dataset.columns = original_column_titles
        if hasattr(y, 'name'):
            resampled_target.name = original_target_title

        return resampled_dataset, resampled_target


class UnlabeledDRAGANSampler(UnlabeledSampler):
    """
    Transformer that implements an unlabeled sampling routine using a DRAGAN
    implementation.

    Parameters
    ----------

    sample_multiplication_factor : float
        Determines the relative amount of generated data, i.e. 0 means that no
        data is generated and 1 means that we generate the same number of data
        points as we already have.

    only_sampled : bool, default=False
        Determines whether the original dataset is prepended to the generated
        samples.

    discriminator_updates_per_step : int, default=1
        Determines how many times the discriminator is updated in each training
        step.

    batch_size: int, default=128
        Number of samples used per training step.

    learning_rate: float, default=1e-4
        The learning rate for each training step.

    betas: tuple, default=(0.5, 0.9)
        Initial decay rates of Adam when estimating the first and second
        moments of the gradient.

    noise_dim: int, default=264
        The length of the noise vector per example.

    layers_dim: int, default=128
        The dimension of the networks layers.

    epochs: int, default=300
        Total number of training steps.

    sample_interval: int, default=50
        The interval between samples.

    random_state : int, default=None
        Control the randomization of the algorithm.

    References
    ----------

    .. [1] https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/regular/dragan/model.py

    """

    def __init__(
            self,
            sample_multiplication_factor,
            only_sampled=False,
            discriminator_updates_per_step=1,
            batch_size=128,
            learning_rate=1e-4,
            betas=(0.5, 0.9),
            noise_dim=264,
            layers_dim=128,
            epochs=300,
            sample_interval=50,
            random_state=None
    ):
        self.sample_multiplication_factor = sample_multiplication_factor
        self.only_sampled = only_sampled
        self.discriminator_updates_per_step = discriminator_updates_per_step
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.betas = betas
        self.noise_dim = noise_dim
        self.layers_dim = layers_dim
        self.epochs = epochs
        self.sample_interval = sample_interval
        self.random_state = random_state

    def fit(self, X, y=None):
        _set_global_random_state(self.random_state)

        # change the column titles for easier use
        original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
        original_dataset.columns = _convert_list_to_string_list(range(0, len(original_dataset.columns)))
        num_cols = original_dataset.columns.copy().tolist()

        self._dragan = DRAGAN(
            model_parameters=ModelParameters(
                batch_size=self.batch_size,
                lr=self.learning_rate,
                betas=self.betas,
                noise_dim=self.noise_dim,
                layers_dim=self.layers_dim
            ),
            n_discriminator=self.discriminator_updates_per_step
        )

        self._dragan.train(
            data=original_dataset,
            train_arguments=TrainParameters(
                epochs=self.epochs,
                sample_interval=self.sample_interval
            ),
            num_cols=num_cols,
            cat_cols=[]
        )

        return self

    def transform(self, X, y=None):
        """
        Sample the requested number of samples from the trained GAN. Returns
        only the generated data.
        """
        _set_global_random_state(self.random_state)

        # return an empty dataframe and target or the original samples if the multiplication factor is too small
        if int(self.sample_multiplication_factor * len(X)) < 1:
            if self.only_sampled:
                return pd.DataFrame(columns=pd.DataFrame(X).columns), pd.Series([])
            else:
                return X, y

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns

        original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)

        number_of_samples = int(self.sample_multiplication_factor * len(X))
        resampled_dataset = self._dragan.sample(n_samples=number_of_samples)
        resampled_dataset = pd.DataFrame(resampled_dataset)

        # remove excess generated samples
        resampled_dataset = resampled_dataset.iloc[:number_of_samples, :]
        resampled_dataset = resampled_dataset.reset_index(drop=True)

        if not self.only_sampled:
            # add original samples
            original_dataset.columns = range(len(original_dataset.columns))
            resampled_dataset.columns = range(len(resampled_dataset.columns))

            resampled_dataset = pd.concat([original_dataset, resampled_dataset], ignore_index=True)
            resampled_dataset.reset_index(drop=True)

            resampled_target = pd.concat([
                pd.Series(y).copy().reset_index(drop=True),
                pd.Series(
                    np.full(int(self.sample_multiplication_factor * len(X)), np.nan)
                ).copy().reset_index(drop=True)
            ], ignore_index=True)
            resampled_target = pd.Series(resampled_target).reset_index(drop=True)
        else:
            resampled_dataset = None

        # restore column titles if available
        if hasattr(X, 'columns'):
            resampled_dataset.columns = original_column_titles

        return resampled_dataset, resampled_target


def _convert_list_to_string_list(item_list):
    string_list = []
    for item in item_list:
        string_list.append(str(item))
    return string_list


def _set_global_random_state(random_state):
    random.seed(random_state)
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
