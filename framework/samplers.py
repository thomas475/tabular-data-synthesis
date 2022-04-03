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


class Sampler(BaseEstimator, TransformerMixin):
    """
    Dummy class that allows us to modify only the methods that interest us,
    avoiding redundancy.
    """

    def __init__(self):
        return None

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X, y)


class ProportionalSMOTESampler(Sampler):
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
            random_state=None,
            k_neighbors=5,
            n_jobs=None
    ):
        self._sample_multiplication_factor = sample_multiplication_factor
        self._random_state = random_state
        self._k_neighbors = k_neighbors
        self._n_jobs = n_jobs

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        """
        Count number of occurrences of each class and resample proportionally.
        Returns only the generated data.
        """
        # return an empty dataframe if the sample multiplication factor is too small
        if int(self._sample_multiplication_factor * len(X)) < 1:
            return pd.DataFrame(columns=X.columns)

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns

        # calculate the number of occurrences per class
        unique, counts = np.unique(y, return_counts=True)
        occurrences_per_class_dict = dict(zip(unique, counts))

        # set the amount of samples per class we want to have; includes original samples
        sampling_strategy = {}
        for class_name in occurrences_per_class_dict:
            sampling_strategy[class_name] = int(
                (self._sample_multiplication_factor + 1) * occurrences_per_class_dict[class_name]
            )

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self._random_state,
            k_neighbors=self._k_neighbors,
            n_jobs=self._n_jobs
        )

        # filter user warning that fires because we do not use SMOTE simply for balancing the dataset
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            X_resampled, y_resampled = smote.fit_resample(X, y)
        X_resampled = pd.DataFrame(X_resampled)

        # remove original dataset
        X_resampled = pd.DataFrame(X_resampled.iloc[len(X):, :])
        X_resampled = X_resampled.reset_index(drop=True)

        # restore column titles if available
        if hasattr(X, 'columns'):
            X_resampled.columns = original_column_titles

        return X_resampled


class UnlabeledSMOTESampler(Sampler):
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
            random_state=None,
            k_neighbors=5,
            n_jobs=None
    ):
        self._sample_multiplication_factor = sample_multiplication_factor
        self._random_state = random_state
        self._k_neighbors = k_neighbors
        self._n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        No labels available, so we view the entire set as belonging to one class
        and use SMOTE on the entire dataset. Because SMOTE requires there to be
        at least two classes, we add a dummy label vector and one dummy data
        entry. After we run SMOTE, we remove the dummy entry. Returns only the
        generated data.
        """
        # return an empty dataframe if the sample multiplication factor is too small
        if int(self._sample_multiplication_factor * len(X)) < 1:
            return pd.DataFrame(columns=X.columns)

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns

        # create a dummy target vector [ 0 1 ... 1 ]
        y_dummy = np.append(0, np.full((len(X), 1), 1))

        # set sampling strategy to ignore the dummy class and resample the rest
        sampling_strategy = {
            0: 1,
            1: int((self._sample_multiplication_factor + 1) * len(X))
        }

        # insert a dummy entry on index 0
        X_dummy = pd.DataFrame(X).copy()
        X_dummy.loc[-1] = [0] * len(X_dummy.columns)
        X_dummy.index = X_dummy.index + 1
        X_dummy.sort_index(inplace=True)

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self._random_state,
            k_neighbors=self._k_neighbors,
            n_jobs=self._n_jobs
        )

        # filter user warning that fires because we do not use SMOTE simply for balancing the dataset
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            X_dummy_resampled, y_dummy_resampled = smote.fit_resample(X_dummy, y_dummy)
        X_dummy_resampled = pd.DataFrame(X_dummy_resampled)

        # remove the dummy entry
        X_resampled = X_dummy_resampled.iloc[1:, :].reset_index(drop=True)

        # remove original dataset
        X_resampled = pd.DataFrame(X_resampled.iloc[len(X):, :])
        X_resampled = X_resampled.reset_index(drop=True)

        # restore column titles if available
        if hasattr(X, 'columns'):
            X_dummy_resampled.columns = original_column_titles

        return X_resampled


class ProportionalRACOGSampler(Sampler):
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

    burnin : int, default=100
        It determines how many examples generated for a given one are going to
        be discarded firstly.

    lag : int, default=20
        Number of iterations between new generated example for a minority one.

    References
    ----------

    .. [1] https://rdrr.io/cran/imbalance/man/racog.html

    .. [2] B. Das, N. C. Krishnan and D. J. Cook, "RACOG and wRACOG: Two
           Probabilistic Oversampling Techniques," in IEEE Transactions on
           Knowledge and Data Engineering, vol. 27, no. 1, pp. 222-234,
           1 Jan. 2015, doi: 10.1109/TKDE.2014.2324567.

    """

    def __init__(
            self,
            sample_multiplication_factor,
            burnin=100,
            lag=20
    ):
        self._sample_multiplication_factor = sample_multiplication_factor
        self._racog = RACOG(burnin, lag)

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        """
        Split dataset by target vector and resample each subset independently.
        Afterwards join the resampled subsets. Returns only the generated data.
        """
        # return an empty dataframe if the sample multiplication factor is too small
        if int(self._sample_multiplication_factor * len(X)) < 1:
            return pd.DataFrame(columns=X.columns)

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns

        original_dataset = pd.DataFrame(X).copy()

        # adding target column
        target_column_title = str(len(original_dataset.columns))
        original_dataset[target_column_title] = y

        # split dataset into subsets according to the target column
        original_subset_per_class = [
            x for _, x in original_dataset.groupby(original_dataset[target_column_title])
        ]

        # resample each subset independently
        resampled_subset_per_class = []
        for original_subset in original_subset_per_class:
            resampled_subset = self._racog.resample(
                dataset=original_subset,
                num_instances=int(self._sample_multiplication_factor * len(original_subset)),
                class_attr=target_column_title
            )
            resampled_subset_per_class.append(resampled_subset)

        # join the resampled subsets into one dataframe
        resampled_dataset = pd.concat(resampled_subset_per_class, ignore_index=True)
        resampled_dataset = resampled_dataset.drop(columns=[target_column_title])

        # restore column titles if available
        if hasattr(X, 'columns'):
            resampled_dataset.columns = original_column_titles

        return resampled_dataset


class UnlabeledRACOGSampler(Sampler):
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

    burnin : int, default=100
        It determines how many examples generated for a given one are going to
        be discarded firstly.

    lag : int, default=20
        Number of iterations between new generated example for a minority one.

    References
    ----------

    .. [1] https://rdrr.io/cran/imbalance/man/racog.html

    .. [2] B. Das, N. C. Krishnan and D. J. Cook, "RACOG and wRACOG: Two
           Probabilistic Oversampling Techniques," in IEEE Transactions on
           Knowledge and Data Engineering, vol. 27, no. 1, pp. 222-234,
           1 Jan. 2015, doi: 10.1109/TKDE.2014.2324567.

    """

    def __init__(
            self,
            sample_multiplication_factor,
            burnin=100,
            lag=20
    ):
        self._sample_multiplication_factor = sample_multiplication_factor
        self._racog = RACOG(burnin, lag)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        No labels available, so we view the entire set as belonging to one class
        and use RACOG on the entire dataset. Because RACOG requires there to be
        a target vector, we add a dummy label vector that assigns each sample
        to the same class. Returns only the generated data.
        """
        # return an empty dataframe if the sample multiplication factor is too small
        if int(self._sample_multiplication_factor * len(X)) < 1:
            return pd.DataFrame(columns=X.columns)

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns

        original_dataset = pd.DataFrame(X).copy()

        # adding target column that only consists of one class
        target_column_title = str(len(original_dataset.columns))
        original_dataset[target_column_title] = np.full((len(X), 1), 1)

        resampled_dataset = self._racog.resample(
            dataset=original_dataset,
            num_instances=int(self._sample_multiplication_factor * len(original_dataset)),
            class_attr=target_column_title
        )

        # drop the target column
        resampled_dataset = pd.DataFrame(resampled_dataset).reset_index(drop=True)
        resampled_dataset = resampled_dataset.drop(columns=[target_column_title])

        # restore column titles if available
        if hasattr(X, 'columns'):
            resampled_dataset.columns = original_column_titles

        return resampled_dataset


class ProportionalVanillaGANSampler(Sampler):
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

    References
    ----------

    .. [1] https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/regular/vanillagan/model.py

    """

    def __init__(
            self,
            sample_multiplication_factor,
            batch_size=128,
            learning_rate=1e-4,
            betas=(0.5, 0.9),
            noise_dim=264,
            layers_dim=128,
            epochs=300,
            sample_interval=50
    ):
        self._sample_multiplication_factor = sample_multiplication_factor

        self._model_parameters = ModelParameters(
            batch_size=batch_size,
            lr=learning_rate,
            betas=betas,
            noise_dim=noise_dim,
            layers_dim=layers_dim
        )

        self._train_parameters = TrainParameters(
            epochs=epochs,
            sample_interval=sample_interval
        )

        self._vanilla_gan = {}

    def fit(self, X, y=None):
        # reset dragan models
        self._vanilla_gan = {}

        # change the column titles for easier use
        original_dataset = pd.DataFrame(X).copy()
        original_dataset.columns = _convert_list_to_string_list(range(0, len(original_dataset.columns)))
        num_cols = original_dataset.columns.copy().tolist()

        # calculate the number of occurrences per class
        unique, counts = np.unique(y, return_counts=True)
        occurrences_per_class_dict = dict(zip(unique, counts))

        for class_name in occurrences_per_class_dict:
            self._vanilla_gan[class_name] = VanilllaGAN(model_parameters=self._model_parameters)

            original_subset = original_dataset.iloc[[x for x in range(0, len(y)) if y[x] == class_name]]

            self._vanilla_gan[class_name].train(
                data=original_subset,
                train_arguments=self._train_parameters,
                num_cols=num_cols,
                cat_cols=[]
            )

        return self

    def transform(self, X, y=None):
        """
        Sample proportionally from each of the DRAGANs trained on the subsets
        split by class. Returns only the generated data.
        """
        # return an empty dataframe if the sample multiplication factor is too small
        if int(self._sample_multiplication_factor * len(X)) < 1:
            return pd.DataFrame(columns=X.columns)

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns

        original_dataset = pd.DataFrame(X).copy()

        # calculate the number of occurrences per class
        unique, counts = np.unique(y, return_counts=True)
        occurrences_per_class_dict = dict(zip(unique, counts))

        resampled_subsets_per_class = []
        for class_name in occurrences_per_class_dict:
            number_of_samples = int(self._sample_multiplication_factor * occurrences_per_class_dict[class_name])
            resampled_subset = self._vanilla_gan[class_name].sample(
                n_samples=number_of_samples,
            )
            resampled_subset = pd.DataFrame(resampled_subset)

            # remove excess generated samples
            resampled_subset = resampled_subset.iloc[:number_of_samples, :]

            resampled_subsets_per_class.append(resampled_subset)

        # join resampled subsets into one dataset
        resampled_dataset = pd.concat(resampled_subsets_per_class, ignore_index=True)
        resampled_dataset = resampled_dataset.reset_index(drop=True)

        # restore column titles if available
        if hasattr(X, 'columns'):
            resampled_dataset.columns = original_column_titles

        return resampled_dataset


class UnlabeledVanillaGANSampler(Sampler):
    """
    Transformer that implements a sampling routine for a trained vanilla GAN
    model on unlabeled data.

    Parameters
    ----------

    sample_multiplication_factor : float
        Determines the relative amount of generated data, i.e. 0 means that no
        data is generated and 1 means that we generate the same number of data
        points as we already have.

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

    References
    ----------

    .. [1] https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/regular/vanillagan/model.py

    """

    def __init__(
            self,
            sample_multiplication_factor,
            batch_size=128,
            learning_rate=1e-4,
            betas=(0.5, 0.9),
            noise_dim=264,
            layers_dim=128,
            epochs=300,
            sample_interval=50
    ):
        self._sample_multiplication_factor = sample_multiplication_factor

        self._model_parameters = ModelParameters(
            batch_size=batch_size,
            lr=learning_rate,
            betas=betas,
            noise_dim=noise_dim,
            layers_dim=layers_dim
        )

        self._train_parameters = TrainParameters(
            epochs=epochs,
            sample_interval=sample_interval
        )

        self._vanilla_gan = VanilllaGAN(model_parameters=self._model_parameters)

    def fit(self, X, y=None):
        original_dataset = pd.DataFrame(X)
        original_dataset.columns = _convert_list_to_string_list(original_dataset.columns)

        self._vanilla_gan.train(
            data=original_dataset,
            train_arguments=self._train_parameters,
            num_cols=original_dataset.columns.tolist(),
            cat_cols=[]
        )

        return self

    def transform(self, X, y=None):
        """
        Returns only the generated data.
        """
        # return an empty dataframe if the sample multiplication factor is too small
        if int(self._sample_multiplication_factor * len(X)) < 1:
            return pd.DataFrame(columns=X.columns)

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns

        original_dataset = pd.DataFrame(X).copy()

        # resample the original dataset
        resampled_dataset = self._vanilla_gan.sample(
            n_samples=int(self._sample_multiplication_factor * len(original_dataset))
        )
        resampled_dataset = pd.DataFrame(resampled_dataset)

        # drop excess samples
        resampled_dataset = resampled_dataset.head(
            int(self._sample_multiplication_factor * len(original_dataset))
        )
        resampled_dataset = resampled_dataset.reset_index(drop=True)

        # restore column titles if available
        if hasattr(X, 'columns'):
            resampled_dataset.columns = original_column_titles

        return resampled_dataset


class ProportionalConditionalGANSampler(Sampler):
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

    References
    ----------

    .. [1] https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/regular/cgan/model.py

    """

    def __init__(
            self,
            sample_multiplication_factor,
            batch_size=128,
            learning_rate=1e-4,
            betas=(0.5, 0.9),
            noise_dim=264,
            layers_dim=128,
            epochs=300,
            sample_interval=50
    ):
        self._sample_multiplication_factor = sample_multiplication_factor

        self._model_parameters = ModelParameters(
            batch_size=batch_size,
            lr=learning_rate,
            betas=betas,
            noise_dim=noise_dim,
            layers_dim=layers_dim
        )

        self._train_parameters = TrainParameters(
            epochs=epochs,
            sample_interval=sample_interval
        )

        self._cgan = CGAN(model_parameters=self._model_parameters, num_classes=1)

    def fit(self, X, y=None):
        # set the number of classes
        unique, _ = np.unique(y, return_counts=True)
        self._cgan.num_classes = len(unique)

        # change the column titles for easier use
        original_dataset = pd.DataFrame(X).copy()
        original_dataset.columns = _convert_list_to_string_list(range(0, len(original_dataset.columns)))
        num_cols = original_dataset.columns.copy().tolist()

        # add the target column to the dataset
        target_column_title = len(original_dataset.columns)
        original_dataset[target_column_title] = y

        self._cgan.train(
            data=original_dataset,
            label_col=target_column_title,
            train_arguments=self._train_parameters,
            num_cols=num_cols,
            cat_cols=[]
        )

        return self

    def transform(self, X, y=None):
        """
        Sample a proportional number of samples from the generated conditional
        GAN model. Returns only the generated data.
        """
        # return an empty dataframe if the sample multiplication factor is too small
        if int(self._sample_multiplication_factor * len(X)) < 1:
            return pd.DataFrame(columns=X.columns)

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns

        original_dataset = pd.DataFrame(X).copy()

        # calculate the number of occurrences per class
        unique, counts = np.unique(y, return_counts=True)
        occurrences_per_class_dict = dict(zip(unique, counts))

        # resample the original dataset proportionally for each class
        resampled_subsets_per_class = []
        for class_name in occurrences_per_class_dict:
            number_of_samples = int(self._sample_multiplication_factor * occurrences_per_class_dict[class_name])
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

        # drop the target column from the generated dataset
        resampled_dataset.columns = range(0, len(resampled_dataset.columns))
        resampled_dataset = resampled_dataset.drop(columns=[len(resampled_dataset.columns) - 1])
        resampled_dataset = resampled_dataset.reset_index(drop=True)

        # restore column titles if available
        if hasattr(X, 'columns'):
            resampled_dataset.columns = original_column_titles

        return resampled_dataset


class UnlabeledConditionalGANSampler(Sampler):
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

    References
    ----------

    .. [1] https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/regular/vanillagan/model.py

    """

    def __init__(
            self,
            sample_multiplication_factor,
            batch_size=128,
            learning_rate=1e-4,
            betas=(0.5, 0.9),
            noise_dim=264,
            layers_dim=128,
            epochs=300,
            sample_interval=50
    ):
        self._sample_multiplication_factor = sample_multiplication_factor

        self._model_parameters = ModelParameters(
            batch_size=batch_size,
            lr=learning_rate,
            betas=betas,
            noise_dim=noise_dim,
            layers_dim=layers_dim
        )

        self._train_parameters = TrainParameters(
            epochs=epochs,
            sample_interval=sample_interval
        )

        self._cgan = CGAN(model_parameters=self._model_parameters, num_classes=1)

    def fit(self, X, y=None):
        # change the column titles for easier use
        original_dataset = pd.DataFrame(X).copy()
        original_dataset.columns = _convert_list_to_string_list(range(0, len(original_dataset.columns)))
        num_cols = original_dataset.columns.copy().tolist()

        # add the target column to the dataset
        target_column_title = len(original_dataset.columns)
        original_dataset[target_column_title] = np.full((len(X),), 0).T

        self._cgan.train(
            data=original_dataset,
            label_col=target_column_title,
            train_arguments=self._train_parameters,
            num_cols=num_cols,
            cat_cols=[]
        )

        return self

    def transform(self, X, y=None):
        """
        Sample a proportional number of samples from the generated conditional
        GAN model. Returns only the generated data.
        """
        # return an empty dataframe if the sample multiplication factor is too small
        if int(self._sample_multiplication_factor * len(X)) < 1:
            return pd.DataFrame(columns=X.columns)

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns

        original_dataset = pd.DataFrame(X).copy()

        number_of_samples = int(self._sample_multiplication_factor * len(X))
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

        # restore column titles if available
        if hasattr(X, 'columns'):
            resampled_dataset.columns = original_column_titles

        return resampled_dataset


class ProportionalDRAGANSampler(Sampler):
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

    References
    ----------

    .. [1] https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/regular/dragan/model.py

    """

    def __init__(
            self,
            sample_multiplication_factor,
            discriminator_updates_per_step=1,
            batch_size=128,
            learning_rate=1e-4,
            betas=(0.5, 0.9),
            noise_dim=264,
            layers_dim=128,
            epochs=300,
            sample_interval=50
    ):
        self._sample_multiplication_factor = sample_multiplication_factor
        self._discriminator_updates_per_step = discriminator_updates_per_step

        self._model_parameters = ModelParameters(
            batch_size=batch_size,
            lr=learning_rate,
            betas=betas,
            noise_dim=noise_dim,
            layers_dim=layers_dim
        )

        self._train_parameters = TrainParameters(
            epochs=epochs,
            sample_interval=sample_interval
        )

        self._dragan = {}

    def fit(self, X, y=None):
        # reset dragan models
        self._dragan = {}

        # change the column titles for easier use
        original_dataset = pd.DataFrame(X).copy()
        original_dataset.columns = _convert_list_to_string_list(range(0, len(original_dataset.columns)))
        num_cols = original_dataset.columns.copy().tolist()

        # calculate the number of occurrences per class
        unique, counts = np.unique(y, return_counts=True)
        occurrences_per_class_dict = dict(zip(unique, counts))

        for class_name in occurrences_per_class_dict:
            self._dragan[class_name] = DRAGAN(
                model_parameters=self._model_parameters,
                n_discriminator=self._discriminator_updates_per_step
            )

            original_subset = original_dataset.iloc[[x for x in range(0, len(y)) if y[x] == class_name]]

            self._dragan[class_name].train(
                data=original_subset,
                train_arguments=self._train_parameters,
                num_cols=num_cols,
                cat_cols=[]
            )

        return self

    def transform(self, X, y=None):
        """
        Sample proportionally from each of the DRAGANs trained on the subsets
        split by class. Returns only the generated data.
        """
        # return an empty dataframe if the sample multiplication factor is too small
        if int(self._sample_multiplication_factor * len(X)) < 1:
            return pd.DataFrame(columns=X.columns)

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns

        original_dataset = pd.DataFrame(X).copy()

        # calculate the number of occurrences per class
        unique, counts = np.unique(y, return_counts=True)
        occurrences_per_class_dict = dict(zip(unique, counts))

        resampled_subsets_per_class = []
        for class_name in occurrences_per_class_dict:
            number_of_samples = int(self._sample_multiplication_factor * occurrences_per_class_dict[class_name])
            resampled_subset = self._dragan[class_name].sample(
                n_samples=number_of_samples,
            )
            resampled_subset = pd.DataFrame(resampled_subset)

            # remove excess generated samples
            resampled_subset = resampled_subset.iloc[:number_of_samples, :]

            resampled_subsets_per_class.append(resampled_subset)

        # join resampled subsets into one dataset
        resampled_dataset = pd.concat(resampled_subsets_per_class, ignore_index=True)
        resampled_dataset = resampled_dataset.reset_index(drop=True)

        # restore column titles if available
        if hasattr(X, 'columns'):
            resampled_dataset.columns = original_column_titles

        return resampled_dataset


class UnlabeledDRAGANSampler(Sampler):
    """
    Transformer that implements an unlabeled sampling routine using a DRAGAN
    implementation.

    Parameters
    ----------

    sample_multiplication_factor : float
        Determines the relative amount of generated data, i.e. 0 means that no
        data is generated and 1 means that we generate the same number of data
        points as we already have.

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

    References
    ----------

    .. [1] https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/regular/dragan/model.py

    """

    def __init__(
            self,
            sample_multiplication_factor,
            discriminator_updates_per_step=1,
            batch_size=128,
            learning_rate=1e-4,
            betas=(0.5, 0.9),
            noise_dim=264,
            layers_dim=128,
            epochs=300,
            sample_interval=50
    ):
        self._sample_multiplication_factor = sample_multiplication_factor

        self._model_parameters = ModelParameters(
            batch_size=batch_size,
            lr=learning_rate,
            betas=betas,
            noise_dim=noise_dim,
            layers_dim=layers_dim
        )

        self._train_parameters = TrainParameters(
            epochs=epochs,
            sample_interval=sample_interval
        )

        self._dragan = DRAGAN(model_parameters=self._model_parameters, n_discriminator=discriminator_updates_per_step)

    def fit(self, X, y=None):
        # change the column titles for easier use
        original_dataset = pd.DataFrame(X).copy()
        original_dataset.columns = _convert_list_to_string_list(range(0, len(original_dataset.columns)))
        num_cols = original_dataset.columns.copy().tolist()

        self._dragan.train(
            data=original_dataset,
            train_arguments=self._train_parameters,
            num_cols=num_cols,
            cat_cols=[]
        )

        return self

    def transform(self, X, y=None):
        """
        Sample the requested number of samples from the trained GAN. Returns
        only the generated data.
        """
        # return an empty dataframe if the sample multiplication factor is too small
        if int(self._sample_multiplication_factor * len(X)) < 1:
            return pd.DataFrame(columns=X.columns)

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns

        original_dataset = pd.DataFrame(X).copy()

        number_of_samples = int(self._sample_multiplication_factor * len(X))
        resampled_dataset = self._dragan.sample(n_samples=number_of_samples)
        resampled_dataset = pd.DataFrame(resampled_dataset)

        # remove excess generated samples
        resampled_dataset = resampled_dataset.iloc[:number_of_samples, :]
        resampled_dataset = resampled_dataset.reset_index(drop=True)

        # restore column titles if available
        if hasattr(X, 'columns'):
            resampled_dataset.columns = original_column_titles

        return resampled_dataset


def _convert_list_to_string_list(item_list):
    string_list = []
    for item in item_list:
        string_list.append(str(item))
    return string_list
