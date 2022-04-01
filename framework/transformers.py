"""Transformer wrappers for SMOTE, GAN and Gibbs sampling for use as samplers
   in sklearn.pipeline.Pipeline"""

# Authors: Thomas Frank <thomas-frank01@gmx.de>
# License: MIT

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from imblearn.over_sampling import SMOTE
from racog import RACOG
from ydata_synthetic.synthesizers.regular import VanilllaGAN
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters


class DummyTransformer(BaseEstimator, TransformerMixin):
    """
    Dummy class that allows us to modify only the methods that interest us,
    avoiding redudancy.
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


class ProportionalSMOTETransformer(DummyTransformer):
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
        Count number of occurences of each class and resample proportionally.
        Returns original and generated data.
        """
        # just return the original dataset if the sample multiplication factor is too small
        if int(self._sample_multiplication_factor * len(X)) < 1:
            return X

        # calculate the number of occurences per class
        unique, counts = np.unique(y, return_counts=True)
        occurences_per_class_dict = dict(zip(unique, counts))

        # set the amount of samples per class we want to have; includes original samples
        sampling_strategy = {}
        for class_name in occurences_per_class_dict:
            sampling_strategy[class_name] = int(
                (self._sample_multiplication_factor + 1) * occurences_per_class_dict[class_name]
            )

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self._random_state,
            k_neighbors=self._k_neighbors,
            n_jobs=self._n_jobs
        )

        X_resampled, y_resampled = smote.fit_resample(X, y)

        return X_resampled


class UnlabeledSMOTETransformer(DummyTransformer):
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
        No labels available so we view the entire set as belonging to one class
        and use SMOTE on the entire dataset. Because SMOTE requires there to be
        at least two classes, we add a dummy label vector and one dummy data
        entry. After we run SMOTE, we remove the dummy entry. Returns original
        and generated data.
        """
        # just return the original dataset if the sample multiplication factor is too small
        if int(self._sample_multiplication_factor * len(X)) < 1:
            return X

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

        X_dummy_resampled, y_dummy_resampled = smote.fit_resample(X_dummy, y_dummy)

        # remove the dummy entry
        X_resampled = X_dummy_resampled.iloc[1:, :].reset_index()

        return X_resampled


class ProportionalRACOGTransformer(DummyTransformer):
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
        Afterwards join the resampled subsets. Returns original and generated
        data.
        """
        # just return the original dataset if the sample multiplication factor is too small
        if int(self._sample_multiplication_factor * len(X)) < 1:
            return X

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

        # join original and generated data into one dataframe
        original_and_resampled_dataset = pd.concat([original_dataset, resampled_dataset], ignore_index=True)
        original_and_resampled_dataset = original_and_resampled_dataset.drop(columns=[target_column_title])

        # restore column titles if available
        if hasattr(X, 'columns'):
            original_and_resampled_dataset.columns = original_column_titles

        return original_and_resampled_dataset


class UnlabeledRACOGTransformer(DummyTransformer):
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
        to the same class. Returns original and generated data.
        """
        # just return the original dataset if the sample multiplication factor is too small
        if int(self._sample_multiplication_factor * len(X)) < 1:
            return X

        # store column titles to restore them after sampling if available
        if hasattr(X, 'columns'):
            original_column_titles = X.columns

        original_dataset = pd.DataFrame(X).copy()

        # adding target column that only consits of one class
        target_column_title = str(len(original_dataset.columns))
        original_dataset[target_column_title] = np.full((len(X), 1), 1)

        resampled_dataset = self._racog.resample(
            dataset=original_dataset,
            num_instances=int(self._sample_multiplication_factor * len(original_dataset)),
            class_attr=target_column_title
        )

        # join original and generated data into one dataframe
        original_and_resampled_dataset = pd.concat([original_dataset, resampled_dataset], ignore_index=True)
        original_and_resampled_dataset = original_and_resampled_dataset.drop(columns=[target_column_title])

        # restore column titles if available
        if hasattr(X, 'columns'):
            original_and_resampled_dataset.columns = original_column_titles

        return original_and_resampled_dataset


class UnlabeledVanillaGANTransformer(DummyTransformer):
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

    learning_rate: float, default=1e-4

    betas: tuple, default=(0.5, 0.9)

    noise_dim: int, default=264

    layers_dim: int, default=128

    epochs: int, default=300,

    sample_interval: int, default=50

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

        self._gan = VanilllaGAN(model_parameters=self._model_parameters)

    def fit(self, X, y=None):
        original_dataset = pd.DataFrame(X)

        self._gan.train(
            data=original_dataset,
            train_arguments=self._train_parameters,
            num_cols=original_dataset.columns.tolist(),
            cat_cols=[]
        )

        return self

    def transform(self, X, y=None):
        original_dataset = pd.DataFrame(X)

        # resample the original dataset
        resampled_dataset = self._gan.sample(int(self._sample_multiplication_factor * len(original_dataset)))
        resampled_dataset = pd.DataFrame(resampled_dataset)

        # drop excess samples
        resampled_dataset = resampled_dataset.head(int(self._sample_multiplication_factor * len(original_dataset)))

        # restore the original column titles
        resampled_dataset.columns = original_dataset.columns

        # join original and generated data into one dataframe
        original_and_resampled_dataset = pd.concat([original_dataset, resampled_dataset], ignore_index=True)

        return original_and_resampled_dataset
