import warnings
import time

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from ctgan import CTGANSynthesizer, TVAESynthesizer
from framework.generators.sdgym.synthesizers import PrivBN, TableGAN, MedGAN
from sdv.tabular import CopulaGAN, GaussianCopula
from framework.generators.ctabgan.model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer
from framework.generators.ydata_synthetic.synthesizers.regular import CWGANGP, WGAN_GP
from framework.generators.ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from framework.generators.snsynth.pytorch.nn import DPCTGAN
from imblearn.over_sampling import SMOTE


class PrivBNGenerator(BaseEstimator):
    def __init__(self, is_classification_task=True, theta=20, max_samples=10000000):
        self.is_classification_task = is_classification_task
        self.theta = theta
        self.max_samples = max_samples

    def get_name(self):
        return 'PrivBN'

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical_columns, ordinal_columns):
        # store the encoded target column name for later use
        self._target_column_title = y.name

        # split the column titles into categorical and numerical columns
        categorical_column_titles = categorical_columns.copy()
        ordinal_columns_titles = ordinal_columns.copy()

        # for classification problems the target is categorical, for regression problems it is numerical
        if self.is_classification_task:
            categorical_column_titles.append(self._target_column_title)
        else:
            ordinal_columns_titles.append(self._target_column_title)

        # this generator is unsupervised and therefore the target is merged with the dataset
        train_data = X.copy()
        train_data[self._target_column_title] = y.copy()

        self._generator = PrivBN(theta=self.theta, max_samples=self.max_samples)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._generator.fit(
                data=train_data.to_numpy(),
                categorical_columns=categorical_column_titles,
                ordinal_columns=ordinal_columns_titles
            )

    def sample(self, n):
        sample_data = pd.DataFrame(data=self._generator.sample(n))

        # split the generated dataset into data and target
        X = sample_data.drop(columns=[self._target_column_title])
        y = sample_data[self._target_column_title]

        return X, y


class ProportionalSMOTEGenerator(BaseEstimator):
    """
    Generator that implements a SMOTE oversampling routine. SMOTE is usually
    used for balancing in imbalanced datasets. This implementation uses SMOTE
    for generating samples of each class in the same proportion as they are
    in the training set.

    Parameters
    ----------

    k_neighbors : int or object, default=5
        If int, number of nearest neighbours to be used to construct synthetic
        samples. If object, an estimator that inherits from KNeighborsMixin that
        will be used to find the k_neighbors.

    random_state : int, default=None
        Control the randomization of the algorithm.

    References
    ----------

    .. [1] https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html

    .. [2] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, “SMOTE:
           synthetic minority over-sampling technique,” Journal of artificial
           intelligence research, 321-357, 2002.

    """

    def __init__(
            self,
            is_classification_task=True,
            k_neighbors=5,
            random_state=None
    ):
        self.is_classification_task = is_classification_task
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def get_name(self):
        return 'Proportional SMOTE'

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical_columns, ordinal_columns):
        self._original_column_titles = X.columns
        self._categorical_column_titles = categorical_columns
        self._original_target_column_title = y.name

        # calculate the occurences of each class
        unique, counts = np.unique(y, return_counts=True)
        self._occurrences_per_class_dict = dict(zip(unique, counts))

        self._X = X.copy()
        self._y = y.copy()

    def sample(self, n):
        if n < 1:
            return pd.DataFrame(), pd.Series()

        # set the amount of samples per class we want to have; includes original samples
        sampling_strategy = {}
        for label in self._occurrences_per_class_dict:
            occurences = self._occurrences_per_class_dict[label]
            number_of_classes = len(self._occurrences_per_class_dict)
            number_of_training_samples = len(self._X)
            sampling_strategy[label] = int(
                round(float(occurences) + float(n) * (float(occurences) / float(number_of_training_samples)))
            )

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state
        )

        # filter user warning that fires because we do not use SMOTE simply for balancing the dataset
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            X_sampled, y_sampled = smote.fit_resample(self._X, self._y)
        X_sampled = pd.DataFrame(X_sampled)
        y_sampled = pd.Series(y_sampled)

        # remove original dataset
        X_sampled = pd.DataFrame(X_sampled.iloc[len(self._X):, :])
        X_sampled = X_sampled.reset_index(drop=True)
        y_sampled = pd.Series(y_sampled.iloc[len(self._y):])
        y_sampled = y_sampled.reset_index(drop=True)

        X_sampled.columns = self._original_column_titles
        y_sampled.name = self._original_target_column_title

        # map categorical columns to the closest category through rounding
        X_sampled[self._categorical_column_titles] = X_sampled[self._categorical_column_titles].round()

        return X_sampled, y_sampled


class SMOTEGenerator(BaseEstimator):
    """
    Generator that implements a SMOTE oversampling routine. SMOTE is usually
    used for balancing in imbalanced datasets. This implementation uses SMOTE
    for generating the same number of samples for each class. For example
    generating 500 samples in a binary classification task returns 250 samples
    for each class.

    Parameters
    ----------

    k_neighbors : int or object, default=5
        If int, number of nearest neighbours to be used to construct synthetic
        samples. If object, an estimator that inherits from KNeighborsMixin that
        will be used to find the k_neighbors.

    random_state : int, default=None
        Control the randomization of the algorithm.

    References
    ----------

    .. [1] https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html

    .. [2] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, “SMOTE:
           synthetic minority over-sampling technique,” Journal of artificial
           intelligence research, 321-357, 2002.

    """

    def __init__(
            self,
            is_classification_task=True,
            k_neighbors=5,
            random_state=None
    ):
        self.is_classification_task = is_classification_task
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def get_name(self):
        return 'SMOTE'

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical_columns, ordinal_columns):
        self._original_column_titles = X.columns
        self._categorical_column_titles = categorical_columns
        self._original_target_column_title = y.name

        # calculate the occurences of each class
        unique, counts = np.unique(y, return_counts=True)
        self._occurrences_per_class_dict = dict(zip(unique, counts))

        self._X = X.copy()
        self._y = y.copy()

    def sample(self, n):
        if n < 1:
            return pd.DataFrame(), pd.Series()

        # set the amount of samples per class we want to have; includes original samples
        sampling_strategy = {}
        for label in self._occurrences_per_class_dict:
            occurences = self._occurrences_per_class_dict[label]
            number_of_classes = len(self._occurrences_per_class_dict)
            sampling_strategy[label] = int(
                round(float(occurences) + float(n) * (1.0 / float(number_of_classes)))
            )

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state
        )

        # filter user warning that fires because we do not use SMOTE simply for balancing the dataset
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            X_sampled, y_sampled = smote.fit_resample(self._X, self._y)
        X_sampled = pd.DataFrame(X_sampled)
        y_sampled = pd.Series(y_sampled)

        # remove original dataset
        X_sampled = pd.DataFrame(X_sampled.iloc[len(self._X):, :])
        X_sampled = X_sampled.reset_index(drop=True)
        y_sampled = pd.Series(y_sampled.iloc[len(self._y):])
        y_sampled = y_sampled.reset_index(drop=True)

        X_sampled.columns = self._original_column_titles
        y_sampled.name = self._original_target_column_title

        # map categorical columns to the closest category through rounding
        X_sampled[self._categorical_column_titles] = X_sampled[self._categorical_column_titles].round()

        return X_sampled, y_sampled


class GaussianCopulaGenerator(BaseEstimator):
    def __init__(self,
                 is_classification_task=True,
                 default_distribution='parametric',
                 rounding='auto',
                 min_value='auto',
                 max_value='auto'
                 ):
        self.is_classification_task = is_classification_task
        self.default_distribution = default_distribution
        self.rounding = rounding
        self.min_value = min_value
        self.max_value = max_value

    def get_name(self):
        return 'GaussianCopula'

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical_columns, ordinal_columns):
        # store the encoded target column name for later use
        self._target_column_title = y.name

        # split the column titles into categorical and numerical columns
        categorical_column_titles = categorical_columns.copy()
        ordinal_column_titles = ordinal_columns.copy()

        # for classification problems the target is categorical, for regression problems it is numerical
        if self.is_classification_task:
            categorical_column_titles.append(self._target_column_title)
        else:
            ordinal_column_titles.append(self._target_column_title)

        # the categorical columns need to be submitted in this format
        field_types = {}
        for column in categorical_column_titles:
            field_types[str(column)] = {'type': 'categorical'}

        self._generator = GaussianCopula(
            field_types=field_types,
            default_distribution=self.default_distribution,
            rounding=self.rounding,
            min_value=self.min_value,
            max_value=self.max_value)

        # this generator is unsupervised and therefore the target is merged with the dataset
        train_data = X.copy()
        train_data[self._target_column_title] = y.copy()

        # categorical columns need to be in string form
        train_data[categorical_column_titles] = train_data[categorical_column_titles].astype(str)

        # column titles have to be strings
        self._column_titles = train_data.columns
        train_data.columns = [str(col) for col in self._column_titles]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._generator.fit(data=train_data)

        self._y_dtype = y.dtype

    def sample(self, n):
        sample_data = self._generator.sample(n)

        # reset the column titles
        sample_data.columns = self._column_titles

        # split the generated dataset into data and target
        X = sample_data.drop(columns=[self._target_column_title])
        y = sample_data[self._target_column_title]

        y = y.astype(self._y_dtype)

        return X, y


class TableGANGenerator(BaseEstimator):
    def __init__(self,
                 is_classification_task=True,
                 random_dim=100,
                 num_channels=64,
                 l2scale=1e-5,
                 batch_size=500,
                 epochs=300):
        self.is_classification_task = is_classification_task
        self.random_dim = random_dim
        self.num_channels = num_channels
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs

    def get_name(self):
        return 'TableGAN'

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical_columns, ordinal_columns):
        # store the encoded target column name for later use
        self._target_column_title = y.name

        # split the column titles into categorical and numerical columns
        categorical_column_titles = categorical_columns.copy()
        ordinal_column_titles = ordinal_columns.copy()

        # for classification problems the target is categorical, for regression problems it is numerical
        if self.is_classification_task:
            categorical_column_titles.append(self._target_column_title)
        else:
            ordinal_column_titles.append(self._target_column_title)

        # this generator is unsupervised and therefore the target is merged with the dataset
        train_data = X.copy()
        train_data[self._target_column_title] = y.copy()

        self._generator = TableGAN(
            random_dim=self.random_dim,
            num_channels=self.num_channels,
            l2scale=self.l2scale,
            batch_size=self.batch_size,
            epochs=self.epochs
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._generator.fit(
                data=train_data.to_numpy(),
                categorical_columns=categorical_column_titles,
                ordinal_columns=ordinal_column_titles
            )

    def sample(self, n):
        sample_data = pd.DataFrame(data=self._generator.sample(n))

        # split the generated dataset into data and target
        X = sample_data.drop(columns=[self._target_column_title])
        y = sample_data[self._target_column_title]

        return X, y


class CTGANGenerator(BaseEstimator):
    def __init__(self, is_classification_task=True, embedding_dim=128, generator_dim=(256, 256),
                 discriminator_dim=(256, 256), generator_lr=2e-4, generator_decay=1e-6,
                 discriminator_lr=2e-4, discriminator_decay=1e-6, batch_size=500, discriminator_steps=5,
                 log_frequency=True, verbose=True, epochs=300, pac=10, cuda=True):
        self.is_classification_task = is_classification_task
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.generator_lr = generator_lr
        self.generator_decay = generator_decay
        self.discriminator_lr = discriminator_lr
        self.discriminator_decay = discriminator_decay
        self.batch_size = batch_size
        self.discriminator_steps = discriminator_steps
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.epochs = epochs
        self.pac = pac
        self.cuda = cuda

    def get_name(self):
        return 'CTGAN'

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical_columns, ordinal_columns):
        self._target_name = y.name

        # join the dataset and the target
        train_data = X.copy()
        train_data[self._target_name] = y.copy()

        # for classification problems the target is categorical, for regression problems it is numerical
        cat_cols = categorical_columns.copy()
        if self.is_classification_task:
            cat_cols.append(self._target_name)

        self._generator = CTGANSynthesizer(embedding_dim=self.embedding_dim, generator_dim=self.generator_dim,
                                           discriminator_dim=self.discriminator_dim, generator_lr=self.generator_lr,
                                           generator_decay=self.generator_decay, discriminator_lr=self.discriminator_lr,
                                           discriminator_decay=self.discriminator_decay, batch_size=self.batch_size,
                                           discriminator_steps=self.discriminator_steps,
                                           log_frequency=self.log_frequency, verbose=self.verbose, epochs=self.epochs,
                                           pac=self.pac, cuda=self.cuda)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._generator.fit(train_data=train_data.to_numpy(), discrete_columns=cat_cols)

    def sample(self, n):
        sample_data = pd.DataFrame(data=self._generator.sample(n))

        X = sample_data.drop(columns=[self._target_name])
        y = sample_data[self._target_name]

        return X, y


class CopulaGANGenerator(BaseEstimator):
    def __init__(self,
                 is_classification_task=True,
                 embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=5,
                 log_frequency=True, verbose=True, epochs=300, cuda=True, rounding='auto',
                 min_value='auto', max_value='auto'):
        self.is_classification_task = is_classification_task
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.generator_lr = generator_lr
        self.generator_decay = generator_decay
        self.discriminator_lr = discriminator_lr
        self.discriminator_decay = discriminator_decay
        self.batch_size = batch_size
        self.discriminator_steps = discriminator_steps
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.epochs = epochs
        self.cuda = cuda
        self.rounding = rounding
        self.min_value = min_value
        self.max_value = max_value

    def get_name(self):
        return 'CopulaGAN'

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical_columns, ordinal_columns):
        # store the encoded target column name for later use
        self._target_column_title = y.name

        # split the column titles into categorical and numerical columns
        categorical_column_titles = categorical_columns.copy()
        ordinal_column_titles = ordinal_columns.copy()

        # for classification problems the target is categorical, for regression problems it is numerical
        if self.is_classification_task:
            categorical_column_titles.append(self._target_column_title)
        else:
            ordinal_column_titles.append(self._target_column_title)

        # the categorical columns need to be submitted in this format
        field_types = {}
        for column in categorical_column_titles:
            field_types[str(column)] = {'type': 'categorical'}

        self._generator = CopulaGAN(
            field_types=field_types,
            embedding_dim=self.embedding_dim,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            generator_lr=self.generator_lr,
            generator_decay=self.generator_decay,
            discriminator_lr=self.discriminator_lr,
            discriminator_decay=self.discriminator_decay,
            batch_size=self.batch_size,
            discriminator_steps=self.discriminator_steps,
            log_frequency=self.log_frequency,
            verbose=self.verbose,
            epochs=self.epochs,
            cuda=self.cuda,
            rounding=self.rounding,
            min_value=self.min_value,
            max_value=self.max_value)

        # this generator is unsupervised and therefore the target is merged with the dataset
        train_data = X.copy()
        train_data[self._target_column_title] = y.copy()

        # categorical columns need to be in string form
        train_data[categorical_column_titles] = train_data[categorical_column_titles].astype(str)

        # column titles have to be strings
        self._column_titles = train_data.columns
        train_data.columns = [str(col) for col in self._column_titles]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._generator.fit(data=train_data)

        self._y_dtype = y.dtype

    def sample(self, n):
        sample_data = self._generator.sample(n)

        # reset the column titles
        sample_data.columns = self._column_titles

        # split the generated dataset into data and target
        X = sample_data.drop(columns=[self._target_column_title])
        y = sample_data[self._target_column_title]

        y = y.astype(self._y_dtype)

        return X, y


class TVAEGenerator(BaseEstimator):
    def __init__(self, is_classification_task=True, embedding_dim=128, compress_dims=(128, 128),
                 decompress_dims=(128, 128), l2scale=1e-5, batch_size=500, epochs=300, loss_factor=2, cuda=True):
        self.is_classification_task = is_classification_task
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_factor = loss_factor
        self.cuda = cuda

    def get_name(self):
        return 'TVAE'

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical_columns, ordinal_columns):
        self._target_name = y.name

        # join the dataset and the target
        train_data = X.copy()
        train_data[self._target_name] = y.copy()

        # for classification problems the target is categorical, for regression problems it is numerical
        cat_cols = categorical_columns.copy()
        if self.is_classification_task:
            cat_cols.append(self._target_name)

        self._generator = TVAESynthesizer(
            embedding_dim=self.embedding_dim,
            compress_dims=self.compress_dims,
            decompress_dims=self.decompress_dims,
            l2scale=self.l2scale,
            batch_size=self.batch_size,
            epochs=self.epochs,
            loss_factor=self.loss_factor,
            cuda=self.cuda
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._generator.fit(train_data=train_data.to_numpy(), discrete_columns=cat_cols)

    def sample(self, n):
        sample_data = pd.DataFrame(data=self._generator.sample(n))

        X = sample_data.drop(columns=[self._target_name])
        y = sample_data[self._target_name]

        return X, y


class WGANGPGenerator(BaseEstimator):
    def __init__(self,
                 is_classification_task=True,
                 noise_dim=264,
                 layers_dim=128,
                 batch_size=128,
                 beta_1=0.5,
                 beta_2=0.9,
                 log_step=100,
                 epochs=300,
                 learning_rate=1e-4,
                 n_critic=1,
                 models_dir='./cache',
                 ):
        self.is_classification_task = is_classification_task
        self.log_step = log_step
        self.epochs = epochs
        self.models_dir = models_dir
        self.n_critic = n_critic
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.noise_dim = noise_dim
        self.layers_dim = layers_dim

    def get_name(self):
        return 'WGAN-GP'

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical_columns, ordinal_columns):
        # store the encoded target column name for later use
        self._target_column_title = y.name

        # split the column titles into categorical and numerical columns
        categorical_column_titles = categorical_columns.copy()
        ordinal_column_titles = ordinal_columns.copy()

        # for classification problems the target is categorical, for regression problems it is numerical
        if self.is_classification_task:
            categorical_column_titles.append(self._target_column_title)
        else:
            ordinal_column_titles.append(self._target_column_title)

        # this generator is unsupervised and therefore the target is merged with the dataset
        train_data = X.copy()
        train_data[self._target_column_title] = y.copy()

        self._column_titles = train_data.columns

        # column titles need to be in string form
        train_data.columns = [str(col) for col in train_data.columns]
        categorical_column_titles = [str(col) for col in categorical_column_titles]
        ordinal_column_titles = [str(col) for col in ordinal_column_titles]

        # calculate the ratios of each class
        unique, counts = np.unique(y, return_counts=True)
        self._labels = unique

        model_parameters = ModelParameters(
            batch_size=self.batch_size,
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2),
            noise_dim=self.noise_dim,
            layers_dim=self.layers_dim
        )

        self._generator = WGAN_GP(
            model_parameters=model_parameters,
            n_critic=self.n_critic
        )

        train_parameters = TrainParameters(
            epochs=self.epochs,
            cache_prefix=str(time.time_ns()) + '_',
            sample_interval=self.log_step
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._generator.train(
                data=train_data,
                train_arguments=train_parameters,
                num_cols=ordinal_column_titles,
                cat_cols=categorical_column_titles
            )

    def sample(self, n):
        if n < 1:
            return pd.DataFrame(), pd.Series()

        sample_data = self._generator.sample(n_samples=n)
        sample_data.columns = self._column_titles

        # split the generated dataset into data and target
        X = sample_data.drop(columns=[self._target_column_title])
        y = sample_data[self._target_column_title]

        return X, y

class ProportionalCWGANGPGenerator(BaseEstimator):
    def __init__(self,
                 is_classification_task=True,
                 noise_dim=264,
                 layers_dim=128,
                 batch_size=128,
                 beta_1=0.5,
                 beta_2=0.9,
                 log_step=100,
                 epochs=300,
                 learning_rate=1e-4,
                 n_critic=1,
                 models_dir='./cache',
                 ):
        self.is_classification_task = is_classification_task
        self.log_step = log_step
        self.epochs = epochs
        self.models_dir = models_dir
        self.n_critic = n_critic
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.noise_dim = noise_dim
        self.layers_dim = layers_dim

    def get_name(self):
        return 'Proportional CWGAN-GP'

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical_columns, ordinal_columns):
        # store the encoded target column name for later use
        self._target_column_title = y.name

        # split the column titles into categorical and numerical columns
        categorical_column_titles = categorical_columns.copy()
        ordinal_column_titles = ordinal_columns.copy()

        # this generator is unsupervised and therefore the target is merged with the dataset
        train_data = X.copy()
        train_data[self._target_column_title] = y.copy()

        self._column_titles = train_data.columns

        # column titles need to be in string form
        train_data.columns = [str(col) for col in train_data.columns]
        categorical_column_titles = [str(col) for col in categorical_column_titles]
        ordinal_column_titles = [str(col) for col in ordinal_column_titles]

        # calculate the ratios of each class
        unique, counts = np.unique(y, return_counts=True)
        self._label_counts = dict(zip(unique, counts))
        self._train_size = len(y)
        self._labels = unique

        model_parameters = ModelParameters(
            batch_size=self.batch_size,
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2),
            noise_dim=self.noise_dim,
            layers_dim=self.layers_dim
        )

        self._generator = CWGANGP(
            model_parameters=model_parameters,
            num_classes=len(self._labels),
            n_critic=self.n_critic
        )

        train_parameters = TrainParameters(
            epochs=self.epochs,
            cache_prefix=str(time.time_ns()) + '_',
            sample_interval=self.log_step,
            label_dim=-1,
            labels=unique
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._generator.train(
                data=train_data,
                label_col=str(self._target_column_title),
                train_arguments=train_parameters,
                num_cols=ordinal_column_titles,
                cat_cols=categorical_column_titles
            )

    def sample(self, n):
        if n < 1:
            return pd.DataFrame(), pd.Series()

        subsets = []
        for label in self._labels:
            n_samples_for_label = int(
                round(float(n) * (float(self._label_counts[label]) / float(self._train_size)))
            )
            sample_data = self._generator.sample(condition=np.array([label]), n_samples=n_samples_for_label)
            sample_data = sample_data[:n_samples_for_label]
            subsets.append(sample_data)
        sample_data = pd.concat(subsets, axis=0).reset_index(drop=True)
        sample_data.columns = self._column_titles

        # split the generated dataset into data and target
        X = sample_data.drop(columns=[self._target_column_title])
        y = sample_data[self._target_column_title]

        return X, y


class CWGANGPGenerator(BaseEstimator):
    def __init__(self,
                 is_classification_task=True,
                 noise_dim=264,
                 layers_dim=128,
                 batch_size=128,
                 beta_1=0.5,
                 beta_2=0.9,
                 log_step=100,
                 epochs=300,
                 learning_rate=1e-4,
                 n_critic=1,
                 models_dir='./cache',
                 ):
        self.is_classification_task = is_classification_task
        self.log_step = log_step
        self.epochs = epochs
        self.models_dir = models_dir
        self.n_critic = n_critic
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.noise_dim = noise_dim
        self.layers_dim = layers_dim

    def get_name(self):
        return 'CWGAN-GP'

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical_columns, ordinal_columns):
        # store the encoded target column name for later use
        self._target_column_title = y.name

        # split the column titles into categorical and numerical columns
        categorical_column_titles = categorical_columns.copy()
        ordinal_column_titles = ordinal_columns.copy()

        # this generator is unsupervised and therefore the target is merged with the dataset
        train_data = X.copy()
        train_data[self._target_column_title] = y.copy()

        self._column_titles = train_data.columns

        # column titles need to be in string form
        train_data.columns = [str(col) for col in train_data.columns]
        categorical_column_titles = [str(col) for col in categorical_column_titles]
        ordinal_column_titles = [str(col) for col in ordinal_column_titles]

        # calculate the ratios of each class
        unique, counts = np.unique(y, return_counts=True)
        self._labels = unique

        model_parameters = ModelParameters(
            batch_size=self.batch_size,
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2),
            noise_dim=self.noise_dim,
            layers_dim=self.layers_dim
        )

        self._generator = CWGANGP(
            model_parameters=model_parameters,
            num_classes=len(self._labels),
            n_critic=self.n_critic
        )

        train_parameters = TrainParameters(
            epochs=self.epochs,
            cache_prefix=str(time.time_ns()) + '_',
            sample_interval=self.log_step,
            label_dim=-1,
            labels=unique
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._generator.train(
                data=train_data,
                label_col=str(self._target_column_title),
                train_arguments=train_parameters,
                num_cols=ordinal_column_titles,
                cat_cols=categorical_column_titles
            )

    def sample(self, n):
        if n < 1:
            return pd.DataFrame(), pd.Series()

        subsets = []
        for label in self._labels:
            n_samples_for_label = int(
                round(float(n) * (1.0 / float(len(self._labels))))
            )
            sample_data = self._generator.sample(condition=np.array([label]), n_samples=n_samples_for_label)
            sample_data = sample_data[:n_samples_for_label]
            subsets.append(sample_data)
        sample_data = pd.concat(subsets, axis=0).reset_index(drop=True)
        sample_data.columns = self._column_titles

        # split the generated dataset into data and target
        X = sample_data.drop(columns=[self._target_column_title])
        y = sample_data[self._target_column_title]

        return X, y


class MedGANGenerator(BaseEstimator):
    def __init__(self,
                 is_classification_task=True,
                 embedding_dim=128,
                 random_dim=128,
                 generator_dims=(128, 128),  # 128 -> 128 -> 128
                 discriminator_dims=(256, 128, 1),  # datadim * 2 -> 256 -> 128 -> 1
                 compress_dims=(),  # datadim -> embedding_dim
                 decompress_dims=(),  # embedding_dim -> datadim
                 bn_decay=0.99,
                 l2scale=0.001,
                 pretrain_epoch=200,
                 batch_size=1000,
                 epochs=2000):
        self.is_classification_task = is_classification_task
        self.embedding_dim = embedding_dim
        self.random_dim = random_dim
        self.generator_dims = generator_dims
        self.discriminator_dims = discriminator_dims
        self.compress_dims = compress_dims  # datadim -> embedding_dim
        self.decompress_dims = decompress_dims  # embedding_dim -> datadim
        self.bn_decay = bn_decay
        self.l2scale = l2scale
        self.pretrain_epoch = pretrain_epoch
        self.batch_size = batch_size
        self.epochs = epochs

    def get_name(self):
        return 'MedGAN'

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical_columns, ordinal_columns):
        # store the encoded target column name for later use
        self._target_column_title = y.name

        # split the column titles into categorical and numerical columns
        categorical_column_titles = categorical_columns.copy()
        ordinal_column_titles = ordinal_columns.copy()

        # for classification problems the target is categorical, for regression problems it is numerical
        if self.is_classification_task:
            categorical_column_titles.append(self._target_column_title)
        else:
            ordinal_column_titles.append(self._target_column_title)

        # this generator is unsupervised and therefore the target is merged with the dataset
        train_data = X.copy()
        train_data[self._target_column_title] = y.copy()

        self._generator = MedGAN(
            embedding_dim=self.embedding_dim,
            random_dim=self.random_dim,
            generator_dims=self.generator_dims,
            discriminator_dims=self.discriminator_dims,
            compress_dims=self.compress_dims,  # datadim -> embedding_dim
            decompress_dims=self.decompress_dims,  # embedding_dim -> datadim
            bn_decay=self.bn_decay,
            l2scale=self.l2scale,
            pretrain_epoch=self.pretrain_epoch,
            batch_size=self.batch_size,
            epochs=self.epochs
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._generator.fit(
                data=train_data.to_numpy(),
                categorical_columns=categorical_column_titles,
                ordinal_columns=ordinal_column_titles
            )

    def sample(self, n):
        sample_data = pd.DataFrame(data=self._generator.sample(n))

        # split the generated dataset into data and target
        X = sample_data.drop(columns=[self._target_column_title])
        y = sample_data[self._target_column_title]

        return X, y


class DPCTGANGenerator(BaseEstimator):
    def __init__(self,
                 is_classification_task=True,
                 embedding_dim=128,
                 generator_dim=(256, 256),
                 discriminator_dim=(256, 256),
                 generator_lr=2e-4,
                 generator_decay=1e-6,
                 discriminator_lr=2e-4,
                 discriminator_decay=1e-6,
                 batch_size=500,
                 discriminator_steps=5,
                 log_frequency=False,
                 verbose=True,
                 epochs=300,
                 pac=1,
                 cuda=True,
                 disabled_dp=False,
                 delta=None,
                 sigma=5,
                 max_per_sample_grad_norm=1.0,
                 epsilon=1,
                 preprocessor_eps=1,
                 loss="cross_entropy",
                 category_epsilon_pct=0.1
                 ):
        self.is_classification_task = is_classification_task
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.generator_lr = generator_lr
        self.generator_decay = generator_decay
        self.discriminator_lr = discriminator_lr
        self.discriminator_decay = discriminator_decay
        self.batch_size = batch_size
        self.discriminator_steps = discriminator_steps
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.epochs = epochs
        self.pac = pac
        self.cuda = cuda
        self.disabled_dp = disabled_dp
        self.delta = delta
        self.sigma = sigma
        self.max_per_sample_grad_norm = max_per_sample_grad_norm
        self.epsilon = epsilon
        self.preprocessor_eps = preprocessor_eps
        self.loss = loss
        self.category_epsilon_pct = category_epsilon_pct

    def get_name(self):
        return 'DP-CTGAN'

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical_columns, ordinal_columns):
        # store the encoded target column name for later use
        self._target_column_title = y.name

        # this generator is unsupervised and therefore the target is merged with the dataset
        train_data = X.copy()
        train_data[self._target_column_title] = y.copy()

        self._original_column_titles = train_data.columns

        # split the column titles into categorical and numerical columns
        categorical_column_titles = categorical_columns.copy()
        ordinal_column_titles = ordinal_columns.copy()

        # for classification problems the target is categorical, for regression problems it is numerical
        if self.is_classification_task:
            categorical_column_titles.append(self._target_column_title)
        else:
            ordinal_column_titles.append(self._target_column_title)

        train_data[categorical_column_titles] = train_data[categorical_column_titles].astype(int)
        train_data.columns = [str(col) for col in train_data.columns]
        categorical_column_titles = [str(col) for col in categorical_column_titles]
        ordinal_column_titles = [str(col) for col in ordinal_column_titles]

        self._generator = DPCTGAN(
            embedding_dim=self.embedding_dim,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            generator_lr=self.generator_lr,
            generator_decay=self.generator_decay,
            discriminator_lr=self.discriminator_lr,
            discriminator_decay=self.discriminator_decay,
            batch_size=self.batch_size,
            discriminator_steps=self.discriminator_steps,
            log_frequency=self.log_frequency,
            verbose=self.verbose,
            epochs=self.epochs,
            pac=self.pac,
            cuda=self.cuda,
            disabled_dp=self.disabled_dp,
            delta=self.delta,
            sigma=self.sigma,
            max_per_sample_grad_norm=self.max_per_sample_grad_norm,
            epsilon=self.epsilon,
            preprocessor_eps=self.preprocessor_eps,
            loss=self.loss,
            category_epsilon_pct=self.category_epsilon_pct
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._generator.train(
                data=train_data,
                categorical_columns=categorical_column_titles,
                ordinal_columns=ordinal_column_titles
            )

    def sample(self, n):
        sample_data = self._generator.generate(n)

        sample_data.columns = self._original_column_titles

        # split the generated dataset into data and target
        X = sample_data.drop(columns=[self._target_column_title])
        y = sample_data[self._target_column_title]

        return X, y


class CTABGANGenerator(BaseEstimator):
    def __init__(self,
                 is_classification_task=True,
                 class_dim=(256, 256, 256, 256),
                 random_dim=100,
                 num_channels=64,
                 l2scale=1e-5,
                 batch_size=500,
                 epochs=1):
        self.is_classification_task = is_classification_task
        self.class_dim = class_dim
        self.random_dim = random_dim
        self.num_channels = num_channels
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs

    def get_name(self):
        return 'CTABGAN'

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical_columns, ordinal_columns):
        # store the encoded target column name for later use
        self._target_column_title = y.name

        # this generator is unsupervised and therefore the target is merged with the dataset
        train_data = X.copy()
        train_data[self._target_column_title] = y.copy()

        # split the column titles into categorical and numerical columns
        categorical_column_titles = categorical_columns.copy()

        # for classification problems the target is categorical, for regression problems it is numerical
        if self.is_classification_task:
            categorical_column_titles.append(self._target_column_title)
            task_type = {"Classification": self._target_column_title}
        else:
            task_type = None

        self._generator = CTABGANSynthesizer(
            class_dim=self.class_dim,
            random_dim=self.random_dim,
            num_channels=self.num_channels,
            l2scale=self.l2scale,
            batch_size=self.batch_size,
            epochs=self.epochs
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._generator.fit(
                train_data=train_data,
                categorical=categorical_column_titles,
                type=task_type
            )

    def sample(self, n):
        sample_data = pd.DataFrame(data=self._generator.sample(n))

        # split the generated dataset into data and target
        X = sample_data.drop(columns=[self._target_column_title])
        y = sample_data[self._target_column_title]

        return X, y


"""Transformer wrappers for SMOTE, GAN and Gibbs sampling methods for use as
   samplers in sklearn.pipeline.Pipeline"""

# Authors: Thomas Frank <thomas-frank01@gmx.de>
# License: MIT

# import numpy as np
# import pandas as pd
# from sklearn.base import BaseEstimator, TransformerMixin
#
# from imblearn.over_sampling import SMOTE
# from ydata_synthetic.synthesizers.regular import CGAN
# from ydata_synthetic.synthesizers.regular import DRAGAN
# from ydata_synthetic.synthesizers.regular import VanilllaGAN
# from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
#
# import os
# import warnings
# import tensorflow as tf
# import random
# import time
# from uuid import uuid4



# class ProportionalSampler(BaseEstimator, TransformerMixin):
#     """
#     Proportional sampler base class. Requires the target value in each step.
#     """
#
#     def fit(self, X, y):
#         return self
#
#     def transform(self, X, y):
#         return self
#
#     def fit_transform(self, X, y, **fit_params):
#         return self.fit(X, y, **fit_params).transform(X, y)
#
#
# class UnlabeledSampler(BaseEstimator, TransformerMixin):
#     """
#     Unlabeled sampler base class. Doesn't require the target.
#     """
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X, y=None):
#         return self
#
#     def fit_transform(self, X, y=None, **fit_params):
#         if y is None:
#             return self.fit(X, **fit_params).transform(X)
#         else:
#             return self.fit(X, y, **fit_params).transform(X, y)
#
#
# class ProportionalSMOTESampler(ProportionalSampler):
#     """
#     Transformer that implements a proportional SMOTE sampling routine.
#     SMOTE is usually used for oversampling in imbalanced datasets, but this
#     implementation allows us to sample all classes proportionally and their
#     ratio remains intact.
#
#     Parameters
#     ----------
#
#     sample_multiplication_factor : float
#         Determines the relative amount of generated data, i.e. 0 means that no
#         data is generated and 1 means that we generate the same number of data
#         points as we already have.
#
#     only_sampled : bool, default=False
#         Determines whether the original dataset is prepended to the generated
#         samples.
#
#     random_state : int, default=None
#         Control the randomization of the algorithm.
#
#     k_neighbors : int or object, default=5
#         If int, number of nearest neighbours to be used to construct synthetic
#         samples. If object, an estimator that inherits from KNeighborsMixin that
#         will be used to find the k_neighbors.
#
#     n_jobs : int, default=None
#         Number of CPU cores used during the cross-validation loop. None means 1
#         unless in a joblib.parallel_backend context. -1 means using all
#         processors. See Glossary for more details.
#
#     References
#     ----------
#
#     .. [1] https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
#
#     .. [2] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, “SMOTE:
#            synthetic minority over-sampling technique,” Journal of artificial
#            intelligence research, 321-357, 2002.
#
#     """
#
#     def __init__(
#             self,
#             sample_multiplication_factor,
#             only_sampled=False,
#             random_state=None,
#             k_neighbors=5,
#             n_jobs=None
#     ):
#         self.sample_multiplication_factor = sample_multiplication_factor
#         self.only_sampled = only_sampled
#         self.random_state = random_state
#         self.k_neighbors = k_neighbors
#         self.n_jobs = n_jobs
#
#     def fit(self, X, y):
#         return self
#
#     def transform(self, X, y):
#         """
#         Count number of occurrences of each class and resample proportionally.
#         """
#         # return an empty dataframe and target or the original samples if the multiplication factor is too small
#         if int(self.sample_multiplication_factor * len(X)) < 1:
#             if self.only_sampled:
#                 return pd.DataFrame(columns=pd.DataFrame(X).columns), pd.Series([])
#             else:
#                 return X, y
#
#         # store column titles to restore them after sampling if available
#         if hasattr(X, 'columns'):
#             original_column_titles = X.columns
#         if hasattr(y, 'name'):
#             original_target_title = y.name
#
#         original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
#         original_target = pd.Series(y).copy().reset_index(drop=True)
#
#         # calculate the number of occurrences per class
#         unique, counts = np.unique(original_target, return_counts=True)
#         occurrences_per_class_dict = dict(zip(unique, counts))
#
#         # set the amount of samples per class we want to have; includes original samples
#         sampling_strategy = {}
#         for class_name in occurrences_per_class_dict:
#             sampling_strategy[class_name] = int(
#                 (self.sample_multiplication_factor + 1) * occurrences_per_class_dict[class_name]
#             )
#
#         smote = SMOTE(
#             sampling_strategy=sampling_strategy,
#             random_state=self.random_state,
#             k_neighbors=self.k_neighbors,
#             n_jobs=self.n_jobs
#         )
#
#         # filter user warning that fires because we do not use SMOTE simply for balancing the dataset
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')
#             resampled_dataset, resampled_target = smote.fit_resample(original_dataset, original_target)
#         resampled_dataset = pd.DataFrame(resampled_dataset)
#         resampled_target = pd.Series(resampled_target)
#
#         if self.only_sampled:
#             # remove original dataset
#             resampled_dataset = pd.DataFrame(resampled_dataset.iloc[len(X):, :])
#             resampled_dataset = resampled_dataset.reset_index(drop=True)
#             resampled_target = pd.Series(resampled_target.iloc[len(X):])
#             resampled_target = resampled_target.reset_index(drop=True)
#
#         # restore column titles if available
#         if hasattr(X, 'columns'):
#             resampled_dataset.columns = original_column_titles
#         if hasattr(y, 'name'):
#             resampled_target.name = original_target_title
#
#         return resampled_dataset, resampled_target
#
#
# class UnlabeledSMOTESampler(UnlabeledSampler):
#     """
#     Transformer that implements a SMOTE sampling routine for unlabeled data.
#     SMOTE is usually used for oversampling in imbalanced datasets, but this
#     implementation allows us to apply SMOTE on data without classes.
#
#     Parameters
#     ----------
#
#     sample_multiplication_factor : float
#         Determines the relative amount of generated data, i.e. 0 means that no
#         data is generated and 1 means that we generate the same number of data
#         points as we already have.
#
#     only_sampled : bool, default=False
#         Determines whether the original dataset is prepended to the generated
#         samples.
#
#     random_state : int, default=None
#         Control the randomization of the algorithm.
#
#     k_neighbors : int or object, default=5
#         If int, number of nearest neighbours to be used to construct synthetic
#         samples. If object, an estimator that inherits from KNeighborsMixin that
#         will be used to find the k_neighbors.
#
#     n_jobs : int, default=None
#         Number of CPU cores used during the cross-validation loop. None means 1
#         unless in a joblib.parallel_backend context. -1 means using all
#         processors. See Glossary for more details.
#
#     References
#     ----------
#
#     .. [1] https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
#
#     .. [2] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, “SMOTE:
#            synthetic minority over-sampling technique,” Journal of artificial
#            intelligence research, 321-357, 2002.
#
#     """
#
#     def __init__(
#             self,
#             sample_multiplication_factor,
#             only_sampled=False,
#             random_state=None,
#             k_neighbors=5,
#             n_jobs=None
#     ):
#         self.sample_multiplication_factor = sample_multiplication_factor
#         self.only_sampled = only_sampled
#         self.random_state = random_state
#         self.k_neighbors = k_neighbors
#         self.n_jobs = n_jobs
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X, y=None):
#         """
#         No labels available, so we view the entire set as belonging to one class
#         and use SMOTE on the entire dataset. Because SMOTE requires there to be
#         at least two classes, we add a dummy label vector and one dummy data
#         entry. After we run SMOTE, we remove the dummy entry.
#         """
#         # return an empty dataframe and target or the original samples if the multiplication factor is too small
#         if int(self.sample_multiplication_factor * len(X)) < 1:
#             if self.only_sampled:
#                 return pd.DataFrame(columns=pd.DataFrame(X).columns), pd.Series([])
#             else:
#                 return X, y
#
#         # store column titles to restore them after sampling if available
#         if hasattr(X, 'columns'):
#             original_column_titles = X.columns
#
#         original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
#
#         # create a dummy target vector [ 0 1 ... 1 ]
#         y_dummy = np.append(0, np.full((len(original_dataset), 1), 1))
#
#         # set sampling strategy to ignore the dummy class and resample the rest
#         sampling_strategy = {
#             0: 1,
#             1: int((self.sample_multiplication_factor + 1) * len(original_dataset))
#         }
#
#         # insert a dummy entry on index 0
#         dummy_dataset = pd.DataFrame(original_dataset).copy()
#         dummy_dataset.loc[-1] = [0] * len(dummy_dataset.columns)
#         dummy_dataset.index = dummy_dataset.index + 1
#         dummy_dataset.sort_index(inplace=True)
#
#         smote = SMOTE(
#             sampling_strategy=sampling_strategy,
#             random_state=self.random_state,
#             k_neighbors=self.k_neighbors,
#             n_jobs=self.n_jobs
#         )
#
#         # filter user warning that fires because we do not use SMOTE simply for balancing the dataset
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')
#             resampled_dummy_dataset, _ = smote.fit_resample(dummy_dataset, y_dummy)
#         resampled_dummy_dataset = pd.DataFrame(resampled_dummy_dataset)
#
#         # remove the dummy entry
#         resampled_dataset = resampled_dummy_dataset.iloc[1:, :].reset_index(drop=True)
#
#         if self.only_sampled:
#             # remove original dataset
#             resampled_dataset = pd.DataFrame(resampled_dataset.iloc[len(X):, :])
#             resampled_dataset = resampled_dataset.reset_index(drop=True)
#             resampled_target = None
#         else:
#             resampled_target = pd.concat([
#                 pd.Series(y).copy().reset_index(drop=True),
#                 pd.Series(
#                     np.full(int(self.sample_multiplication_factor * len(X)), np.nan)
#                 ).copy().reset_index(drop=True)
#             ], ignore_index=True)
#             resampled_target = pd.Series(resampled_target).reset_index(drop=True)
#
#         # restore column titles if available
#         if hasattr(X, 'columns'):
#             resampled_dataset.columns = original_column_titles
#
#         return resampled_dataset, resampled_target
#
#
# class ProportionalVanillaGANSampler(ProportionalSampler):
#     """
#     Transformer that implements a proportional sampling routine using a vanilla
#     GAN implementation. We train and sample from a different vanilla GAN model
#     for each class.
#
#     Parameters
#     ----------
#
#     sample_multiplication_factor : float
#         Determines the relative amount of generated data, i.e. 0 means that no
#         data is generated and 1 means that we generate the same number of data
#         points as we already have.
#
#     only_sampled : bool, default=False
#         Determines whether the original dataset is prepended to the generated
#         samples.
#
#     batch_size: int, default=128
#         Number of samples used per training step.
#
#     learning_rate: float, default=1e-4
#         The learning rate for each training step.
#
#     betas: tuple, default=(0.5, 0.9)
#         Initial decay rates of Adam when estimating the first and second
#         moments of the gradient.
#
#     noise_dim: int, default=264
#         The length of the noise vector per example.
#
#     layers_dim: int, default=128
#         The dimension of the networks layers.
#
#     epochs: int, default=300
#         Total number of training steps.
#
#     sample_interval: int, default=50
#         The interval between samples.
#
#     random_state : int, default=None
#         Control the randomization of the algorithm.
#
#     References
#     ----------
#
#     .. [1] https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/regular/vanillagan/model.py
#
#     """
#
#     def __init__(
#             self,
#             sample_multiplication_factor,
#             only_sampled=False,
#             batch_size=128,
#             learning_rate=1e-4,
#             betas=(0.5, 0.9),
#             noise_dim=264,
#             layers_dim=128,
#             epochs=300,
#             sample_interval=50,
#             random_state=None
#     ):
#         self.sample_multiplication_factor = sample_multiplication_factor
#         self.only_sampled = only_sampled
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.betas = betas
#         self.noise_dim = noise_dim
#         self.layers_dim = layers_dim
#         self.epochs = epochs
#         self.sample_interval = sample_interval
#         self.random_state = random_state
#
#     def fit(self, X, y):
#         _set_global_random_state(self.random_state)
#
#         if not hasattr(self, '_cache_prefixes'):
#             self._cache_prefixes = []
#
#         # reset vanilla gan models
#         self._vanilla_gan = {}
#
#         original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
#         y = pd.Series(y).copy().reset_index(drop=True)
#
#         # change the column titles for easier use
#         original_dataset.columns = _convert_list_to_string_list(range(0, len(original_dataset.columns)))
#         num_cols = original_dataset.columns.copy().tolist()
#
#         # calculate the number of occurrences per class
#         unique, counts = np.unique(y, return_counts=True)
#         occurrences_per_class_dict = dict(zip(unique, counts))
#
#         for class_name in occurrences_per_class_dict:
#             self._vanilla_gan[class_name] = VanilllaGAN(
#                 model_parameters=ModelParameters(
#                     batch_size=self.batch_size,
#                     lr=self.learning_rate,
#                     betas=self.betas,
#                     noise_dim=self.noise_dim,
#                     layers_dim=self.layers_dim
#                 )
#             )
#
#             original_subset = original_dataset.iloc[[x for x in range(0, len(y)) if y[x] == class_name]]
#
#             cache_prefix = str(int(time.time() * 1000)) + '_' + str(uuid4())
#             self._cache_prefixes.append(cache_prefix)
#
#             self._vanilla_gan[class_name].train(
#                 data=original_subset,
#                 train_arguments=TrainParameters(
#                     cache_prefix=cache_prefix,
#                     epochs=self.epochs,
#                     sample_interval=self.sample_interval
#                 ),
#                 num_cols=num_cols,
#                 cat_cols=[]
#             )
#
#         return self
#
#     def transform(self, X, y):
#         """
#         Sample proportionally from each of the vanilla GANs trained on the
#         subsets split by class.
#         """
#         _set_global_random_state(self.random_state)
#
#         # return an empty dataframe and target or the original samples if the multiplication factor is too small
#         if int(self.sample_multiplication_factor * len(X)) < 1:
#             if self.only_sampled:
#                 return pd.DataFrame(columns=pd.DataFrame(X).columns), pd.Series([])
#             else:
#                 return X, y
#
#         # store column titles to restore them after sampling if available
#         if hasattr(X, 'columns'):
#             original_column_titles = X.columns
#         if hasattr(y, 'name'):
#             original_target_title = y.name
#
#         original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
#         original_target = pd.Series(y).copy().reset_index(drop=True)
#
#         # calculate the number of occurrences per class
#         unique, counts = np.unique(original_target, return_counts=True)
#         occurrences_per_class_dict = dict(zip(unique, counts))
#
#         resampled_subsets_per_class = []
#         resampled_targets_per_class = []
#         for class_name in occurrences_per_class_dict:
#             number_of_samples = int(self.sample_multiplication_factor * occurrences_per_class_dict[class_name])
#
#             resampled_subset = self._vanilla_gan[class_name].sample(
#                 n_samples=number_of_samples,
#             )
#             resampled_target = np.full((number_of_samples,), class_name).T
#
#             resampled_subset = pd.DataFrame(resampled_subset)
#             resampled_target = pd.Series(resampled_target)
#
#             # remove excess generated samples
#             resampled_subset = resampled_subset.iloc[:number_of_samples, :]
#
#             resampled_subsets_per_class.append(resampled_subset)
#             resampled_targets_per_class.append(resampled_target)
#
#         # join resampled subsets and targets
#         resampled_dataset = pd.concat(resampled_subsets_per_class, ignore_index=True)
#         resampled_dataset = resampled_dataset.reset_index(drop=True)
#         resampled_target = pd.concat(resampled_targets_per_class, ignore_index=True)
#         resampled_target = resampled_target.reset_index(drop=True)
#
#         if not self.only_sampled:
#             # add original samples and target
#             original_dataset.columns = range(len(original_dataset.columns))
#             original_target.name = len(original_dataset.columns)
#             resampled_dataset.columns = range(len(resampled_dataset.columns))
#             resampled_target.name = len(resampled_dataset.columns)
#
#             resampled_dataset = pd.concat([original_dataset, resampled_dataset], ignore_index=True)
#             resampled_dataset.reset_index(drop=True)
#             resampled_target = pd.concat([original_target, resampled_target], ignore_index=True)
#             resampled_target.reset_index(drop=True)
#
#         # restore column titles if available
#         if hasattr(X, 'columns'):
#             resampled_dataset.columns = original_column_titles
#         if hasattr(y, 'name'):
#             resampled_target.name = original_target_title
#
#         self.clear_cache()
#
#         return resampled_dataset, resampled_target
#
#     def clear_cache(self):
#         if hasattr(self, '_cache_prefixes'):
#             for cache_prefix in self._cache_prefixes:
#                 _clear_gan_cache(cache_prefix)
#             self._cache_prefixes = []
#
#
# class UnlabeledVanillaGANSampler(UnlabeledSampler):
#     """
#     Transformer that implements a sampling routine for a trained vanilla GAN
#     model on unlabeled data.
#
#     Parameters
#     ----------
#
#     sample_multiplication_factor : float
#         Determines the relative amount of generated data, i.e. 0 means that no
#         data is generated and 1 means that we generate the same number of data
#         points as we already have.
#
#     only_sampled : bool, default=False
#         Determines whether the original dataset is prepended to the generated
#         samples.
#
#     batch_size: int, default=128
#         Number of samples used per training step.
#
#     learning_rate: float, default=1e-4
#         The learning rate for each training step.
#
#     betas: tuple, default=(0.5, 0.9)
#         Initial decay rates of Adam when estimating the first and second
#         moments of the gradient.
#
#     noise_dim: int, default=264
#         The length of the noise vector per example.
#
#     layers_dim: int, default=128
#         The dimension of the networks layers.
#
#     epochs: int, default=300
#         Total number of training steps.
#
#     sample_interval: int, default=50
#         The interval between samples.
#
#     random_state : int, default=None
#         Control the randomization of the algorithm.
#
#     References
#     ----------
#
#     .. [1] https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/regular/vanillagan/model.py
#
#     """
#
#     def __init__(
#             self,
#             sample_multiplication_factor,
#             only_sampled=False,
#             batch_size=128,
#             learning_rate=1e-4,
#             betas=(0.5, 0.9),
#             noise_dim=264,
#             layers_dim=128,
#             epochs=300,
#             sample_interval=50,
#             random_state=None
#     ):
#         self.sample_multiplication_factor = sample_multiplication_factor
#         self.only_sampled = only_sampled
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.betas = betas
#         self.noise_dim = noise_dim
#         self.layers_dim = layers_dim
#         self.epochs = epochs
#         self.sample_interval = sample_interval
#         self.random_state = random_state
#
#     def fit(self, X, y=None):
#         _set_global_random_state(self.random_state)
#
#         if not hasattr(self, '_cache_prefixes'):
#             self._cache_prefixes = []
#
#         original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
#         original_dataset.columns = _convert_list_to_string_list(original_dataset.columns)
#
#         self._vanilla_gan = VanilllaGAN(
#             model_parameters=ModelParameters(
#                 batch_size=self.batch_size,
#                 lr=self.learning_rate,
#                 betas=self.betas,
#                 noise_dim=self.noise_dim,
#                 layers_dim=self.layers_dim
#             )
#         )
#
#         cache_prefix = str(int(time.time() * 1000)) + '_' + str(uuid4())
#         self._cache_prefixes.append(cache_prefix)
#
#         self._vanilla_gan.train(
#             data=original_dataset,
#             train_arguments=TrainParameters(
#                 cache_prefix=cache_prefix,
#                 epochs=self.epochs,
#                 sample_interval=self.sample_interval
#             ),
#             num_cols=original_dataset.columns.tolist(),
#             cat_cols=[]
#         )
#
#         return self
#
#     def transform(self, X, y=None):
#         """
#         Runs a vanilla GAN sampling routine trained on the entire dataset.
#         """
#         _set_global_random_state(self.random_state)
#
#         # return an empty dataframe and target or the original samples if the multiplication factor is too small
#         if int(self.sample_multiplication_factor * len(X)) < 1:
#             if self.only_sampled:
#                 return pd.DataFrame(columns=pd.DataFrame(X).columns), pd.Series([])
#             else:
#                 return X, y
#
#         # store column titles to restore them after sampling if available
#         if hasattr(X, 'columns'):
#             original_column_titles = X.columns
#
#         original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
#
#         # resample the original dataset
#         resampled_dataset = self._vanilla_gan.sample(
#             n_samples=int(self.sample_multiplication_factor * len(original_dataset))
#         )
#         resampled_dataset = pd.DataFrame(resampled_dataset)
#
#         # drop excess samples
#         resampled_dataset = resampled_dataset.head(
#             int(self.sample_multiplication_factor * len(original_dataset))
#         )
#         resampled_dataset = resampled_dataset.reset_index(drop=True)
#
#         if not self.only_sampled:
#             # add original samples
#             original_dataset.columns = range(len(original_dataset.columns))
#             resampled_dataset.columns = range(len(resampled_dataset.columns))
#
#             resampled_dataset = pd.concat([original_dataset, resampled_dataset], ignore_index=True)
#             resampled_dataset.reset_index(drop=True)
#
#             resampled_target = pd.concat([
#                 pd.Series(y).copy().reset_index(drop=True),
#                 pd.Series(
#                     np.full(int(self.sample_multiplication_factor * len(X)), np.nan)
#                 ).copy().reset_index(drop=True)
#             ], ignore_index=True)
#             resampled_target = pd.Series(resampled_target).reset_index(drop=True)
#         else:
#             resampled_dataset = None
#
#         # restore column titles if available
#         if hasattr(X, 'columns'):
#             resampled_dataset.columns = original_column_titles
#
#         self.clear_cache()
#
#         return resampled_dataset, resampled_target
#
#     def clear_cache(self):
#         if hasattr(self, '_cache_prefixes'):
#             for cache_prefix in self._cache_prefixes:
#                 _clear_gan_cache(cache_prefix)
#             self._cache_prefixes = []
#
#
# class ProportionalConditionalGANSampler(ProportionalSampler):
#     """
#     Transformer that implements a proportional sampling routine using a
#     conditional GAN implementation. For each class, we sample a proportional
#     amount of samples using the condition vector.
#
#     Parameters
#     ----------
#
#     sample_multiplication_factor : float
#         Determines the relative amount of generated data, i.e. 0 means that no
#         data is generated and 1 means that we generate the same number of data
#         points as we already have.
#
#     only_sampled : bool, default=False
#         Determines whether the original dataset is prepended to the generated
#         samples.
#
#     batch_size: int, default=128
#         Number of samples used per training step.
#
#     learning_rate: float, default=1e-4
#         The learning rate for each training step.
#
#     betas: tuple, default=(0.5, 0.9)
#         Initial decay rates of Adam when estimating the first and second
#         moments of the gradient.
#
#     noise_dim: int, default=264
#         The length of the noise vector per example.
#
#     layers_dim: int, default=128
#         The dimension of the networks layers.
#
#     epochs: int, default=300
#         Total number of training steps.
#
#     sample_interval: int, default=50
#         The interval between samples.
#
#     random_state : int, default=None
#         Control the randomization of the algorithm.
#
#     References
#     ----------
#
#     .. [1] https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/regular/cgan/model.py
#
#     """
#
#     def __init__(
#             self,
#             sample_multiplication_factor,
#             only_sampled=False,
#             batch_size=128,
#             learning_rate=1e-4,
#             betas=(0.5, 0.9),
#             noise_dim=264,
#             layers_dim=128,
#             epochs=300,
#             sample_interval=50,
#             random_state=None
#     ):
#         self.sample_multiplication_factor = sample_multiplication_factor
#         self.only_sampled = only_sampled
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.betas = betas
#         self.noise_dim = noise_dim
#         self.layers_dim = layers_dim
#         self.epochs = epochs
#         self.sample_interval = sample_interval
#         self.random_state = random_state
#
#     def fit(self, X, y):
#         _set_global_random_state(self.random_state)
#
#         if not hasattr(self, '_cache_prefixes'):
#             self._cache_prefixes = []
#
#         # set the number of classes
#         unique, _ = np.unique(y, return_counts=True)
#         self._cgan = CGAN(
#             model_parameters=ModelParameters(
#                 batch_size=self.batch_size,
#                 lr=self.learning_rate,
#                 betas=self.betas,
#                 noise_dim=self.noise_dim,
#                 layers_dim=self.layers_dim
#             ),
#             num_classes=len(unique)
#         )
#
#         # change the column titles for easier use
#         original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
#         y = pd.Series(y).copy().reset_index(drop=True)
#         original_dataset.columns = _convert_list_to_string_list(range(0, len(original_dataset.columns)))
#         num_cols = original_dataset.columns.copy().tolist()
#
#         # add the target column to the dataset
#         target_column_title = str(len(original_dataset.columns))
#         original_dataset[target_column_title] = y
#
#         cache_prefix = str(int(time.time() * 1000)) + '_' + str(uuid4())
#         self._cache_prefixes.append(cache_prefix)
#
#         self._cgan.train(
#             data=original_dataset,
#             label_col=target_column_title,
#             train_arguments=TrainParameters(
#                 cache_prefix=cache_prefix,
#                 epochs=self.epochs,
#                 sample_interval=self.sample_interval
#             ),
#             num_cols=num_cols,
#             cat_cols=[]
#         )
#
#         return self
#
#     def transform(self, X, y):
#         """
#         Sample a proportional number of samples from the generated conditional
#         GAN model.
#         """
#         _set_global_random_state(self.random_state)
#
#         # return an empty dataframe and target or the original samples if the multiplication factor is too small
#         if int(self.sample_multiplication_factor * len(X)) < 1:
#             if self.only_sampled:
#                 return pd.DataFrame(columns=pd.DataFrame(X).columns), pd.Series([])
#             else:
#                 return X, y
#
#         # store column titles to restore them after sampling if available
#         if hasattr(X, 'columns'):
#             original_column_titles = X.columns
#         if hasattr(y, 'name'):
#             original_target_title = y.name
#
#         original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
#         original_target = pd.Series(y).copy().reset_index(drop=True)
#
#         # calculate the number of occurrences per class
#         unique, counts = np.unique(original_target, return_counts=True)
#         occurrences_per_class_dict = dict(zip(unique, counts))
#
#         # resample the original dataset proportionally for each class
#         resampled_subsets_per_class = []
#         for class_name in occurrences_per_class_dict:
#             number_of_samples = int(self.sample_multiplication_factor * occurrences_per_class_dict[class_name])
#             condition = np.array([class_name])
#             resampled_subset = self._cgan.sample(
#                 n_samples=number_of_samples,
#                 condition=condition
#             )
#             resampled_subset = pd.DataFrame(resampled_subset)
#
#             # remove excess generated samples
#             resampled_subset = resampled_subset.iloc[:number_of_samples, :]
#
#             resampled_subsets_per_class.append(resampled_subset)
#
#         # join resampled subsets into one dataset
#         resampled_dataset = pd.concat(resampled_subsets_per_class, ignore_index=True)
#         resampled_dataset = pd.DataFrame(resampled_dataset)
#
#         # extract the target column from the generated dataset
#         resampled_dataset.columns = range(0, len(resampled_dataset.columns))
#         target_column_name = len(resampled_dataset.columns) - 1
#         resampled_target = resampled_dataset[target_column_name]
#         resampled_dataset = resampled_dataset.drop(columns=[target_column_name])
#
#         # convert target entries to numpy because they are returned as tensorflow tensor
#         resampled_target_numpy = []
#         for entry in resampled_target:
#             resampled_target_numpy.append(entry[0].numpy())
#         resampled_target = pd.Series(resampled_target_numpy)
#
#         resampled_dataset = resampled_dataset.reset_index(drop=True)
#         resampled_target = resampled_target.reset_index(drop=True)
#
#         if not self.only_sampled:
#             # add original samples and target
#             original_dataset.columns = range(len(original_dataset.columns))
#             original_target.name = len(original_dataset.columns)
#             resampled_dataset.columns = range(len(resampled_dataset.columns))
#             resampled_target.name = len(resampled_dataset.columns)
#
#             resampled_dataset = pd.concat([original_dataset, resampled_dataset], ignore_index=True)
#             resampled_dataset.reset_index(drop=True)
#             resampled_target = pd.concat([original_target, resampled_target], ignore_index=True)
#             resampled_target.reset_index(drop=True)
#
#         # restore column titles if available
#         if hasattr(X, 'columns'):
#             resampled_dataset.columns = original_column_titles
#         if hasattr(y, 'name'):
#             resampled_target.name = original_target_title
#
#         self.clear_cache()
#
#         return resampled_dataset, resampled_target
#
#     def clear_cache(self):
#         if hasattr(self, '_cache_prefixes'):
#             for cache_prefix in self._cache_prefixes:
#                 _clear_gan_cache(cache_prefix)
#             self._cache_prefixes = []
#
#
# class UnlabeledConditionalGANSampler(UnlabeledSampler):
#     """
#     Transformer that implements an unlabeled sampling routine using a
#     conditional GAN implementation where we set the target vector to be all one
#     class.
#
#     Parameters
#     ----------
#
#     sample_multiplication_factor : float
#         Determines the relative amount of generated data, i.e. 0 means that no
#         data is generated and 1 means that we generate the same number of data
#         points as we already have.
#
#     only_sampled : bool, default=False
#         Determines whether the original dataset is prepended to the generated
#         samples.
#
#     batch_size: int, default=128
#         Number of samples used per training step.
#
#     learning_rate: float, default=1e-4
#         The learning rate for each training step.
#
#     betas: tuple, default=(0.5, 0.9)
#         Initial decay rates of Adam when estimating the first and second
#         moments of the gradient.
#
#     noise_dim: int, default=264
#         The length of the noise vector per example.
#
#     layers_dim: int, default=128
#         The dimension of the networks layers.
#
#     epochs: int, default=300
#         Total number of training steps.
#
#     sample_interval: int, default=50
#         The interval between samples.
#
#     random_state : int, default=None
#         Control the randomization of the algorithm.
#
#     References
#     ----------
#
#     .. [1] https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/regular/vanillagan/model.py
#
#     """
#
#     def __init__(
#             self,
#             sample_multiplication_factor,
#             only_sampled=False,
#             batch_size=128,
#             learning_rate=1e-4,
#             betas=(0.5, 0.9),
#             noise_dim=264,
#             layers_dim=128,
#             epochs=300,
#             sample_interval=50,
#             random_state=None
#     ):
#         self.sample_multiplication_factor = sample_multiplication_factor
#         self.only_sampled = only_sampled
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.betas = betas
#         self.noise_dim = noise_dim
#         self.layers_dim = layers_dim
#         self.epochs = epochs
#         self.sample_interval = sample_interval
#         self.random_state = random_state
#
#     def fit(self, X, y=None):
#         _set_global_random_state(self.random_state)
#
#         if not hasattr(self, '_cache_prefixes'):
#             self._cache_prefixes = []
#
#         # change the column titles for easier use
#         original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
#         original_dataset.columns = _convert_list_to_string_list(range(0, len(original_dataset.columns)))
#         num_cols = original_dataset.columns.copy().tolist()
#
#         # add the target column to the dataset
#         target_column_title = len(original_dataset.columns)
#         original_dataset[target_column_title] = np.full((len(X),), 0).T
#
#         self._cgan = CGAN(
#             model_parameters=ModelParameters(
#                 batch_size=self.batch_size,
#                 lr=self.learning_rate,
#                 betas=self.betas,
#                 noise_dim=self.noise_dim,
#                 layers_dim=self.layers_dim
#             ),
#             num_classes=1
#         )
#
#         cache_prefix = str(int(time.time() * 1000)) + '_' + str(uuid4())
#         self._cache_prefixes.append(cache_prefix)
#
#         self._cgan.train(
#             data=original_dataset,
#             label_col=target_column_title,
#             train_arguments=TrainParameters(
#                 cache_prefix=cache_prefix,
#                 epochs=self.epochs,
#                 sample_interval=self.sample_interval
#             ),
#             num_cols=num_cols,
#             cat_cols=[]
#         )
#
#         return self
#
#     def transform(self, X, y=None):
#         """
#         Sample a proportional number of samples from the generated conditional
#         GAN model.
#         """
#         _set_global_random_state(self.random_state)
#
#         # return an empty dataframe and target or the original samples if the multiplication factor is too small
#         if int(self.sample_multiplication_factor * len(X)) < 1:
#             if self.only_sampled:
#                 return pd.DataFrame(columns=pd.DataFrame(X).columns), pd.Series([])
#             else:
#                 return X, y
#
#         # store column titles to restore them after sampling if available
#         if hasattr(X, 'columns'):
#             original_column_titles = X.columns
#
#         original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
#
#         number_of_samples = int(self.sample_multiplication_factor * len(X))
#         condition = np.array([0])
#         resampled_dataset = self._cgan.sample(
#             n_samples=number_of_samples,
#             condition=condition
#         )
#         resampled_dataset = pd.DataFrame(resampled_dataset)
#
#         # remove excess generated samples
#         resampled_dataset = resampled_dataset.iloc[:number_of_samples, :]
#
#         # drop the target column from the generated dataset
#         resampled_dataset.columns = range(0, len(resampled_dataset.columns))
#         resampled_dataset = resampled_dataset.drop(columns=[len(resampled_dataset.columns) - 1])
#         resampled_dataset = resampled_dataset.reset_index(drop=True)
#
#         if not self.only_sampled:
#             # add original samples
#             original_dataset.columns = range(len(original_dataset.columns))
#             resampled_dataset.columns = range(len(resampled_dataset.columns))
#
#             resampled_dataset = pd.concat([original_dataset, resampled_dataset], ignore_index=True)
#             resampled_dataset.reset_index(drop=True)
#
#             resampled_target = pd.concat([
#                 pd.Series(y).copy().reset_index(drop=True),
#                 pd.Series(
#                     np.full(int(self.sample_multiplication_factor * len(X)), np.nan)
#                 ).copy().reset_index(drop=True)
#             ], ignore_index=True)
#             resampled_target = pd.Series(resampled_target).reset_index(drop=True)
#         else:
#             resampled_dataset = None
#
#         # restore column titles if available
#         if hasattr(X, 'columns'):
#             resampled_dataset.columns = original_column_titles
#
#         self.clear_cache()
#
#         return resampled_dataset, resampled_target
#
#     def clear_cache(self):
#         if hasattr(self, '_cache_prefixes'):
#             for cache_prefix in self._cache_prefixes:
#                 _clear_gan_cache(cache_prefix)
#             self._cache_prefixes = []
#
#
# class ProportionalDRAGANSampler(ProportionalSampler):
#     """
#     Transformer that implements a proportional sampling routine using a DRAGAN
#     implementation. We train and sample from a different DRAGAN model for each
#     class.
#
#     Parameters
#     ----------
#
#     sample_multiplication_factor : float
#         Determines the relative amount of generated data, i.e. 0 means that no
#         data is generated and 1 means that we generate the same number of data
#         points as we already have.
#
#     only_sampled : bool, default=False
#         Determines whether the original dataset is prepended to the generated
#         samples.
#
#     discriminator_updates_per_step : int, default=1
#         Determines how many times the discriminator is updated in each training
#         step.
#
#     batch_size: int, default=128
#         Number of samples used per training step.
#
#     learning_rate: float, default=1e-4
#         The learning rate for each training step.
#
#     betas: tuple, default=(0.5, 0.9)
#         Initial decay rates of Adam when estimating the first and second
#         moments of the gradient.
#
#     noise_dim: int, default=264
#         The length of the noise vector per example.
#
#     layers_dim: int, default=128
#         The dimension of the networks layers.
#
#     epochs: int, default=300
#         Total number of training steps.
#
#     sample_interval: int, default=50
#         The interval between samples.
#
#     random_state : int, default=None
#         Control the randomization of the algorithm.
#
#     References
#     ----------
#
#     .. [1] https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/regular/dragan/model.py
#
#     """
#
#     def __init__(
#             self,
#             sample_multiplication_factor,
#             only_sampled=False,
#             discriminator_updates_per_step=1,
#             batch_size=128,
#             learning_rate=1e-4,
#             betas=(0.5, 0.9),
#             noise_dim=264,
#             layers_dim=128,
#             epochs=300,
#             sample_interval=50,
#             random_state=None
#     ):
#         self.sample_multiplication_factor = sample_multiplication_factor
#         self.only_sampled = only_sampled
#         self.discriminator_updates_per_step = discriminator_updates_per_step
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.betas = betas
#         self.noise_dim = noise_dim
#         self.layers_dim = layers_dim
#         self.epochs = epochs
#         self.sample_interval = sample_interval
#         self.random_state = random_state
#
#     def fit(self, X, y):
#         _set_global_random_state(self.random_state)
#
#         if not hasattr(self, '_cache_prefixes'):
#             self._cache_prefixes = []
#
#         # reset dragan models
#         self._dragan = {}
#
#         # change the column titles for easier use
#         original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
#         y = pd.Series(y).copy().reset_index(drop=True)
#         original_dataset.columns = _convert_list_to_string_list(range(0, len(original_dataset.columns)))
#         num_cols = original_dataset.columns.copy().tolist()
#
#         # calculate the number of occurrences per class
#         unique, counts = np.unique(y, return_counts=True)
#         occurrences_per_class_dict = dict(zip(unique, counts))
#
#         for class_name in occurrences_per_class_dict:
#             self._dragan[class_name] = DRAGAN(
#                 model_parameters=ModelParameters(
#                     batch_size=self.batch_size,
#                     lr=self.learning_rate,
#                     betas=self.betas,
#                     noise_dim=self.noise_dim,
#                     layers_dim=self.layers_dim
#                 ),
#                 n_discriminator=self.discriminator_updates_per_step
#             )
#
#             original_subset = original_dataset.iloc[[x for x in range(0, len(y)) if y[x] == class_name]]
#
#             cache_prefix = str(int(time.time() * 1000)) + '_' + str(uuid4())
#             self._cache_prefixes.append(cache_prefix)
#
#             self._dragan[class_name].train(
#                 data=original_subset,
#                 train_arguments=TrainParameters(
#                     cache_prefix=cache_prefix,
#                     epochs=self.epochs,
#                     sample_interval=self.sample_interval
#                 ),
#                 num_cols=num_cols,
#                 cat_cols=[]
#             )
#
#         return self
#
#     def transform(self, X, y):
#         """
#         Sample proportionally from each of the DRAGANs trained on the subsets
#         split by class.
#         """
#         _set_global_random_state(self.random_state)
#
#         # return an empty dataframe and target or the original samples if the multiplication factor is too small
#         if int(self.sample_multiplication_factor * len(X)) < 1:
#             if self.only_sampled:
#                 return pd.DataFrame(columns=pd.DataFrame(X).columns), pd.Series([])
#             else:
#                 return X, y
#
#         # store column titles to restore them after sampling if available
#         if hasattr(X, 'columns'):
#             original_column_titles = X.columns
#         if hasattr(y, 'name'):
#             original_target_title = y.name
#
#         original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
#         original_target = pd.Series(y).copy().reset_index(drop=True)
#
#         # calculate the number of occurrences per class
#         unique, counts = np.unique(original_target, return_counts=True)
#         occurrences_per_class_dict = dict(zip(unique, counts))
#
#         resampled_subsets_per_class = []
#         resampled_targets_per_class = []
#         for class_name in occurrences_per_class_dict:
#             number_of_samples = int(self.sample_multiplication_factor * occurrences_per_class_dict[class_name])
#
#             resampled_subset = self._dragan[class_name].sample(
#                 n_samples=number_of_samples,
#             )
#             resampled_target = np.full((number_of_samples,), class_name).T
#
#             resampled_subset = pd.DataFrame(resampled_subset)
#             resampled_target = pd.Series(resampled_target)
#
#             # remove excess generated samples
#             resampled_subset = resampled_subset.iloc[:number_of_samples, :]
#
#             resampled_subsets_per_class.append(resampled_subset)
#             resampled_targets_per_class.append(resampled_target)
#
#         # join resampled subsets and targets
#         resampled_dataset = pd.concat(resampled_subsets_per_class, ignore_index=True)
#         resampled_dataset = resampled_dataset.reset_index(drop=True)
#         resampled_target = pd.concat(resampled_targets_per_class, ignore_index=True)
#         resampled_target = resampled_target.reset_index(drop=True)
#
#         if not self.only_sampled:
#             # add original samples and target
#             original_dataset.columns = range(len(original_dataset.columns))
#             original_target.name = len(original_dataset.columns)
#             resampled_dataset.columns = range(len(resampled_dataset.columns))
#             resampled_target.name = len(resampled_dataset.columns)
#
#             resampled_dataset = pd.concat([original_dataset, resampled_dataset], ignore_index=True)
#             resampled_dataset.reset_index(drop=True)
#             resampled_target = pd.concat([original_target, resampled_target], ignore_index=True)
#             resampled_target.reset_index(drop=True)
#
#         # restore column titles if available
#         if hasattr(X, 'columns'):
#             resampled_dataset.columns = original_column_titles
#         if hasattr(y, 'name'):
#             resampled_target.name = original_target_title
#
#         self.clear_cache()
#
#         return resampled_dataset, resampled_target
#
#     def clear_cache(self):
#         if hasattr(self, '_cache_prefixes'):
#             for cache_prefix in self._cache_prefixes:
#                 _clear_gan_cache(cache_prefix)
#             self._cache_prefixes = []
#
#
# class UnlabeledDRAGANSampler(UnlabeledSampler):
#     """
#     Transformer that implements an unlabeled sampling routine using a DRAGAN
#     implementation.
#
#     Parameters
#     ----------
#
#     sample_multiplication_factor : float
#         Determines the relative amount of generated data, i.e. 0 means that no
#         data is generated and 1 means that we generate the same number of data
#         points as we already have.
#
#     only_sampled : bool, default=False
#         Determines whether the original dataset is prepended to the generated
#         samples.
#
#     discriminator_updates_per_step : int, default=1
#         Determines how many times the discriminator is updated in each training
#         step.
#
#     batch_size: int, default=128
#         Number of samples used per training step.
#
#     learning_rate: float, default=1e-4
#         The learning rate for each training step.
#
#     betas: tuple, default=(0.5, 0.9)
#         Initial decay rates of Adam when estimating the first and second
#         moments of the gradient.
#
#     noise_dim: int, default=264
#         The length of the noise vector per example.
#
#     layers_dim: int, default=128
#         The dimension of the networks layers.
#
#     epochs: int, default=300
#         Total number of training steps.
#
#     sample_interval: int, default=50
#         The interval between samples.
#
#     random_state : int, default=None
#         Control the randomization of the algorithm.
#
#     References
#     ----------
#
#     .. [1] https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/regular/dragan/model.py
#
#     """
#
#     def __init__(
#             self,
#             sample_multiplication_factor,
#             only_sampled=False,
#             discriminator_updates_per_step=1,
#             batch_size=128,
#             learning_rate=1e-4,
#             betas=(0.5, 0.9),
#             noise_dim=264,
#             layers_dim=128,
#             epochs=300,
#             sample_interval=50,
#             random_state=None
#     ):
#         self.sample_multiplication_factor = sample_multiplication_factor
#         self.only_sampled = only_sampled
#         self.discriminator_updates_per_step = discriminator_updates_per_step
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.betas = betas
#         self.noise_dim = noise_dim
#         self.layers_dim = layers_dim
#         self.epochs = epochs
#         self.sample_interval = sample_interval
#         self.random_state = random_state
#
#     def fit(self, X, y=None):
#         _set_global_random_state(self.random_state)
#
#         if not hasattr(self, '_cache_prefixes'):
#             self._cache_prefixes = []
#
#         # change the column titles for easier use
#         original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
#         original_dataset.columns = _convert_list_to_string_list(range(0, len(original_dataset.columns)))
#         num_cols = original_dataset.columns.copy().tolist()
#
#         self._dragan = DRAGAN(
#             model_parameters=ModelParameters(
#                 batch_size=self.batch_size,
#                 lr=self.learning_rate,
#                 betas=self.betas,
#                 noise_dim=self.noise_dim,
#                 layers_dim=self.layers_dim
#             ),
#             n_discriminator=self.discriminator_updates_per_step
#         )
#
#         cache_prefix = str(int(time.time() * 1000)) + '_' + str(uuid4())
#         self._cache_prefixes.append(cache_prefix)
#
#         self._dragan.train(
#             data=original_dataset,
#             train_arguments=TrainParameters(
#                 cache_prefix=cache_prefix,
#                 epochs=self.epochs,
#                 sample_interval=self.sample_interval
#             ),
#             num_cols=num_cols,
#             cat_cols=[]
#         )
#
#         return self
#
#     def transform(self, X, y=None):
#         """
#         Sample the requested number of samples from the trained GAN. Returns
#         only the generated data.
#         """
#         _set_global_random_state(self.random_state)
#
#         # return an empty dataframe and target or the original samples if the multiplication factor is too small
#         if int(self.sample_multiplication_factor * len(X)) < 1:
#             if self.only_sampled:
#                 return pd.DataFrame(columns=pd.DataFrame(X).columns), pd.Series([])
#             else:
#                 return X, y
#
#         # store column titles to restore them after sampling if available
#         if hasattr(X, 'columns'):
#             original_column_titles = X.columns
#
#         original_dataset = pd.DataFrame(X).copy().reset_index(drop=True)
#
#         number_of_samples = int(self.sample_multiplication_factor * len(X))
#         resampled_dataset = self._dragan.sample(n_samples=number_of_samples)
#         resampled_dataset = pd.DataFrame(resampled_dataset)
#
#         # remove excess generated samples
#         resampled_dataset = resampled_dataset.iloc[:number_of_samples, :]
#         resampled_dataset = resampled_dataset.reset_index(drop=True)
#
#         if not self.only_sampled:
#             # add original samples
#             original_dataset.columns = range(len(original_dataset.columns))
#             resampled_dataset.columns = range(len(resampled_dataset.columns))
#
#             resampled_dataset = pd.concat([original_dataset, resampled_dataset], ignore_index=True)
#             resampled_dataset.reset_index(drop=True)
#
#             resampled_target = pd.concat([
#                 pd.Series(y).copy().reset_index(drop=True),
#                 pd.Series(
#                     np.full(int(self.sample_multiplication_factor * len(X)), np.nan)
#                 ).copy().reset_index(drop=True)
#             ], ignore_index=True)
#             resampled_target = pd.Series(resampled_target).reset_index(drop=True)
#         else:
#             resampled_dataset = None
#
#         # restore column titles if available
#         if hasattr(X, 'columns'):
#             resampled_dataset.columns = original_column_titles
#
#         self.clear_cache()
#
#         return resampled_dataset, resampled_target
#
#     def clear_cache(self):
#         if hasattr(self, '_cache_prefixes'):
#             for cache_prefix in self._cache_prefixes:
#                 _clear_gan_cache(cache_prefix)
#             self._cache_prefixes = []
#
#
# def _convert_list_to_string_list(item_list):
#     string_list = []
#     for item in item_list:
#         string_list.append(str(item))
#     return string_list
#
#
# def _set_global_random_state(random_state):
#     random.seed(random_state)
#     np.random.seed(random_state)
#     tf.random.set_seed(random_state)
#
#
# def _clear_gan_cache(cache_prefix):
#     files_to_remove = [
#         './cache/' + cache_prefix + '_discriminator_model_weights_step_0.h5',
#         './cache/' + cache_prefix + '_generator_model_weights_step_0.h5',
#         './cache/' + cache_prefix + '_sample_0.npy'
#     ]
#
#     for file in files_to_remove:
#         try:
#             os.remove(file)
#         except:
#             pass
