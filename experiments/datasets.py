import os
from pathlib import Path
from itertools import compress

from openml.datasets import get_dataset

import numpy as np
import pandas as pd

DATASET_FOLDER = os.path.join('experiments', 'datasets')
PREPROCESSED_SUFFIX = '_preprocessed'


BINARY_CLASSIFICATION = 'binary_classification'
MULTICLASS_CLASSIFICATION = 'multiclass_classification'
REGRESSION = 'regression'


def load_dataset(dataset_name, dataset_task, dataset_id, target_column=None, drop_columns=[]):
    dataset = get_dataset(dataset_id=dataset_id)

    target_attribute = dataset.default_target_attribute
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=target_attribute
    )
    categorical_columns = list(compress(attribute_names, categorical_indicator))
    ordinal_columns = list(compress(attribute_names, [not i for i in categorical_indicator]))

    if target_column:
        target_attribute = target_column
        y = X[target_attribute]
        X = X.drop(columns=target_attribute)

    X = X.drop(columns=[column for column in X.columns if column in drop_columns])
    categorical_columns = [column for column in categorical_columns if column not in drop_columns]
    ordinal_columns = [column for column in ordinal_columns if column not in drop_columns]

    processed_dataset_path = os.path.join(
        DATASET_FOLDER,
        dataset_name + '_' + str(dataset_id) + PREPROCESSED_SUFFIX + '.csv'
    )
    if not os.path.exists(processed_dataset_path):
        # preprocess dataset by removing rows with empty entries
        X.replace({'?': np.nan})
        row_indices_with_nan = [index for index, row in X.iterrows() if row.isnull().any()]
        X = X.drop(index=row_indices_with_nan).reset_index(drop=True)
        y = y.drop(index=row_indices_with_nan).reset_index(drop=True)

        # # convert target column to numerical values
        # y = y.map({'<=50K': 0, '>50K': 1})

        dataset = X.copy()
        dataset[target_attribute] = y.copy()

        Path(DATASET_FOLDER).mkdir(parents=True, exist_ok=True)
        dataset.to_csv(processed_dataset_path, index=False)
    else:
        dataset = pd.read_csv(processed_dataset_path, index_col=None)
        X = dataset.drop(columns=target_attribute)
        y = dataset[target_attribute]

    return dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns


# like in "Fast, Accurate, and Simple Models for Tabular Data via Augmented Distillation" and many more
def load_adult():
    dataset_name = 'adult'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 1590

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


# like in "Fast, Accurate, and Simple Models for Tabular Data via Augmented Distillation" and many more
def load_amazon():
    dataset_name = 'amazon'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 43900

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


# like in "Synthesizing Tabular Data using Generative Adversarial Networks"
def load_census_income():
    dataset_name = 'census_income'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 4535

    return load_dataset(
        dataset_name=dataset_name,
        dataset_task=dataset_task,
        dataset_id=dataset_id,
        target_column='V42'
    )


# like in "Pedagogical Rule Extraction to Learn Interpretable Models - an Empirical Study"
def load_electricity():
    dataset_name = 'electricity'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 151

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


# like in "Deep Neural Networks and Tabular Data: A Survey"
def load_higgs():
    dataset_name = 'higgs'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 23512

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


# like in "Fast, Accurate, and Simple Models for Tabular Data via Augmented Distillation"
def load_covertype():
    dataset_name = 'covertype'
    dataset_task = MULTICLASS_CLASSIFICATION
    dataset_id = 1596

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


# like in "Fast, Accurate, and Simple Models for Tabular Data via Augmented Distillation"
def load_credit_g():
    dataset_name = 'credit_g'
    dataset_task = MULTICLASS_CLASSIFICATION
    dataset_id = 31

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


# like in "Fast, Accurate, and Simple Models for Tabular Data via Augmented Distillation"
def load_jungle_chess():
    dataset_name = 'jungle_chess'
    dataset_task = MULTICLASS_CLASSIFICATION
    dataset_id = 41027

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


# like in "Deep Neural Networks and Tabular Data: A Survey"
def load_california():
    dataset_name = 'california'
    dataset_task = REGRESSION
    dataset_id = 43939

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


# like in "Why do tree-based models still outperform deep learning on tabular data?"
def load_diamonds():
    dataset_name = 'diamonds'
    dataset_task = REGRESSION
    dataset_id = 42225

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


# like in "CTAB-GAN+: Enhancing Tabular Data Synthesis"
def load_king():
    dataset_name = 'king'
    dataset_task = REGRESSION
    dataset_id = 42092

    return load_dataset(
        dataset_name=dataset_name,
        dataset_task=dataset_task,
        dataset_id=dataset_id,
        drop_columns=[
            'id',
            'date'
        ]
    )