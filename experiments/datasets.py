import os
from pathlib import Path
from itertools import compress

from openml.datasets import get_dataset

import numpy as np
import pandas as pd

DATASET_FOLDER = os.path.join('experiments', 'datasets')
PREPROCESSED_SUFFIX = '_preprocessed'


BINARY_CLASSIFICATION = 'binary'
MULTICLASS_CLASSIFICATION = 'multiclass'
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

        if target_column in categorical_columns:
            categorical_columns.remove(target_column)
        if target_column in ordinal_columns:
            ordinal_columns.remove(target_column)

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

    if dataset_task == REGRESSION:
        y = y.astype(float)

    return dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns


# =================================================================================================================== #
# ////////////////////////////////////////// BINARY CLASSIFICATION ////////////////////////////////////////////////// #
# =================================================================================================================== #


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


def load_bank_marketing():
    dataset_name = 'bank_marketing'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 1461

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


def load_compass():
    dataset_name = 'compass'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 44162

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_credit_approval():
    dataset_name = 'credit_approval'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 29

    return load_dataset(
        dataset_name=dataset_name,
        dataset_task=dataset_task,
        dataset_id=dataset_id
    )


# like in "Pedagogical Rule Extraction to Learn Interpretable Models - an Empirical Study"
def load_electricity():
    dataset_name = 'electricity'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 151

    return load_dataset(
        dataset_name=dataset_name,
        dataset_task=dataset_task,
        dataset_id=dataset_id,
        drop_columns=[
            'date',
            'day'
        ]
    )


def load_eye_movements():
    dataset_name = 'eye_movements'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 44157

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


# like in "Deep Neural Networks and Tabular Data: A Survey"
def load_higgs():
    dataset_name = 'higgs'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 23512

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_ibm_employee_performance():
    dataset_name = 'ibm_employee_performance'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 43897

    return load_dataset(
        dataset_name=dataset_name,
        dataset_task=dataset_task,
        dataset_id=dataset_id,
        drop_columns=[
            'EmployeeNumber'
        ]
    )


def load_kdd_cup_09_upselling():
    dataset_name = 'kdd_cup_09_upselling'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 44158

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_kr_vs_kp():
    dataset_name = 'kr_vs_kp'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 3

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_law_school_admission_binary():
    dataset_name = 'law_school_admission_binary'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 43890

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_national_longitudinal_survey_binary():
    dataset_name = 'national_longitudinal_survey_binary'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 43892

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_monks_problems_1():
    dataset_name = 'monks_problems_1'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 333

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_monks_problems_2():
    dataset_name = 'monks_problems_2'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 334

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_mushroom():
    dataset_name = 'mushroom'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 43922

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_mv():
    dataset_name = 'mv'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 881

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_nursery():
    dataset_name = 'nursery'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 959

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_sf_police_incidents():
    dataset_name = 'sf_police_incidents'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 42344

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_tic_tac_toe():
    dataset_name = 'tic_tac_toe'
    dataset_task = BINARY_CLASSIFICATION
    dataset_id = 50

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


# =================================================================================================================== #
# ////////////////////////////////////////// MULTICLASS CLASSIFICATION ////////////////////////////////////////////// #
# =================================================================================================================== #


def load_analcatdata_dmft():
    dataset_name = 'analcatdata_dmft'
    dataset_task = MULTICLASS_CLASSIFICATION
    dataset_id = 469

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_car():
    dataset_name = 'car'
    dataset_task = MULTICLASS_CLASSIFICATION
    dataset_id = 40975

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_cmc():
    dataset_name = 'cmc'
    dataset_task = MULTICLASS_CLASSIFICATION
    dataset_id = 23

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_collins():
    dataset_name = 'collins'
    dataset_task = MULTICLASS_CLASSIFICATION
    dataset_id = 40971

    return load_dataset(
        dataset_name=dataset_name,
        dataset_task=dataset_task,
        dataset_id=dataset_id,
        drop_columns=[
            'Text'
        ]
    )


def load_connect_4():
    dataset_name = 'connect_4'
    dataset_task = MULTICLASS_CLASSIFICATION
    dataset_id = 40668

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


def load_eucalyptus():
    dataset_name = 'eucalyptus'
    dataset_task = MULTICLASS_CLASSIFICATION
    dataset_id = 188

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


# like in "Fast, Accurate, and Simple Models for Tabular Data via Augmented Distillation"
def load_jungle_chess():
    dataset_name = 'jungle_chess'
    dataset_task = MULTICLASS_CLASSIFICATION
    dataset_id = 41027

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_splice():
    dataset_name = 'splice'
    dataset_task = MULTICLASS_CLASSIFICATION
    dataset_id = 46

    return load_dataset(
        dataset_name=dataset_name,
        dataset_task=dataset_task,
        dataset_id=dataset_id,
        drop_columns=[
            'Instance_name'
        ]
    )


def load_vowel():
    dataset_name = 'vowel'
    dataset_task = MULTICLASS_CLASSIFICATION
    dataset_id = 307

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


# =================================================================================================================== #
# ///////////////////////////////////////////////// REGRESSION ////////////////////////////////////////////////////// #
# =================================================================================================================== #


def load_bike_sharing_demand():
    dataset_name = 'bike_sharing_demand'
    dataset_task = REGRESSION
    dataset_id = 44063

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_black_friday():
    dataset_name = 'black_friday'
    dataset_task = REGRESSION
    dataset_id = 44057

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_brazilian_houses():
    dataset_name = 'brazilian_houses'
    dataset_task = REGRESSION
    dataset_id = 42688

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


def load_kaggle_30_days_of_ml():
    dataset_name = 'kaggle_30_days_of_ml'
    dataset_task = REGRESSION
    dataset_id = 43090

    return load_dataset(
        dataset_name=dataset_name,
        dataset_task=dataset_task,
        dataset_id=dataset_id,
        drop_columns=[
            'id'
        ]
    )


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


def load_nyc_taxi_green_dec_2016():
    dataset_name = 'nyc_taxi_green_dec_2016'
    dataset_task = REGRESSION
    dataset_id = 44065

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_online_news_popularity():
    dataset_name = 'online_news_popularity'
    dataset_task = REGRESSION
    dataset_id = 44064

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_sensory():
    dataset_name = 'sensory'
    dataset_task = REGRESSION
    dataset_id = 546

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_socmob():
    dataset_name = 'socmob'
    dataset_task = REGRESSION
    dataset_id = 541

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)


def load_yprop_4_1():
    dataset_name = 'yprop_4_1'
    dataset_task = REGRESSION
    dataset_id = 44054

    return load_dataset(dataset_name=dataset_name, dataset_task=dataset_task, dataset_id=dataset_id)
