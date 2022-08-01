import numpy as np
import pandas as pd

import tensorflow as tf
import torch

import random

from joblib import Parallel, delayed
from experiments.datasets import *

from framework.encoders import DeepOrdinalEncoder

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer

from category_encoders import BinaryEncoder, CatBoostEncoder, CountEncoder, GLMMEncoder, OneHotEncoder, TargetEncoder
from framework.encoders import CollapseEncoder, CV5GLMMEncoder, CV5TargetEncoder, CVEncoder, CVEncoderOriginal

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from itertools import product
import traceback

import timeit
import datetime


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))


def get_encoder_list(categorical_columns):
    return [
        BinaryEncoder(cols=categorical_columns),
        CatBoostEncoder(cols=categorical_columns),
        CollapseEncoder(cols=categorical_columns),
        CountEncoder(cols=categorical_columns),
        GLMMEncoder(cols=categorical_columns),
        CV5GLMMEncoder(cols=categorical_columns),
        OneHotEncoder(cols=categorical_columns),
        TargetEncoder(cols=categorical_columns),
        CV5TargetEncoder(cols=categorical_columns),
    ]


def get_metric_list(dataset_task):
    metric_list = []
    if dataset_task == BINARY_CLASSIFICATION:
        metric_list = [
            ('accuracy', accuracy_score, {}),
            ('f1', f1_score, {}),
            ('roc_auc', roc_auc_score, {}),
        ]
    elif dataset_task == MULTICLASS_CLASSIFICATION:
        metric_list = [
            ('accuracy', accuracy_score, {}),
            ('f1_macro', f1_score, {'average': 'macro'})
        ]
    elif dataset_task == REGRESSION:
        metric_list = [
            ('neg_mean_absolute_error', mean_absolute_error, {}),
            ('r2', r2_score, {})
        ]
    return metric_list

def evaluate_teacher(
        dataset,
        teacher,
        teacher_parameters,
        metric_list,
        random_state
):
    tf.random.set_seed(random_state)
    torch.manual_seed(random_state)
    os.environ['PYTHONHASHSEED'] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)

    dataset_name, X_train, X_test, y_train, y_test, categorical_columns, ordinal_columns = dataset
    train_size = len(X_train)
    test_size = len(X_test)

    result_frame = pd.DataFrame()

    try:
        teacher.set_params(**teacher_parameters)

        for metric_name, metric_function, metric_parameters in metric_list:
            teacher_tuning_start_time = timeit.default_timer()

            teacher.fit(X_train, y_train)

            teacher_tuning_time = timeit.default_timer() - teacher_tuning_start_time

            performance = metric_function(
                y_test,
                teacher.predict(X_test),
                **metric_parameters
            )

            result = {
                'dataset': dataset_name,
                'teacher': type(teacher).__name__,
                'optimization_metric': metric_name,
                'performance': str(performance),
                'train_size': str(train_size),
                'test_size': str(test_size),
                'random_state': str(random_state),
                'run_time': str(teacher_tuning_time)
            }

            result.update(teacher_parameters)

            result_frame = result_frame.append(result, ignore_index=True)
    except Exception as e:
        result_frame = {
            'exception': str(e),
            'traceback': traceback.format_exc()
        }
    finally:
        return result_frame


def test_teacher(
        is_classification_task,
        experiment_directory,
        experiment_basename,
        dataset,
        encoder,
        scaler,
        teacher_list,
        metric_list,
        random_state_list,
        verbose
):
    total_run_start_time = timeit.default_timer()

    run_identifier = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_title = '_'.join([experiment_basename, dataset[0], run_identifier])
    results = pd.DataFrame()
    log_messages = []

    dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns = dataset

    for random_state in random_state_list:

        tf.random.set_seed(random_state)
        torch.manual_seed(random_state)
        os.environ['PYTHONHASHSEED'] = str(random_state)
        random.seed(random_state)
        np.random.seed(random_state)

        if is_classification_task:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

        # apply the encoder on the categorical columns and the scaler on the numerical columns
        column_transformer = ColumnTransformer(
            [
                ("scaler", scaler, ordinal_columns.copy()),
                ("encoder", encoder, categorical_columns.copy())
            ]
        )

        index = X_train.index
        X_train = column_transformer.fit_transform(X_train.copy(), y_train.copy())
        X_train = pd.DataFrame(X_train)
        X_train.index = index

        index = X_test.index
        X_test = column_transformer.transform(X_test.copy())
        X_test = pd.DataFrame(X_test)
        X_test.index = index

        # our dataset is completely numerical now, so we update the columns
        categorical_columns = []
        ordinal_columns = X_train.columns
        y_train = y_train.copy()
        y_train.name = len(ordinal_columns)
        y_test = y_test.copy()
        y_test.name = len(ordinal_columns)

        for teacher, teacher_grid in teacher_list:

            teacher_results = Parallel(
                n_jobs=-1,
                verbose=verbose
            )(
                delayed(evaluate_teacher)(
                    dataset=(dataset_name, X_train, X_test, y_train, y_test, categorical_columns, ordinal_columns),
                    teacher=teacher,
                    teacher_parameters=teacher_parameters,
                    metric_list=metric_list,
                    random_state=random_state
                )
                for teacher_parameters in list(product_dict(**teacher_grid))
            )

            for result in teacher_results:
                if isinstance(result, dict):
                    log_message = 'There has been an error with evaluating the teacher ' + type(teacher).__name__ \
                                  + ' with randomstate ' + str(random_state) + ':' \
                                  + '\n' + result['exception'] \
                                  + '\n' + result['traceback']
                else:
                    results = results.append(result, ignore_index=True).reset_index(drop=True)
                    log_message = 'Evaluating the teacher ' + type(teacher).__name__ + ' with randomstate ' \
                                  + str(random_state) + ' was successful.'
                log_messages.append(log_message)

    total_run_time = timeit.default_timer() - total_run_start_time
    total_time_message = 'The total runtime was ' + str(round(total_run_time, 4)) + ' seconds (' \
                         + str(datetime.timedelta(seconds=int(total_run_time))) + ').'
    log_messages.append(total_time_message)

    print(total_time_message)

    Path(experiment_directory).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(experiment_directory, run_title)).mkdir(parents=True, exist_ok=True)
    os.chdir(os.path.join(experiment_directory, run_title))
    with open(run_title + '.log', 'w') as log_file:
        log_file.write('\n\n'.join(log_messages))
    results.to_csv(run_title + '.csv', index=False)


if __name__ == '__main__':
    dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns = load_adult()

    deep_ordinal_encoder = DeepOrdinalEncoder(categorical_columns=categorical_columns)
    deep_ordinal_encoder.fit(X, y)
    X, y = deep_ordinal_encoder.transform(X, y)
    categorical_columns = deep_ordinal_encoder.transform_column_titles(categorical_columns)
    ordinal_columns = deep_ordinal_encoder.transform_column_titles(ordinal_columns)

    is_classification_task = dataset_task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]
    experiment_directory = os.path.join(os.getcwd(), 'snippets', 'runs')
    experiment_basename = 'teacher_test'
    encoder = CollapseEncoder()
    scaler = RobustScaler()
    teacher_list = [
        (CatBoostClassifier(), {
            'iterations': [10, 25, 50, 100, 500]
        }),
        # (LGBMClassifier(), {
        #
        # })
    ]
    metric_list = get_metric_list(dataset_task)
    train_size = 500
    random_state_list = [1, 2, 3, 4, 5]
    verbose = 0

    test_teacher(
        is_classification_task=is_classification_task,
        experiment_directory=experiment_directory,
        experiment_basename=experiment_basename,
        dataset=(dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns),
        encoder=encoder,
        scaler=scaler,
        teacher_list=teacher_list,
        metric_list=metric_list,
        random_state_list=random_state_list,
        verbose=verbose
    )
