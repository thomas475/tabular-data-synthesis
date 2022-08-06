import itertools
import os

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

from framework.samplers import *

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from itertools import product
import traceback

import timeit
import datetime

import copy


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


def get_generator_list(is_classification_task):
    # the weight decay
    default_l2_scales = [1e-5, 1e-4, 1e-3] # default 1e-5, 1e-3
    # the learning rate
    default_learning_rates = [1e-4, 5e-4, 1e-3] # default 1e-4
    default_generator_learning_rates = default_learning_rates.copy() # default 2e-4
    default_discriminator_learning_rates = default_learning_rates.copy() # default 2e-4
    # how many samples are trained with in each training step
    default_batch_sizes = [10, 20, 50]
    # how many times the entire dataset is passed through
    default_epochs = [50]

    generator_list = [
        (TableGANGenerator(is_classification_task=is_classification_task), {
            'batch_size': default_batch_sizes,  # default 500
            'epochs': default_epochs,  # default 300
            # 'l2scale': default_l2_scales
        }),
        (CTGANGenerator(is_classification_task=is_classification_task), {
            'batch_size': default_batch_sizes,  # default 500
            'epochs': default_epochs,  # default 300
            # 'generator_lr': default_generator_learning_rates,
            # 'discriminator_lr': default_discriminator_learning_rates
        }),
        (CopulaGANGenerator(is_classification_task=is_classification_task), {
            'batch_size': default_batch_sizes,  # default 500
            'epochs': default_epochs,  # default 300
            # 'generator_lr': default_generator_learning_rates,
            # 'discriminator_lr': default_discriminator_learning_rates
        }),
        (TVAEGenerator(is_classification_task=is_classification_task), {
            'batch_size': default_batch_sizes,  # default 500
            'epochs': default_epochs,  # default 300
            # 'l2scale': default_l2_scales
        }),
        (MedGANGenerator(is_classification_task=is_classification_task), {
            'batch_size': default_batch_sizes,  # default 1000
            'epochs': default_epochs,  # default 2000
            # 'l2scale': default_l2_scales
        }),
        (DPCTGANGenerator(is_classification_task=is_classification_task), {
            'batch_size': default_batch_sizes,  # default 500
            'epochs': default_epochs,  # default 300
            # 'generator_lr': default_generator_learning_rates,
            # 'discriminator_lr': default_discriminator_learning_rates
        }),
        # (CTABGANGenerator(is_classification_task=is_classification_task), {
        #     'batch_size': default_batch_sizes,  # default 500
        #     'epochs': [10],  # default 1
        #     # 'l2scale': default_l2_scales
        # })
    ]
    if is_classification_task:
        generator_list.append(
            (ProportionalCWGANGPGenerator(is_classification_task=is_classification_task), {
                'batch_size': default_batch_sizes,  # default 128
                'epochs': default_epochs,  # default 300
                # 'learning_rate': default_learning_rates
            }),
        )
    else:
        generator_list.append(
            (WGANGPGenerator(is_classification_task=is_classification_task), {
                'batch_size': default_batch_sizes,  # default 128
                'epochs': default_epochs,  # default 300
                # 'learning_rate': default_learning_rates
            }),
        )
    return generator_list


def get_student(is_classification_task, encoder):
    if is_classification_task:
        return (DecisionTreeClassifier(max_depth=5), {
            'max_depth': [3, 4, 5, 6],
            'criterion': ['gini', 'entropy']
        }, encoder)
    else:
        return (DecisionTreeRegressor(max_depth=5), {
            'max_depth': [3, 4, 5, 6],
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
        }, encoder)


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

def evaluate_generator(
        dataset,
        generator,
        student,
        batch_size,
        metric,
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

    generator, _ = generator
    generator = copy.deepcopy(generator)
    student, _, _ = student
    student = copy.deepcopy(student)
    metric_name, metric_function, metric_parameters = metric

    try:
        generator_tuning_start_time = timeit.default_timer()

        generator.epochs = 300
        generator.batch_size = batch_size
        generator.fit(X_train, y_train, categorical_columns, ordinal_columns)
        X_sampled, y_sampled = generator.sample(len(X_train))

        student.fit(X_sampled, y_sampled)

        generator_tuning_time = timeit.default_timer() - generator_tuning_start_time

        performance = metric_function(
            y_test,
            student.predict(X_test),
            **metric_parameters
        )

        result = {
            'dataset': dataset_name,
            'generator': type(generator).__name__,
            'student': type(student).__name__,
            'batch_size': str(batch_size),
            'optimization_metric': metric_name,
            'performance': str(performance),
            'train_size': str(train_size),
            'test_size': str(test_size),
            'random_state': str(random_state),
            'run_time': str(generator_tuning_time)
        }

        result_frame = result_frame.append(result, ignore_index=True)
    except Exception as e:
        result_frame = {
            'exception': str(e),
            'traceback': traceback.format_exc(),
            'generator': type(generator).__name__,
            'batch_size': str(batch_size)
        }
    finally:
        return result_frame


def test_generator(
        is_classification_task,
        experiment_directory,
        experiment_basename,
        dataset,
        encoder,
        scaler,
        generator_list,
        student,
        batch_size_list,
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

    processed_datasets = {}
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

        generator_permutations = itertools.product(
            generator_list, batch_size_list, metric_list
        )

        generator_results = Parallel(
            n_jobs=-1,
            verbose=verbose
        )(
            delayed(evaluate_generator)(
                dataset=(dataset_name, X_train, X_test, y_train, y_test, categorical_columns, ordinal_columns),
                generator=generator,
                student=student,
                batch_size=batch_size,
                metric=metric,
                random_state=random_state
            )
            for index, (
                generator,
                batch_size,
                metric
            ) in enumerate(generator_permutations)
        )

        for result in generator_results:
            if isinstance(result, dict):
                log_message = 'There has been an error with evaluating the generator ' + str(result['generator']) \
                              + ' for batchsize ' + str(result['batch_size']) + ' with randomstate ' + str(random_state) + ':' \
                              + '\n' + result['exception'] \
                              + '\n' + result['traceback']
            else:
                results = results.append(result, ignore_index=True).reset_index(drop=True)
                result = result.to_dict()
                log_message = 'Evaluating the generator ' + str(result['generator']) + ' for batchsize ' + str(result['batch_size']) \
                              + ' with randomstate ' + str(random_state) + ' was successful.'
            log_messages.append(log_message)

    total_run_time = timeit.default_timer() - total_run_start_time
    total_time_message = 'The total runtime was ' + str(round(total_run_time, 4)) + ' seconds (' \
                         + str(datetime.timedelta(seconds=int(total_run_time))) + ').'
    log_messages.append(total_time_message)

    print(total_time_message)

    Path(experiment_directory).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(experiment_directory, run_title)).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(experiment_directory, run_title, run_title + '.log'), 'w') as log_file:
        log_file.write('\n\n'.join(log_messages))
    results.to_csv(os.path.join(experiment_directory, run_title, run_title + '.csv'), index=False)


def get_teacher_list(is_classification_task):
    teacher_list = []
    if is_classification_task:
        teacher_list = [
            (CatBoostClassifier(), {
                'iterations': [1, 5, 10, 25, 50, 75, 100, 175, 250, 375, 500, 750, 1000]
            }),
            (LGBMClassifier(), {
                'num_iterations': [1, 5, 10, 25, 50, 75, 100, 175, 250, 375, 500, 750, 1000]
            })
        ]
    else:
        teacher_list = [
            (CatBoostRegressor(), {
                'iterations': [1, 5, 10, 25, 50, 75, 100, 175, 250, 375, 500, 750, 1000]
            }),
            (LGBMRegressor(), {
                'num_iterations': [1, 5, 10, 25, 50, 75, 100, 175, 250, 375, 500, 750, 1000]
            })
        ]
    return teacher_list

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_experiments(experiment_directory, experiment_basename, repeat_visualization=False):
    experiment_directory_path = os.path.join(os.getcwd(), experiment_directory)

    for directory_path in os.listdir(experiment_directory_path):
        absolute_directory_path = os.path.join(experiment_directory_path, directory_path)
        if directory_path.startswith(experiment_basename) and os.path.isdir(absolute_directory_path):
            files = os.listdir(absolute_directory_path)
            if 'results' not in files or repeat_visualization:
            # if True:
                csv_files = find_files_of_type(absolute_directory_path, '.csv')

                if len(csv_files) == 0:
                    print('no csv files found.')
                    continue

                csv_file = csv_files[0]

                results = pd.read_csv(csv_file)

                aggregated = False

                absolute_results_directory_path = os.path.join(absolute_directory_path, 'results')
                Path(absolute_results_directory_path).mkdir(parents=True, exist_ok=True)

                if aggregated:
                    processed_results = results[
                        ['generator', 'student', 'batch_size', 'optimization_metric', 'performance', 'run_time']
                    ].groupby(['generator', 'student', 'optimization_metric', 'batch_size'], as_index=False).mean()

                    processed_results.to_csv(
                        os.path.join(
                            absolute_results_directory_path,
                            os.path.basename(absolute_directory_path) + '_processed.csv'
                        ), index=False
                    )

                    for (generator, metric), processed_results_per_metric in processed_results.groupby(['generator', 'optimization_metric'], as_index=False):

                        for x, y in [('batch_size', 'performance'), ('batch_size', 'run_time')]:

                            sns.lineplot(
                                x=processed_results_per_metric[x],
                                y=processed_results_per_metric[y],
                                marker='o'
                            )
                            plt.title(generator + '_' + metric + ' ' + x + '_' + y)

                            file_name = generator + '_' + metric+ ' ' + x + '_' + y + '_aggregated.png'
                            plt.savefig(os.path.join(absolute_results_directory_path, file_name))

                            plt.clf()

                    if len(processed_results['generator'].value_counts()) > 1:

                        for metric, processed_results_per_metric in processed_results.groupby(['optimization_metric'], as_index=False):

                            for x, y in [('batch_size', 'performance'), ('batch_size', 'run_time')]:
                                sns.lineplot(
                                    data=processed_results_per_metric,
                                    x=x,
                                    y=y,
                                    hue='generator',
                                    marker='o'
                                )
                                plt.title(metric + ' ' + x + '_' + y)

                                file_name = metric + ' ' + x + '_' + y + '_aggregated.png'
                                plt.savefig(os.path.join(absolute_results_directory_path, file_name))

                                plt.clf()

                                # plot graph without legend
                                ax = sns.lineplot(
                                    data=processed_results_per_metric,
                                    x=x,
                                    y=y,
                                    hue='generator',
                                    marker='o'
                                )
                                ax.get_legend().remove()
                                plt.title(metric + ' ' + x + '_' + y)

                                file_name = metric + ' ' + x + '_' + y + '_no_legend_aggregated.png'
                                plt.savefig(os.path.join(absolute_results_directory_path, file_name))

                                plt.clf()
                else:
                    processed_results = results[
                        ['generator', 'student', 'batch_size', 'optimization_metric', 'performance', 'run_time']
                    ]

                    for (generator, metric), processed_results_per_metric in processed_results.groupby(['generator', 'optimization_metric'], as_index=False):

                        for x, y in [('batch_size', 'performance'), ('batch_size', 'run_time')]:

                            sns.lineplot(
                                x=processed_results_per_metric[x],
                                y=processed_results_per_metric[y],
                                marker='o'
                            )
                            plt.title(generator + '_' + metric + ' ' + x + '_' + y)

                            file_name = generator + '_' + metric+ ' ' + x + '_' + y + '.png'
                            plt.savefig(os.path.join(absolute_results_directory_path, file_name))

                            plt.clf()

                    if len(processed_results['generator'].value_counts()) > 1:

                        for metric, processed_results_per_metric in processed_results.groupby(['optimization_metric'], as_index=False):

                            for other in ['performance', 'run_time']:
                                new_processed_results = processed_results_per_metric[
                                    ['generator', 'batch_size', other]
                                ]

                                sns.relplot(
                                    data=new_processed_results,
                                    x='batch_size',
                                    y=other,
                                    col='generator',
                                    kind='line',
                                    col_wrap=4
                                )

                                file_name = metric + '_' + other + '.png'
                                plt.savefig(os.path.join(absolute_results_directory_path, file_name))

                                plt.clf()

                                # # plot graph without legend
                                # ax = sns.lineplot(
                                #     data=processed_results_per_metric,
                                #     x=x,
                                #     y=y,
                                #     hue='generator',
                                #     marker='o'
                                # )
                                # ax.get_legend().remove()
                                # plt.title(metric + ' ' + x + '_' + y)
                                #
                                # file_name = metric + ' ' + x + '_' + y + '_no_legend.png'
                                # plt.savefig(os.path.join(absolute_results_directory_path, file_name))
                                #
                                # plt.clf()

def find_files_of_type(absolute_directory_path, filetype):
    filenames = os.listdir(absolute_directory_path)
    return [os.path.join(absolute_directory_path, filename) for filename in filenames if filename.endswith(filetype)]


if __name__ == '__main__':
    for load_set in [
        load_adult
    ]:
        dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns = load_set()

        is_classification_task = dataset_task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]
        experiment_directory = os.path.join(os.getcwd(), 'experiments', 'preliminaries')
        experiment_basename = 'generator_batch_size_test'

        # user_input = 'y'
        user_input = input('Run Generators for dataset "' + dataset_name + '" ? [y/N]')
        if user_input == 'y':
            deep_ordinal_encoder = DeepOrdinalEncoder(categorical_columns=categorical_columns)
            deep_ordinal_encoder.fit(X, y)
            X, y = deep_ordinal_encoder.transform(X, y)
            categorical_columns = deep_ordinal_encoder.transform_column_titles(categorical_columns)
            ordinal_columns = deep_ordinal_encoder.transform_column_titles(ordinal_columns)

            encoder = CollapseEncoder()
            scaler = RobustScaler()
            generator_list = get_generator_list(is_classification_task)
            student = get_student(is_classification_task, BinaryEncoder(categorical_columns))
            batch_size_list = [10, 20, 25, 50, 100]
            metric_list = get_metric_list(dataset_task)
            train_size = 500
            random_state_list = [1, 2, 3, 4, 5]
            verbose = 0

            test_generator(
                is_classification_task=is_classification_task,
                experiment_directory=experiment_directory,
                experiment_basename=experiment_basename,
                dataset=(dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns),
                encoder=encoder,
                scaler=scaler,
                generator_list=generator_list,
                student=student,
                batch_size_list=batch_size_list,
                metric_list=metric_list,
                random_state_list=random_state_list,
                verbose=verbose
            )

        # user_input = 'y'
        user_input = input('Visualize Runs for dataset "' + dataset_name + '" ? [y/N]')
        if user_input == 'y':
            visualize_experiments(experiment_directory, experiment_basename, True)