import os
import random
import sys

import numpy as np
import pandas as pd

import torch
import tensorflow as tf

from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, KFold, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_absolute_error

from lightgbm import LGBMClassifier, LGBMRegressor
from lightgbm import early_stopping
from catboost import CatBoostClassifier, CatBoostRegressor

from experiments.datasets import \
    load_adult, load_amazon, load_census_income, load_electricity, load_higgs, \
    load_covertype, load_credit_g, load_jungle_chess, \
    load_california, load_diamonds, load_king, \
    BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, REGRESSION

from framework.encoders import DeepOrdinalEncoder
from framework.samplers import *

from category_encoders import BinaryEncoder, CatBoostEncoder, CountEncoder, GLMMEncoder, OneHotEncoder, TargetEncoder
from framework.encoders import CollapseEncoder, CV5GLMMEncoder, CV5TargetEncoder, CVEncoder, CVEncoderOriginal

from framework.pipelines import AugmentedEstimation

from tests.test_exploration import get_test_encoder_list, get_test_teacher, get_test_generator_list, \
    get_test_student, get_test_metric_list

from sdv.metrics.tabular import KSTest, CSTest
from dython.nominal import compute_associations
from framework.evaluation import calculate_dcr_nndr, calculate_jsd_wd

import itertools
from joblib import Parallel, delayed
import traceback
import datetime
import os
from pathlib import Path
import copy
import timeit

import warnings

import platform
import signal
from contextlib import contextmanager


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException('Execution took longer than ' + str(seconds) + ' seconds.')
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def join_grids(grids):
    joined_grid = {}
    for grid_name, grid in grids:
        for parameter in grid:
            joined_grid[grid_name + '__' + parameter] = grid[parameter]
    return joined_grid


def get_lgbm_scoring(scoring):
    def lgbm_scoring(y_true, y_pred):
        return scoring.__name__, scoring(y_true, np.round(y_pred)), True
    return lgbm_scoring


def get_encoder_list(categorical_columns, ordinal_columns):
    encoder_list = [
        BinaryEncoder(cols=categorical_columns),
        CatBoostEncoder(cols=categorical_columns),
        CountEncoder(cols=categorical_columns),
        GLMMEncoder(cols=categorical_columns),
        CV5GLMMEncoder(cols=categorical_columns),
        # OneHotEncoder(cols=categorical_columns),
        TargetEncoder(cols=categorical_columns),
        CV5TargetEncoder(cols=categorical_columns),
    ]

    if ordinal_columns:
        encoder_list.append(
            CollapseEncoder(cols=categorical_columns)
        )

    return encoder_list


def get_teacher(is_classification_task):
    if is_classification_task:
        return (LGBMClassifier(), {
            'max_depth': [-1, 2, 5, 8],
            'n_estimators': [10, 25, 50]
        })
        # CatBoostClassifier(),
        #     {
        #         'depth': [4, 6, 8, 10],
        #         'iterations': [10, 25, 50]
        #     }
    else:
        return (LGBMRegressor(), {
            'max_depth': [-1, 2, 5, 8],
            'n_estimators': [10, 25, 50]
        })


def get_student(is_classification_task, encoder):
    if is_classification_task:
        return (DecisionTreeClassifier(), {
            'max_depth': [3, 4, 5, 6],
            'criterion': ['gini', 'entropy']
        }, encoder)
    else:
        return (DecisionTreeRegressor(), {
            'max_depth': [3, 4, 5, 6],
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
        }, encoder)


def get_generator_list(is_classification_task):
    # the weight decay
    default_l2_scales = [1e-5, 1e-4, 1e-3] # default 1e-5, 1e-3
    # the learning rate
    default_learning_rates = [1e-4, 5e-4, 1e-3] # default 1e-4
    default_generator_learning_rates = default_learning_rates.copy() # default 2e-4
    default_discriminator_learning_rates = default_learning_rates.copy() # default 2e-4
    # how many samples are trained with in each training step
    default_batch_sizes = [50]
    # default_batch_sizes = [10, 20, 50]
    # how many times the entire dataset is passed through
    default_epochs = [150]

    generator_list = [
        (PrivBNGenerator(is_classification_task=is_classification_task), {
            # a noisy distribution is θ-useful if the ratio of average scale of information to average scale of noise is no less than θ
            # in a k-degree bayesian network theta as a parameter is fixed and a corresponding k is calculated
            'theta': [15, 20, 30] # default 20
        }),
        (GaussianCopulaGenerator(is_classification_task=is_classification_task), {
            'default_distribution': [  # default 'parametric'
                'univariate',
                # Let ``copulas`` select the optimal univariate distribution. This may result in non-parametric models being used.
                'parametric',
                # Let ``copulas`` select the optimal univariate distribution, but restrict the selection to parametric distributions only.
                'bounded',
                # Let ``copulas`` select the optimal univariate distribution, but restrict the selection to bounded distributions only.
            ]
        }),
        (TableGANGenerator(is_classification_task=is_classification_task), {
            'batch_size': default_batch_sizes,  # default 500
            'epochs': [300],  # default 300
            # 'l2scale': default_l2_scales
        }),
        (CTGANGenerator(is_classification_task=is_classification_task), {
            'batch_size': default_batch_sizes,  # default 500
            'epochs': [200],  # default 300
            # 'generator_lr': default_generator_learning_rates,
            # 'discriminator_lr': default_discriminator_learning_rates
        }),
        (CopulaGANGenerator(is_classification_task=is_classification_task), {
            'batch_size': default_batch_sizes,  # default 500
            'epochs': [300],  # default 300
            # 'generator_lr': default_generator_learning_rates,
            # 'discriminator_lr': default_discriminator_learning_rates
        }),
        (TVAEGenerator(is_classification_task=is_classification_task), {
            'batch_size': default_batch_sizes,  # default 500
            'epochs': [300],  # default 300
            # 'l2scale': default_l2_scales
        }),
        (MedGANGenerator(is_classification_task=is_classification_task), {
            'batch_size': default_batch_sizes,  # default 1000
            'epochs': [200],  # default 2000
            # 'l2scale': default_l2_scales
        }),
        (DPCTGANGenerator(is_classification_task=is_classification_task), {
            'batch_size': default_batch_sizes,  # default 500
            'epochs': [200],  # default 300
            # 'generator_lr': default_generator_learning_rates,
            # 'discriminator_lr': default_discriminator_learning_rates
        }),
        (CTABGANGenerator(is_classification_task=is_classification_task), {
            'batch_size': default_batch_sizes,  # default 500
            'epochs': [50],  # default 1
            # 'l2scale': default_l2_scales
        })
    ]
    if is_classification_task:
        generator_list.append(
            (ProportionalSMOTEGenerator(is_classification_task=is_classification_task), {
                'k_neighbors': [4, 5, 6, 7]  # default 5
            })
        )
        generator_list.append(
            (ProportionalCWGANGPGenerator(is_classification_task=is_classification_task), {
                'batch_size': default_batch_sizes,  # default 128
                'epochs': [200],  # default 300
                # 'learning_rate': default_learning_rates
            }),
        )
    else:
        generator_list.append(
            (WGANGPGenerator(is_classification_task=is_classification_task), {
                'batch_size': default_batch_sizes,  # default 128
                'epochs': [200],  # default 300
                # 'learning_rate': default_learning_rates
            }),
        )
    return generator_list


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


def tune_teacher(
        is_classification_task,
        dataset,
        encoder,
        scaler,
        teacher,
        metric,
        cv,
        random_state,
        verbose
):
    warnings.filterwarnings("ignore")

    tf.random.set_seed(random_state)
    torch.manual_seed(random_state)
    os.environ['PYTHONHASHSEED'] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)

    dataset_name, X_train, X_test, y_train, y_test, categorical_columns, ordinal_columns = dataset
    if encoder is not None:
        encoder = copy.deepcopy(encoder)
    scaler = copy.deepcopy(scaler)
    teacher, teacher_grid = teacher
    teacher = copy.deepcopy(teacher)
    metric_name, metric_function, metric_parameters = metric
    train_size = len(X_train)
    test_size = len(X_test)

    teacher_tuning_start_time = timeit.default_timer()

    # apply encoder/scaler
    if encoder is None:
        # if no encoder is submitted we only apply the scaler on the numerical columns
        column_transformer = ColumnTransformer(
            [
                ("scaler", scaler, ordinal_columns.copy())
            ],
            remainder='passthrough'
        )
    else:
        # apply the encoder on the categorical columns and the scaler on the numerical columns
        column_transformer = ColumnTransformer(
            [
                ("scaler", scaler, ordinal_columns.copy()),
                ("encoder", encoder, categorical_columns.copy())
            ]
        )

    original_column_order = X_train.columns

    index = X_train.index
    X_train = column_transformer.fit_transform(X_train.copy(), y_train.copy())
    X_train = pd.DataFrame(X_train)
    X_train.index = index

    index = X_test.index
    X_test = column_transformer.transform(X_test.copy())
    X_test = pd.DataFrame(X_test)
    X_test.index = index

    if encoder is None:
        # our dataset has only been encoded, so we restore the original column titles and order
        X_train.columns = ordinal_columns + categorical_columns
        X_train = X_train[original_column_order]
        X_test.columns = ordinal_columns + categorical_columns
        X_test = X_test[original_column_order]
    else:
        # our dataset is completely numerical now, so we update the columns
        categorical_columns = []
        ordinal_columns = X_train.columns
        y_train = y_train.copy()
        y_train.name = len(ordinal_columns)
        y_test = y_test.copy()
        y_test.name = len(ordinal_columns)

    # convert categorical numerical entries to int
    X_train[categorical_columns] = X_train[categorical_columns].astype(int)
    X_test[categorical_columns] = X_test[categorical_columns].astype(int)

    # tune the teacher
    tuned_teacher = GridSearchCV(
        estimator=teacher,
        param_grid=teacher_grid,
        scoring=metric_name,
        error_score="raise",
        refit=True,
        cv=cv,
        verbose=verbose
    )

    try:
        # if is_classification_task:
        #     X_train_train, X_train_validation, y_train_train, y_train_validation = train_test_split(
        #         X_train, y_train, test_size=0.1, stratify=y_train
        #     )
        # else:
        #     X_train_train, X_train_validation, y_train_train, y_train_validation = train_test_split(
        #         X_train, y_train, test_size=0.1
        #     )

        tuned_teacher.fit(
            X=X_train.copy(),
            y=y_train.copy(),
            categorical_feature=categorical_columns,
            # X=X_train_train,
            # y=y_train_train,
            # categorical_feature=categorical_columns,
            # eval_set=[(X_train_validation, y_train_validation)],
            # eval_metric=get_lgbm_scoring(metric_function),
            # callbacks=[early_stopping(50)]
        )

        teacher_tuning_time = timeit.default_timer() - teacher_tuning_start_time

        performance = metric_function(y_test, pd.Series(tuned_teacher.predict(X_test)))

        result = pd.DataFrame({
            'dataset': [dataset_name],
            'encoder': [type(encoder).__name__],
            'scaler': [type(scaler).__name__],
            'teacher': [type(teacher).__name__],
            'optimization_metric': [metric_name],
            'performance': [str(performance)],
            'train_size': [str(train_size)],
            'test_size': [str(test_size)],
            'random_state': [str(random_state)],
            'run_time': [str(teacher_tuning_time)]
        })

        # replace gridsearch instance with the tuned teacher
        tuned_teacher = tuned_teacher.best_estimator_
    except Exception as e:
        tuned_teacher = None
        result = {
            'exception': str(e),
            'traceback': traceback.format_exc()
        }
    finally:
        return type(encoder).__name__, tuned_teacher, result


def tune_student(
        is_classification_task,
        dataset,
        encoder,
        scaler,
        generator,
        student,
        tuned_teacher_dict,
        metric,
        cv,
        n_samples_list,
        random_state,
        verbose,
        timeout
):
    if type(encoder).__name__ not in tuned_teacher_dict:
        result_frame = {
            'exception': 'Teacher is NoneType',
            'traceback': traceback.format_exc()
        }
        return type(encoder).__name__, \
               type(generator).__name__, \
               result_frame

    warnings.filterwarnings("ignore")

    tf.random.set_seed(random_state)
    torch.manual_seed(random_state)
    os.environ['PYTHONHASHSEED'] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)

    result_frame = pd.DataFrame()

    dataset_name, X_train, X_test, y_train, y_test, categorical_columns, ordinal_columns = dataset
    if encoder is not None:
        encoder = copy.deepcopy(encoder)
    scaler = copy.deepcopy(scaler)
    generator, generator_grid = generator
    generator = copy.deepcopy(generator)
    student, student_grid, student_encoder = student
    student = copy.deepcopy(student)
    if student_encoder is not None:
        student_encoder = copy.deepcopy(student_encoder)
    metric_name, metric_function, metric_parameters = metric
    train_size = len(X_train)
    test_size = len(X_test)
    if type(encoder).__name__ in tuned_teacher_dict:
        teacher = tuned_teacher_dict[type(encoder).__name__]
    else:
        teacher = None

    # handle student with no augmentation
    if 0 in n_samples_list:
        n_samples_list = n_samples_list.copy()
        n_samples_list.remove(0)

    try:
        # apply encoder/scaler
        X_train, X_test, y_train, y_test, categorical_columns, ordinal_columns, preprocessing_time = preprocess(
            encoder,
            scaler,
            X_train,
            X_test,
            y_train,
            y_test,
            categorical_columns,
            ordinal_columns
        )

        # tune a student on the original train set
        tuned_student, performance, student_tuning_time = train_student(X_train, X_test, y_train, y_test,
                                                                        categorical_columns, student, student_grid,
                                                                        student_encoder, metric_name, metric_function,
                                                                        metric_parameters, cv, verbose)

        result = {
            'dataset': dataset_name,
            'encoder': type(encoder).__name__,
            'scaler': type(scaler).__name__,
            'student': type(student).__name__,
            'student_encoder': type(student_encoder).__name__,
            'optimization_metric': metric_name,
            'performance': str(performance),
            'train_size': str(train_size),
            'test_size': str(test_size),
            'random_state': str(random_state),
            'run_time': str(preprocessing_time + student_tuning_time)
        }
        result_frame = result_frame.append(result, ignore_index=True)

        # tune the generator using the tuned student in the evaluation
        tuned_generator, generator_tuning_time = train_generator(X_train, y_train, categorical_columns, ordinal_columns,
                                                                 generator, generator_grid, tuned_student,
                                                                 student_encoder, metric_name, train_size, timeout,
                                                                 verbose)

        # ========================================================================================================== #

        max_n_samples_generation_start_time = timeit.default_timer()

        # generate the maximum number of samples defined in n_samples_list using the tuned generator, so we can
        # subsample from it instead of generating again
        X_sampled_max, y_sampled_max = tuned_generator.sample(max(n_samples_list))

        # convert categorical numerical entries to int
        X_sampled_max[categorical_columns] = X_sampled_max[categorical_columns].astype(int)
        X_sampled_max[categorical_columns] = X_sampled_max[categorical_columns].astype(int)

        max_n_samples_generation_time = timeit.default_timer() - max_n_samples_generation_start_time

        # ========================================================================================================== #

        max_n_samples_relabelling_start_time = timeit.default_timer()

        # relabeld the maximum number of generated samples defined in n_samples_list using the tuned teacher, so we can
        # subsample from it instead of relabelling again
        y_sampled_relabeled_max = pd.Series(teacher.predict(X_sampled_max))

        max_n_samples_relabelling_time = timeit.default_timer() - max_n_samples_relabelling_start_time

        # ========================================================================================================== #

        # generate the different amount of samples with the tuned generator and evaluate them on the tuned student
        for n_samples in n_samples_list:
            X_sampled = X_sampled_max.head(n_samples)
            y_sampled = y_sampled_max.head(n_samples)

            student_fitting_start_time = timeit.default_timer()

            # use generator labels for training
            X_augmented = pd.concat([X_train, X_sampled], ignore_index=True).reset_index(drop=True)
            y_augmented = pd.concat([y_train, y_sampled], ignore_index=True).reset_index(drop=True)

            if student_encoder is not None and categorical_columns:
                tuned_student.fit(student_encoder.fit_transform(X_augmented), y_augmented)

                student_fitting_time = timeit.default_timer() - student_fitting_start_time

                performance = metric_function(
                    y_test,
                    tuned_student.predict(student_encoder.transform(X_test)),
                    **metric_parameters
                )
            else:
                tuned_student.fit(X_augmented, y_augmented)

                student_fitting_time = timeit.default_timer() - student_fitting_start_time

                performance = metric_function(
                    y_test,
                    tuned_student.predict(X_test),
                    **metric_parameters
                )

            result = {
                'dataset': dataset_name,
                'generator': type(generator).__name__,
                'encoder': type(encoder).__name__,
                'scaler': type(scaler).__name__,
                'student': type(student).__name__,
                'student_encoder': type(student_encoder).__name__,
                'optimization_metric': metric_name,
                'performance': str(performance),
                'n_samples': str(n_samples),
                'train_size': str(train_size),
                'test_size': str(test_size),
                'random_state': str(random_state),
                'run_time': str(
                    preprocessing_time + student_tuning_time + generator_tuning_time + int(
                        float(max_n_samples_generation_time) * (float(n_samples) / float(max(n_samples_list)))
                    ) + student_fitting_time
                )
            }
            result_frame = result_frame.append(result, ignore_index=True)

            # ======================================================================================================= #

            y_sampled_relabeled = y_sampled_relabeled_max.head(n_samples)

            X_augmented = pd.concat([X_train, X_sampled], ignore_index=True).reset_index(drop=True)
            y_augmented = pd.concat([y_train, y_sampled_relabeled], ignore_index=True).reset_index(drop=True)

            student_fitting_start_time = timeit.default_timer()

            if student_encoder is not None and categorical_columns:
                tuned_student.fit(student_encoder.fit_transform(X_augmented), y_augmented)

                student_fitting_time = timeit.default_timer() - student_fitting_start_time

                performance = metric_function(
                    y_test,
                    tuned_student.predict(student_encoder.transform(X_test)),
                    **metric_parameters
                )
            else:
                tuned_student.fit(X_augmented, y_augmented)

                student_fitting_time = timeit.default_timer() - student_fitting_start_time

                performance = metric_function(
                    y_test,
                    tuned_student.predict(X_test),
                    **metric_parameters
                )

            result = {
                'dataset': dataset_name,
                'generator': type(generator).__name__,
                'encoder': type(encoder).__name__,
                'scaler': type(scaler).__name__,
                'teacher': type(teacher).__name__,
                'student': type(student).__name__,
                'student_encoder': type(student_encoder).__name__,
                'optimization_metric': metric_name,
                'performance': str(performance),
                'n_samples': str(n_samples),
                'train_size': str(train_size),
                'test_size': str(test_size),
                'random_state': str(random_state),
                'run_time': str(
                    preprocessing_time + student_tuning_time + generator_tuning_time + int(
                        float(max_n_samples_generation_time) * (float(n_samples) / float(max(n_samples_list)))
                    ) + int(
                        float(max_n_samples_relabelling_time) * (float(n_samples) / float(max(n_samples_list)))
                    ) + student_fitting_time
                )
            }
            result_frame = result_frame.append(result, ignore_index=True)

        # =========================================================================================================== #

        # do a strafified split of the max number of generated samples into batches of size train_size and compare each
        # fold to the original train set.
        n_splits = int(len(X_sampled_max) / train_size)
        total_selection_size = n_splits * train_size

        if is_classification_task:
            kfold_split = StratifiedKFold(n_splits=n_splits, shuffle=False).split(
                X_sampled_max.head(total_selection_size),
                y_sampled_max.head(total_selection_size)
            )
        else:
            kfold_split = KFold(n_splits=n_splits, shuffle=False).split(
                X_sampled_max.head(total_selection_size),
                y_sampled_max.head(total_selection_size)
            )

        real_dataset = X_train.copy()
        real_dataset[y_train.name] = y_train.copy()

        if is_classification_task:
            complete_categorical_columns = categorical_columns.copy() + [y_sampled_max.name]
            complete_ordinal_columns = ordinal_columns.copy()
        else:
            complete_categorical_columns = categorical_columns.copy()
            complete_ordinal_columns = ordinal_columns.copy() + [y_sampled_max.name]

        categorical_real_dataset = real_dataset[complete_categorical_columns]
        ordinal_real_dataset = real_dataset[ordinal_columns]

        for fold, (_, fold_indeces) in enumerate(kfold_split):
            synthetic_dataset = X_sampled_max.iloc[fold_indeces].copy()
            synthetic_dataset[y_sampled_max.name] = y_sampled_max.iloc[fold_indeces].copy()

            categorical_synthetic_dataset = synthetic_dataset[complete_categorical_columns]
            ordinal_synthetic_dataset = synthetic_dataset[ordinal_columns]

            # statistical evaluation
            kstest_performance = KSTest.compute(
                real_data=ordinal_real_dataset,
                synthetic_data=ordinal_synthetic_dataset
            )
            cstest_performance = CSTest.compute(
                real_data=categorical_real_dataset.astype('category'),
                synthetic_data=categorical_synthetic_dataset.astype('category')
            )
            jsd_performance, wd_performance = calculate_jsd_wd(
                real_dataset.reset_index(drop=True),
                synthetic_dataset.reset_index(drop=True),
                complete_categorical_columns
            )
            real_corr = compute_associations(real_dataset, nominal_columns=categorical_columns)
            synthetic_corr = compute_associations(synthetic_dataset, nominal_columns=categorical_columns)
            diff_corr_performance = np.linalg.norm(real_corr - synthetic_corr)

            # privacy evaluation
            dcr_performance, nndr_performance = calculate_dcr_nndr(real_dataset, synthetic_dataset)

            result = {
                'dataset': dataset_name,
                'generator': type(generator).__name__,
                'encoder': type(encoder).__name__,
                'scaler': type(scaler).__name__,
                'student': type(student).__name__,
                'student_encoder': type(student_encoder).__name__,
                'optimization_metric': metric_name,
                'n_samples': str(train_size),
                'generated_fold': str(fold),
                'kstest': str(kstest_performance),
                'cstest': str(cstest_performance),
                'jsd': str(jsd_performance),
                'wd': str(wd_performance),
                'diff_corr': str(diff_corr_performance),
                'dcr': str(dcr_performance),
                'nndr': str(nndr_performance),
                'train_size': str(train_size),
                'test_size': str(test_size),
                'random_state': str(random_state),
                'run_time': str(
                    preprocessing_time + student_tuning_time + generator_tuning_time + int(
                        float(max_n_samples_generation_time) * (float(train_size) / float(len(X_sampled_max)))
                    )
                )
            }
            result_frame = result_frame.append(result, ignore_index=True)
    except Exception as e:
        result_frame = {
            'exception': str(e),
            'traceback': traceback.format_exc()
        }
    finally:
        return type(encoder).__name__, \
               type(generator).__name__, \
               result_frame


def train_generator(X_train, y_train, categorical_columns, ordinal_columns, generator, generator_grid, tuned_student,
                    student_encoder, metric_name, train_size, timeout, verbose):
    generator_tuning_start_time = timeit.default_timer()

    # tune the generator
    augmented_student = AugmentedEstimation(
        generator=generator,
        estimator=tuned_student,
        n_samples=train_size,
        categorical_columns=categorical_columns,
        ordinal_columns=ordinal_columns,
        encoder=student_encoder
    )

    tuned_augmented_student = GridSearchCV(
        estimator=augmented_student,
        param_grid=join_grids([('generator', generator_grid)]),
        scoring=metric_name,
        error_score="raise",
        refit=True,
        cv=ShuffleSplit(test_size=0.20, n_splits=1),
        verbose=verbose
    )

    if platform.system() == 'Linux':
        with time_limit(timeout):
            tuned_augmented_student.fit(X=X_train.copy(), y=y_train.copy())
    else:
        tuned_augmented_student.fit(X=X_train.copy(), y=y_train.copy())

    generator_tuning_time = timeit.default_timer() - generator_tuning_start_time

    tuned_generator = tuned_augmented_student.best_estimator_.get_generator()

    return tuned_generator, generator_tuning_time


def train_student(X_train, X_test, y_train, y_test, categorical_columns, student, student_grid, student_encoder,
                  metric_name, metric_function, metric_parameters, cv, verbose):
    student_tuning_start_time = timeit.default_timer()

    tuned_student = GridSearchCV(
        estimator=student,
        param_grid=student_grid,
        scoring=metric_name,
        error_score="raise",
        refit=True,
        cv=cv,
        verbose=verbose
    )

    if student_encoder is not None and categorical_columns:
        tuned_student.fit(student_encoder.fit_transform(X_train), y_train)

        student_tuning_time = timeit.default_timer() - student_tuning_start_time

        performance = metric_function(
            y_test,
            tuned_student.predict(student_encoder.transform(X_test)),
            **metric_parameters
        )
    else:
        tuned_student.fit(X_train, y_train)

        student_tuning_time = timeit.default_timer() - student_tuning_start_time

        performance = metric_function(
            y_test,
            tuned_student.predict(X_test),
            **metric_parameters
        )

    tuned_student = tuned_student.best_estimator_

    return tuned_student, performance, student_tuning_time


def preprocess(encoder, scaler, X_train, X_test, y_train, y_test, categorical_columns, ordinal_columns):
    preprocessing_start_time = timeit.default_timer()

    if encoder is None:
        # if no encoder is submitted we only apply the scaler on the numerical columns
        column_transformer = ColumnTransformer(
            [
                ("scaler", scaler, ordinal_columns.copy())
            ],
            remainder='passthrough'
        )
    else:
        # apply the encoder on the categorical columns and the scaler on the numerical columns
        column_transformer = ColumnTransformer(
            [
                ("scaler", scaler, ordinal_columns.copy()),
                ("encoder", encoder, categorical_columns.copy())
            ]
        )

    original_column_order = X_train.columns
    index = X_train.index
    X_train = column_transformer.fit_transform(X_train.copy(), y_train.copy())
    X_train = pd.DataFrame(X_train)
    X_train.index = index

    index = X_test.index
    X_test = column_transformer.transform(X_test.copy())
    X_test = pd.DataFrame(X_test)
    X_test = pd.DataFrame(X_test)
    X_test.index = index

    if encoder is None:
        # our dataset has only been encoded, so we restore the original column titles and order
        X_train.columns = ordinal_columns + categorical_columns
        X_train = X_train[original_column_order]
        X_test.columns = ordinal_columns + categorical_columns
        X_test = X_test[original_column_order]
    else:
        # our dataset is completely numerical now, so we update the columns
        categorical_columns = []
        ordinal_columns = X_train.columns
        y_train = y_train.copy()
        y_train.name = len(ordinal_columns)
        y_test = y_test.copy()
        y_test.name = len(ordinal_columns)

    # convert categorical numerical entries to int
    X_train[categorical_columns] = X_train[categorical_columns].astype(int)
    X_test[categorical_columns] = X_test[categorical_columns].astype(int)

    preprocessing_time = timeit.default_timer() - preprocessing_start_time

    return X_train, X_test, y_train, y_test, categorical_columns, ordinal_columns, preprocessing_time


def parallelized_run(
        experiment_directory,
        experiment_basename,
        is_classification_task,
        dataset,
        encoder_list,
        scaler,
        generator_list,
        student,
        teacher,
        metric_list,
        cv,
        train_size,
        n_samples_list,
        random_state_list,
        verbose,
        generator_timeout
):
    total_run_start_time = timeit.default_timer()

    run_identifier = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_title = '_'.join([experiment_basename, dataset[0], run_identifier])
    results = pd.DataFrame()
    log_messages = []

    dataset_name, X, y, categorical_columns, ordinal_columns = dataset

    completed_random_states = []
    interrupted = False

    try:

        for random_state in random_state_list:

            if is_classification_task:
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

            for metric in metric_list:

                tf.random.set_seed(random_state)
                torch.manual_seed(random_state)
                os.environ['PYTHONHASHSEED'] = str(random_state)
                random.seed(random_state)
                np.random.seed(random_state)

                teacher_permutations = itertools.product(
                    encoder_list + [None]
                )

                teacher_results = Parallel(
                    n_jobs=-1,
                    verbose=verbose
                )(
                    delayed(tune_teacher)(
                        is_classification_task=is_classification_task,
                        dataset=(dataset_name, X_train, X_test, y_train, y_test, categorical_columns, ordinal_columns),
                        encoder=encoder,
                        scaler=scaler,
                        teacher=teacher,
                        metric=metric,
                        cv=cv,
                        random_state=random_state,
                        verbose=verbose
                    )
                    for (
                        index, (
                            encoder,
                        )
                    ) in enumerate(teacher_permutations)
                )

                tuned_teacher_dict = {}
                for encoder_name, tuned_teacher, result in teacher_results:
                    if tuned_teacher is not None:
                        results = results.append(result, ignore_index=True).reset_index(drop=True)
                        tuned_teacher_dict[encoder_name] = tuned_teacher
                        log_message = 'Tuning and evaluating the teacher ' + type(teacher[0]).__name__ + ' with encoder ' \
                                      + encoder_name + ' with randomstate ' + str(random_state) + ' for metric "' + metric[0] \
                                      + '" was successful.'
                    else:
                        log_message = 'There has been an error with tuning and evaluating the teacher ' + type(teacher[0]).__name__ \
                                      + ' with encoder ' + encoder_name + ' with randomstate ' + str(random_state) \
                                      + ' for metric "' + metric[0] + '":' \
                                      + '\n' + result['exception'] \
                                      + '\n' + result['traceback']
                    log_messages.append(log_message)

                student_permutations = itertools.product(
                    encoder_list + [None],
                    generator_list
                )

                student_results = Parallel(
                    n_jobs=-1,
                    verbose=verbose
                )(
                    delayed(tune_student)(
                        is_classification_task=is_classification_task,
                        dataset=(dataset_name, X_train, X_test, y_train, y_test, categorical_columns, ordinal_columns),
                        encoder=encoder,
                        scaler=scaler,
                        generator=generator,
                        student=student,
                        tuned_teacher_dict=tuned_teacher_dict,
                        metric=metric,
                        cv=cv,
                        n_samples_list=n_samples_list,
                        random_state=random_state,
                        verbose=verbose,
                        timeout=generator_timeout
                    )
                    for (
                        index, (
                            encoder,
                            generator
                        )
                    ) in enumerate(student_permutations)
                )

                for encoder_name, generator_name, result_frame in student_results:
                    if isinstance(result_frame, dict):
                        log_message = 'There has been an error with tuning and evaluating the student ' + type(student[0]).__name__ \
                                      + ' with encoder ' + encoder_name + ' and generator ' + generator_name \
                                      + ' with randomstate ' + str(random_state) + ' with teacher ' + type(teacher[0]).__name__ \
                                      + ' for metric "' + metric[0] + '":' \
                                      + '\n' + result_frame['exception'] \
                                      + '\n' + result_frame['traceback']
                    else:
                        results = results.append(result_frame, ignore_index=True).reset_index(drop=True)
                        log_message = 'Tuning and evaluating the student ' + type(student[0]).__name__ + ' with encoder ' \
                                      + encoder_name + ' and generator ' + generator_name + ' with teacher ' \
                                      + type(teacher[0]).__name__ + ' with randomstate ' + str(random_state) + ' for metric "' \
                                      + metric[0] + '" was successful.'
                    log_messages.append(log_message)

            completed_random_states.append(random_state)

    except KeyboardInterrupt as e:
        print(e)
        interrupted = True

    total_run_time = timeit.default_timer() - total_run_start_time
    total_time_message = 'The total runtime was ' + str(round(total_run_time, 4)) + ' seconds (' \
                         + str(datetime.timedelta(seconds=int(total_run_time))) + ').'
    log_messages.append(total_time_message)

    print(total_time_message)

    # add used random_states to experiment name
    if completed_random_states:
        run_title = run_title + '_' + '_'.join(
            [str(random_state) for random_state in completed_random_states]
        )

    Path(experiment_directory).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(experiment_directory, run_title)).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(experiment_directory, run_title, run_title + '.log'), 'w') as log_file:
        log_file.write('\n\n'.join(log_messages))
    results.to_csv(os.path.join(experiment_directory, run_title, run_title + '.csv'), index=False)

    if interrupted:
        sys.exit(0)


def start_parallelized_run():
    for load_set in [
        # load_adult,
        load_amazon,
        load_census_income,
        load_electricity,
        load_higgs,
        load_covertype,
        load_credit_g,
        load_jungle_chess
    ]:
        dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns = load_set()

        deep_ordinal_encoder = DeepOrdinalEncoder(categorical_columns=categorical_columns)
        deep_ordinal_encoder.fit(X, y)
        X, y = deep_ordinal_encoder.transform(X, y)
        categorical_columns = deep_ordinal_encoder.transform_column_titles(categorical_columns)
        ordinal_columns = deep_ordinal_encoder.transform_column_titles(ordinal_columns)

        experiment_directory = os.path.join(os.getcwd(), 'experiments', 'runs')
        experiment_basename = 'exploration'
        is_classification_task = dataset_task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]
        encoder_list = get_encoder_list(categorical_columns=categorical_columns, ordinal_columns=ordinal_columns)
        scaler = RobustScaler()
        generator_list = get_generator_list(is_classification_task=is_classification_task)
        student = get_student(
            is_classification_task=is_classification_task,
            encoder=BinaryEncoder(cols=categorical_columns)
        )
        teacher = get_teacher(
            is_classification_task=is_classification_task
        )
        metric_list = get_metric_list(dataset_task)
        cv = 5
        train_size = 500
        n_samples_list = [
            0, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000, 10000
        ]
        random_state_list = [1, 2, 3, 4, 5]
        verbose = 100
        generator_timeout = 1800

        parallelized_run(
            experiment_directory=experiment_directory,
            experiment_basename=experiment_basename,
            is_classification_task=is_classification_task,
            dataset=(dataset_name, X, y, categorical_columns, ordinal_columns),
            encoder_list=encoder_list,
            scaler=scaler,
            generator_list=generator_list,
            student=student,
            teacher=teacher,
            metric_list=metric_list,
            cv=cv,
            train_size=train_size,
            n_samples_list=n_samples_list,
            random_state_list=random_state_list,
            verbose=verbose,
            generator_timeout=generator_timeout
        )


def test_parallelized_run():
    dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns = load_adult()

    deep_ordinal_encoder = DeepOrdinalEncoder(categorical_columns=categorical_columns)
    deep_ordinal_encoder.fit(X, y)
    X, y = deep_ordinal_encoder.transform(X, y)
    categorical_columns = deep_ordinal_encoder.transform_column_titles(categorical_columns)
    ordinal_columns = deep_ordinal_encoder.transform_column_titles(ordinal_columns)

    experiment_directory = os.path.join(os.getcwd(), 'experiments', 'tests')
    experiment_basename = 'exploration'
    is_classification_task = dataset_task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]
    encoder_list = get_test_encoder_list(categorical_columns=categorical_columns)
    scaler = RobustScaler()
    generator_list = get_test_generator_list(is_classification_task=is_classification_task)
    student = get_test_student(
        is_classification_task=is_classification_task,
        encoder=BinaryEncoder(cols=categorical_columns)
    )
    teacher = get_test_teacher(
        is_classification_task=is_classification_task
    )
    metric_list = get_test_metric_list(dataset_task)
    cv = 5
    train_size = 500
    n_samples_list = [0, 500, 1000]
    random_state_list = [1]
    verbose = 100
    generator_timeout = 1000

    parallelized_run(
        experiment_directory=experiment_directory,
        experiment_basename=experiment_basename,
        is_classification_task=is_classification_task,
        dataset=(dataset_name, X, y, categorical_columns, ordinal_columns),
        encoder_list=encoder_list,
        scaler=scaler,
        generator_list=generator_list,
        student=student,
        teacher=teacher,
        metric_list=metric_list,
        cv=cv,
        train_size=train_size,
        n_samples_list=n_samples_list,
        random_state_list=random_state_list,
        verbose=verbose,
        generator_timeout=generator_timeout
    )


if __name__ == '__main__':
    start_parallelized_run()
