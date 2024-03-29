import os
import random

import numpy as np
import pandas as pd

import torch
import tensorflow as tf

from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
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
from framework.encoders import CollapseEncoder, StratifiedCV5GLMMEncoder, StratifiedCV5TargetEncoder, StratifiedCVEncoder, StratifiedCVEncoderOriginal

from framework.pipelines import AugmentedEstimation

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

import signal
from contextlib import contextmanager

from experiments.exploration import *


def get_lgbm_scoring(scoring):
    def lgbm_scoring(y_true, y_pred):
        return scoring.__name__, scoring(y_true, np.round(y_pred)), True
    return lgbm_scoring


def get_test_teacher(is_classification_task):
    if is_classification_task:
        return (LGBMClassifier(), {
            'max_depth': [2],
            'n_estimators': [10]
        })
        # CatBoostClassifier(),
        #     {
        #         'depth': [4, 6, 8, 10],
        #         'iterations': [10, 25, 50]
        #     }
    else:
        return (LGBMRegressor(), {
            'max_depth': [2],
            'n_estimators': [10]
        })


def get_test_student(is_classification_task, encoder):
    if is_classification_task:
        return (DecisionTreeClassifier(), {
            'max_depth': [5],
            'criterion': ['gini']
        }, encoder)
    else:
        return (DecisionTreeRegressor(), {
            'max_depth': [5],
            'criterion': ['squared_error']
        }, encoder)


def get_test_encoder_list(categorical_columns):
    return [
        CountEncoder(cols=categorical_columns)
    ]


def get_test_metric_list(dataset_task):
    metric_list = []
    if dataset_task == BINARY_CLASSIFICATION:
        metric_list = [
            ('roc_auc', roc_auc_score, {})
        ]
    elif dataset_task == MULTICLASS_CLASSIFICATION:
        metric_list = [
            ('f1_macro', f1_score, {'average': 'macro'})
        ]
    elif dataset_task == REGRESSION:
        metric_list = [
            ('neg_mean_absolute_error', mean_absolute_error, {})
        ]
    return metric_list


def get_test_generator_list(is_classification_task):
    generator_list = [
        # (TableGANGenerator(is_classification_task=is_classification_task), {
        #     'batch_size': [500],  # default 500
        #     'epochs': [1],  # default 300
        # })
    ]
    if is_classification_task:
        generator_list.append(
            (ProportionalSMOTEGenerator(is_classification_task=is_classification_task), {
                'k_neighbors': [5]  # default 5
            })
        )
    else:
        generator_list.append(
            (WGANGPGenerator(is_classification_task=is_classification_task), {
                'batch_size': [500],  # default 128
                'epochs': [1],  # default 300
                # 'learning_rate': default_learning_rates
            }),
        )
    return generator_list


def get_fast_tuned_generators(is_classification_task):
    default_batch_sizes = [500]

    generator_list = [
        (PrivBNGenerator(is_classification_task=is_classification_task), {
            # a noisy distribution is θ-useful if the ratio of average scale of information to average scale of noise is no less than θ
            # in a k-degree bayesian network theta as a parameter is fixed and a corresponding k is calculated
            'theta': [20]  # default 20
        }),
        (GaussianCopulaGenerator(is_classification_task=is_classification_task), {
            'default_distribution': [  # default 'parametric'
                'univariate'
            ]
        }),
        (TableGANGenerator(is_classification_task=is_classification_task), {
            'batch_size': default_batch_sizes,  # default 500
            'epochs': [2],  # default 300
            # 'l2scale': default_l2_scales
        }),
        (CTGANGenerator(is_classification_task=is_classification_task), {
            'batch_size': default_batch_sizes,  # default 500
            'epochs': [2],  # default 300
            # 'generator_lr': default_generator_learning_rates,
            # 'discriminator_lr': default_discriminator_learning_rates
        }),
        (CopulaGANGenerator(is_classification_task=is_classification_task), {
            'batch_size': default_batch_sizes,  # default 500
            'epochs': [2],  # default 300
            # 'generator_lr': default_generator_learning_rates,
            # 'discriminator_lr': default_discriminator_learning_rates
        }),
        (TVAEGenerator(is_classification_task=is_classification_task), {
            'batch_size': default_batch_sizes,  # default 500
            'epochs': [2],  # default 300
            # 'l2scale': default_l2_scales
        }),
        (MedGANGenerator(is_classification_task=is_classification_task), {
            'batch_size': default_batch_sizes,  # default 1000
            'epochs': [2],  # default 2000
            # 'l2scale': default_l2_scales
        }),
        (DPCTGANGenerator(is_classification_task=is_classification_task), {
            'batch_size': [50],  # default 500
            'epochs': [2],  # default 300
            # 'generator_lr': default_generator_learning_rates,
            # 'discriminator_lr': default_discriminator_learning_rates
        }),
        (CTABGANGenerator(is_classification_task=is_classification_task), {
            'batch_size': default_batch_sizes,  # default 500
            'epochs': [1],  # default 1
            # 'l2scale': default_l2_scales
        })
    ]
    if is_classification_task:
        generator_list.append(
            (ProportionalSMOTEGenerator(is_classification_task=is_classification_task), {
                'k_neighbors': [5]  # default 5
            })
        )
        generator_list.append(
            (ProportionalCWGANGPGenerator(is_classification_task=is_classification_task), {
                'batch_size': default_batch_sizes,  # default 128
                'epochs': [2],  # default 300
                # 'learning_rate': default_learning_rates
            }),
        )
    else:
        generator_list.append(
            (WGANGPGenerator(is_classification_task=is_classification_task), {
                'batch_size': default_batch_sizes,  # default 128
                'epochs': [2],  # default 300
                # 'learning_rate': default_learning_rates
            }),
        )
    return generator_list


def test_parallelized_run():
    for load_set in [
        # load_adult,
        # load_amazon, # dont use
        # load_bank_marketing,
        # load_census_income,
        # load_credit_approval,
        # load_electricity,
        # load_higgs,
        load_ibm_employee_performance,
        # load_kr_vs_kp,
        load_national_longitudinal_survey_binary,
        load_monks_problems_2,
        load_mushroom,
        load_tic_tac_toe,
        # load_car,
        # load_cmc,
        # load_connect_4,
        # load_covertype,
        # load_credit_g,
        # load_jungle_chess,
        # load_splice,
        # load_vowel,
        # load_kaggle_30_days_of_ml,
        # load_brazilian_houses,
        # load_california,
        # load_diamonds,
        # load_king,
    ]:
        dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns = load_set()

        experiment_directory = os.path.join(os.getcwd(), 'experiments', 'tests')
        experiment_basename = 'exploration_' + dataset_task
        is_classification_task = dataset_task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]

        deep_ordinal_encoder = DeepOrdinalEncoder(
            categorical_columns=categorical_columns,
            discrete_target=is_classification_task
        )
        deep_ordinal_encoder.fit(X, y)
        X, y = deep_ordinal_encoder.transform(X, y)
        categorical_columns = deep_ordinal_encoder.transform_column_titles(categorical_columns)
        ordinal_columns = deep_ordinal_encoder.transform_column_titles(ordinal_columns)

        encoder_list = get_encoder_list(
            task=dataset_task,
            categorical_columns=categorical_columns,
            ordinal_columns=ordinal_columns
        )
        scaler = RobustScaler()
        generator_list = get_fast_tuned_generators(is_classification_task=is_classification_task)
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
        max_test_size = 500
        n_samples_list = [
            0, 500, 1000
        ]
        random_state_list = [1]
        verbose = 100
        generator_timeout = 600

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
            max_test_size=max_test_size,
            n_samples_list=n_samples_list,
            random_state_list=random_state_list,
            verbose=verbose,
            generator_timeout=generator_timeout
        )


if __name__ == '__main__':
    test_parallelized_run()


# ENCODERS = [
#     BinaryEncoder,
#     CatBoostEncoder,
#     CollapseEncoder,
#     CountEncoder,
#     GLMMEncoder,
#     CVEncoder(base_encoder=GLMMEncoder),
#     OneHotEncoder,
#     TargetEncoder,
#     CVEncoder(base_encoder=TargetEncoder)
# ]

# def run_scaler_test():
#     dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns = load_adult()
#
#     X = X.head(10)
#     y = y.head(10)
#
#     dataset = X.copy()
#     dataset[y.name] = y
#     print('categorical')
#     print(categorical_columns)
#     print(dataset[categorical_columns])
#     print('ordinal')
#     print(ordinal_columns)
#     print(dataset[ordinal_columns])
#
#     column_transformer = ColumnTransformer(
#         [
#             ("scaler", RobustScaler(), ordinal_columns.copy())
#         ],
#         remainder='passthrough'
#     )
#
#     dataset = pd.DataFrame(column_transformer.fit_transform(X, y))
#     dataset.columns = ordinal_columns + categorical_columns
#     dataset.index = X.index
#     print('categorical')
#     print(categorical_columns)
#     print(dataset[categorical_columns])
#     print('ordinal')
#     print(ordinal_columns)
#     print(dataset[ordinal_columns])
#
#     deep_ordinal_encoder = DeepOrdinalEncoder(categorical_columns=categorical_columns)
#     deep_ordinal_encoder.fit(X, y)
#     X, y = deep_ordinal_encoder.transform(X, y)
#     categorical_columns = deep_ordinal_encoder.transform_column_titles(categorical_columns)
#     ordinal_columns = deep_ordinal_encoder.transform_column_titles(ordinal_columns)
#
#     dataset = X.copy()
#     print('categorical')
#     print(categorical_columns)
#     print(dataset[categorical_columns])
#     print('ordinal')
#     print(ordinal_columns)
#     print(dataset[ordinal_columns])
#
#     column_transformer = ColumnTransformer(
#         [
#             ("scaler", RobustScaler(), ordinal_columns.copy())
#         ],
#         remainder='passthrough'
#     )
#
#     dataset = pd.DataFrame(column_transformer.fit_transform(X, y))
#     dataset.columns = ordinal_columns + categorical_columns
#     dataset.index = X.index
#
#     print('categorical')
#     print(categorical_columns)
#     print(dataset[categorical_columns])
#     print('ordinal')
#     print(ordinal_columns)
#     print(dataset[ordinal_columns])
#
#
# def run_sampler_tuning_test():
#     dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns = load_jungle_chess()
#     metric_name, metric_function, metric_parameters = ('accuracy', accuracy_score, {})
#
#     generator_grid = {
#         'theta': [15, 20, 30]
#     }
#     student_grid = {
#         'max_depth': [4, 5, 6, 7]
#     }
#
#     deep_ordinal_encoder = DeepOrdinalEncoder(categorical_columns=categorical_columns)
#     deep_ordinal_encoder.fit(X, y)
#     X, y = deep_ordinal_encoder.transform(X, y)
#     categorical_columns = deep_ordinal_encoder.transform_column_titles(categorical_columns)
#     ordinal_columns = deep_ordinal_encoder.transform_column_titles(ordinal_columns)
#
#     encoder = TargetEncoder(cols=categorical_columns)
#     scaler = RobustScaler()
#     sampler = PrivBNGenerator(is_classification_task=True)
#     student_encoder = BinaryEncoder(cols=categorical_columns)
#     student = DecisionTreeClassifier()
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=500, stratify=y)
#
#     column_transformer = ColumnTransformer(
#         [
#             ("encoder", encoder, categorical_columns),
#             ("scaler", scaler, ordinal_columns)
#         ]
#     )
#
#     # X_train_encoded = column_transformer.fit_transform(X_train, y_train)
#     # X_train_encoded = pd.DataFrame(X_train_encoded)
#     # X_train_encoded.columns = X_train.columns
#     # X_train_encoded.index = X_train.index
#     #
#     # X_test_encoded = column_transformer.transform(X_test)
#     # X_test_encoded = pd.DataFrame(X_test_encoded)
#     # X_test_encoded.columns = X_test.columns
#     # X_test_encoded.index = X_test.index
#
#     augmented_classification = AugmentedEstimation(
#         generator=sampler,
#         estimator=student,
#         n_samples=500,
#         categorical_columns=categorical_columns,
#         ordinal_columns=ordinal_columns,
#         encoder=student_encoder
#     )
#
#     tuned_student = GridSearchCV(
#         estimator=augmented_classification,
#         param_grid=join_grids([('generator', generator_grid), ('estimator', student_grid)]),
#         scoring=metric_name,
#         error_score="raise",
#         refit=True,
#         cv=None,
#         verbose=100
#     )
#     tuned_student.fit(X_train, y_train)
#     print(tuned_student.score(X_test, y_test))
#     print(metric_function(
#         y_test,
#         tuned_student.predict(X_test),
#         **metric_parameters
#     ))
#
#
# def run_deep_ordinal_encoder_test():
#     dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns = load_adult()
#
#     original_column_titles = categorical_columns.copy()
#
#     dataset = X.copy()
#     dataset[y.name] = y
#     print(original_column_titles)
#     print(dataset)
#
#     deep_ordinal_encoder = DeepOrdinalEncoder(categorical_columns=categorical_columns)
#     deep_ordinal_encoder.fit(X, y)
#     X, y = deep_ordinal_encoder.transform(X, y)
#     categorical_columns = deep_ordinal_encoder.transform_column_titles(categorical_columns)
#     ordinal_columns = deep_ordinal_encoder.transform_column_titles(ordinal_columns)
#
#     dataset = X.copy()
#     dataset[y.name] = y
#     print(categorical_columns)
#     print(dataset)
#
#     X, y = deep_ordinal_encoder.inverse_transform(X, y)
#
#     dataset = X.copy()
#     dataset[y.name] = y
#     print(original_column_titles)
#     print(dataset)
#
#
# def run_encoder_test():
#     dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns = load_adult()
#
#     deep_ordinal_encoder = DeepOrdinalEncoder(categorical_columns=categorical_columns)
#     deep_ordinal_encoder.fit(X, y)
#     X, y = deep_ordinal_encoder.transform(X, y)
#     categorical_columns = deep_ordinal_encoder.transform_column_titles(categorical_columns)
#     ordinal_columns = deep_ordinal_encoder.transform_column_titles(ordinal_columns)
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=500, stratify=y)
#
#     ENCODERS = [
#         BinaryEncoder(cols=categorical_columns),
#         CatBoostEncoder(cols=categorical_columns),
#         CollapseEncoder(cols=categorical_columns),
#         CountEncoder(cols=categorical_columns),
#         GLMMEncoder(cols=categorical_columns),
#         # CVEncoderOriginal(base_encoder=GLMMEncoder(cols=categorical_columns, handle_missing='return_nan', handle_unknown='return_nan')),
#         CVEncoder(base_encoder=GLMMEncoder, cols=categorical_columns),
#         OneHotEncoder(cols=categorical_columns),
#         TargetEncoder(cols=categorical_columns),
#         # CVEncoderOriginal(base_encoder=TargetEncoder(cols=categorical_columns, handle_missing='return_nan', handle_unknown='return_nan')),
#         CVEncoder(base_encoder=TargetEncoder, cols=categorical_columns)
#     ]
#
#     for encoder in ENCODERS:
#         column_transformer = ColumnTransformer(
#             [
#                 ("encoder", encoder, categorical_columns),
#                 ("scaler", RobustScaler(), ordinal_columns)
#             ]
#         )
#
#         X_train_encoded = column_transformer.fit_transform(X_train, y_train)
#         print(pd.DataFrame(X_train_encoded))
#
#         X_test_encoded = column_transformer.transform(X_test.copy())
#         print(pd.DataFrame(X_test_encoded))
#
# def run_sampler_test():
#     dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns = load_adult()
#
#     deep_ordinal_encoder = DeepOrdinalEncoder(categorical_columns=categorical_columns)
#     deep_ordinal_encoder.fit(X, y)
#     X, y = deep_ordinal_encoder.transform(X, y)
#     categorical_columns = deep_ordinal_encoder.transform_column_titles(categorical_columns)
#     ordinal_columns = deep_ordinal_encoder.transform_column_titles(ordinal_columns)
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=500, stratify=y)
#
#     X_decoded, y_decoded = deep_ordinal_encoder.inverse_transform(X_train, y_train)
#     data = X_decoded
#     data[y_decoded.name] = y_decoded
#     print(data)
#
#     batch_size = 50
#     epochs = 10
#     for generator in [
#         PrivBNGenerator(),
#         SMOTEGenerator(),
#         GaussianCopulaGenerator(),
#         TableGANGenerator(batch_size=batch_size, epochs=epochs),
#         MedGANGenerator(batch_size=batch_size, epochs=epochs),
#         DPCTGANGenerator(batch_size=batch_size, epochs=epochs),
#         WGANGPGenerator(batch_size=batch_size, epochs=epochs),
#         CWGANGPGenerator(batch_size=batch_size, epochs=epochs),
#         CopulaGANGenerator(batch_size=batch_size, epochs=epochs),
#         CTABGANGenerator(batch_size=batch_size, epochs=epochs),
#         CTGANGenerator(batch_size=batch_size, epochs=epochs),
#         TVAEGenerator(batch_size=batch_size, epochs=epochs)
#     ]:
#         generator.fit(X_train, y_train, categorical_columns=categorical_columns, ordinal_columns=ordinal_columns)
#         X_sampled, y_sampled = generator.sample(n=500)
#
#         print(generator.get_name())
#
#         # data_sampled = X_sampled.copy()
#         # data_sampled[target_attribute] = y_sampled
#         # print(data_sampled)
#
#         X_sampled_decoded, y_sampled_decoded = deep_ordinal_encoder.inverse_transform(X_sampled, y_sampled)
#         data_sampled = X_sampled_decoded
#         data_sampled[y_decoded.name] = y_sampled_decoded
#         print(data_sampled)
#
#         # with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
#         #     print(data_sampled.head())


# def run(
#         classification_task,
#         dataset,
#         encoder,
#         scaler,
#         generator,
#         student,
#         teacher,
#         metric,
#         train_size,
#         n_samples_list,
#         random_state
# ):
#     tf.random.set_seed(random_state)
#     torch.manual_seed(random_state)
#     os.environ['PYTHONHASHSEED'] = str(random_state)
#     random.seed(random_state)
#     np.random.seed(random_state)
#
#     dataset_name, X, y, categorical_columns, ordinal_columns = dataset
#     teacher, teacher_grid = teacher
#     generator, generator_grid = generator
#     student, student_grid, student_encoder = student
#     metric_name, metric_function, metric_parameters = metric
#
#     if classification_task:
#         X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y)
#     else:
#         X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
#
#     # apply encoder/scaler
#     if encoder is None:
#         # if no encoder is submitted we only apply the scaler on the numerical columns
#         column_transformer = ColumnTransformer(
#             [
#                 ("scaler", scaler, ordinal_columns.copy())
#             ],
#             remainder='passthrough'
#         )
#     else:
#         # apply the encoder on the categorical columns and the scaler on the numerical columns
#         column_transformer = ColumnTransformer(
#             [
#                 ("scaler", scaler, ordinal_columns.copy()),
#                 ("encoder", encoder, categorical_columns.copy())
#             ]
#         )
#         # our dataset is completely numerical now, so we update the columns
#         categorical_columns = []
#         ordinal_columns = X_train.columns
#
#     original_column_order = X.columns
#
#     index = X_train.index
#     X_train = column_transformer.fit_transform(X_train, y_train)
#     X_train = pd.DataFrame(X_train)
#     X_train.columns = ordinal_columns + categorical_columns
#     X_train.index = index
#     X_train = X_train[original_column_order]
#
#     index = X_test.index
#     X_test = column_transformer.transform(X_test)
#     X_test = pd.DataFrame(X_test)
#     X_test.columns = ordinal_columns + categorical_columns
#     X_test.index = index
#     X_test = X_test[original_column_order]
#
#     # convert categorical numerical entries to int
#     X_train[categorical_columns] = X_train[categorical_columns].astype(int)
#     X_test[categorical_columns] = X_test[categorical_columns].astype(int)
#
#     # tune the teacher
#     tuned_teacher = GridSearchCV(
#         estimator=teacher,
#         param_grid=teacher_grid,
#         scoring=metric_name,
#         error_score="raise",
#         refit=True,
#         cv=None,
#         verbose=10
#     )
#     tuned_teacher.fit(X=X_train.copy(), y=y_train.copy(), cat_features=categorical_columns)
#
#     # tune the generator/student
#     if encoder is None:
#         # because our dataset is not completely numerical, we apply the encoder submitted with the student if needed
#         augmented_student = AugmentedEstimation(
#             generator=generator,
#             estimator=student,
#             n_samples=train_size,
#             categorical_columns=categorical_columns,
#             ordinal_columns=ordinal_columns,
#             encoder=student_encoder
#         )
#     else:
#         augmented_student = AugmentedEstimation(
#             generator=generator,
#             estimator=student,
#             n_samples=train_size,
#             categorical_columns=categorical_columns,
#             ordinal_columns=ordinal_columns
#         )
#
#     tuned_augmented_student = GridSearchCV(
#         estimator=augmented_student,
#         param_grid=join_grids([('generator', generator_grid), ('estimator', student_grid)]),
#         scoring=metric_name,
#         error_score="raise",
#         refit=True,
#         cv=None,
#         verbose=10
#     )
#     tuned_augmented_student.fit(X=X_train.copy(), y=y_train.copy())
#
#     # get machine learning efficacy
#     performance = metric_function(
#         y_test,
#         tuned_augmented_student.predict(X_test),
#         **metric_parameters
#     )
#
#     tuned_generator = tuned_augmented_student.best_estimator_.get_generator()
#     tuned_student = tuned_augmented_student.best_estimator_.get_estimator()
#
#     X_sampled, y_sampled = tuned_generator.sample(train_size)
#
#     real_dataset = X_train.copy()
#     real_dataset[y_train.name] = y_train.copy()
#     synthetic_dataset = X_sampled.copy()
#     synthetic_dataset[y_sampled.name] = y_sampled.copy()
#
#     if classification_task:
#         complete_categorical_columns = categorical_columns.copy() + [y.name]
#         complete_ordinal_columns = ordinal_columns.copy()
#     else:
#         complete_categorical_columns = categorical_columns.copy()
#         complete_ordinal_columns = ordinal_columns.copy() + [y.name]
#
#     categorical_real_dataset = real_dataset[complete_categorical_columns]
#     categorical_synthetic_dataset = synthetic_dataset[complete_categorical_columns]
#     ordinal_real_dataset = real_dataset[ordinal_columns]
#     ordinal_synthetic_dataset = synthetic_dataset[ordinal_columns]
#
#     # statistical evaluation
#     kstest_performance = KSTest.compute(
#         real_data=ordinal_real_dataset,
#         synthetic_data=ordinal_synthetic_dataset
#     )
#     print('kstest', str(kstest_performance))
#     cstest_performance = CSTest.compute(
#         real_data=categorical_real_dataset.astype('category'),
#         synthetic_data=categorical_synthetic_dataset.astype('category')
#     )
#     print('cstest', str(cstest_performance))
#     jsd_performance, wd_performance = calculate_jsd_wd(
#         real_dataset.reset_index(drop=True),
#         synthetic_dataset.reset_index(drop=True),
#         complete_categorical_columns
#     )
#     print('jsd', str(jsd_performance))
#     print('wd', str(wd_performance))
#     real_corr = compute_associations(real_dataset, nominal_columns=categorical_columns)
#     synthetic_corr = compute_associations(synthetic_dataset, nominal_columns=categorical_columns)
#     diff_corr_performance = np.linalg.norm(real_corr - synthetic_corr)
#     print('diff_cor_performance', diff_corr_performance)
#
#     # privacy evaluation
#     dcr_performance, nndr_performance = calculate_dcr_nndr(real_dataset, synthetic_dataset)
#     print('dcr', dcr_performance)
#     print('nndr', nndr_performance)
#
#     result = {
#         'dataset': dataset_name,
#         'generator': type(generator).__name__,
#         'encoder': type(encoder).__name__,
#         'scaler': type(scaler).__name__,
#         'teacher': type(teacher).__name__,
#         'student': type(student).__name__,
#         'student_encoder': type(student_encoder).__name__,
#         'optimization_metric': metric_name,
#         'performance': str(performance),
#         'kstest': str(kstest_performance),
#         'cstest': str(cstest_performance),
#         'jsd': str(jsd_performance),
#         'wd': str(wd_performance),
#         'diff_corr': str(diff_corr_performance),
#         'dcr': str(dcr_performance),
#         'nndr': str(nndr_performance),
#         'n_samples': train_size,
#         'train_size': str(train_size),
#         'random_state': str(random_state)
#     }
#
#     results = pd.DataFrame()
#     results = results.append(result, ignore_index=True)
#
#     # avoid augmenting the student with the same number of samples again
#     if train_size in n_samples_list:
#         n_samples_list = n_samples_list.copy()
#         n_samples_list.remove(train_size)
#
#     # generate the maximum number of samples defined in n_samples_list using the tuned generator, so we can subsample
#     # from it instead of generating again
#     X_sampled_max, y_sampled_max = tuned_generator.sample(max(n_samples_list))
#
#     # generate the different amount of samples with the tuned generator and evaluate them on the tuned student
#     for n_samples in n_samples_list:
#         X_sampled = X_sampled_max.head(n_samples)
#         y_sampled = y_sampled_max.head(n_samples)
#
#         X_augmented = pd.concat([X_train, X_sampled], ignore_index=True).reset_index(drop=True)
#         y_augmented = pd.concat([y_train, y_sampled], ignore_index=True).reset_index(drop=True)
#
#         if student_encoder is not None and categorical_columns:
#             tuned_student.fit(student_encoder.fit_transform(X_augmented), y_augmented)
#             performance = metric_function(
#                 y_test,
#                 tuned_student.predict(student_encoder.transform(X_test)),
#                 **metric_parameters
#             )
#         else:
#             tuned_student.fit(X_augmented, y_augmented)
#             performance = metric_function(
#                 y_test,
#                 tuned_student.predict(X_test),
#                 **metric_parameters
#             )
#         print('performance', str(performance))
#
#         result = {
#             'dataset': dataset_name,
#             'generator': type(generator).__name__,
#             'encoder': type(encoder).__name__,
#             'scaler': type(scaler).__name__,
#             'teacher': type(teacher).__name__,
#             'student': type(student).__name__,
#             'student_encoder': type(student_encoder).__name__,
#             'optimization_metric': metric_name,
#             'performance': str(performance),
#             'n_samples': str(n_samples),
#             'train_size': str(train_size),
#             'random_state': str(random_state)
#         }
#
#         results = results.append(result, ignore_index=True)
#
#     print(results[['n_samples', 'performance', 'kstest', 'cstest', 'diff_corr', 'dcr', 'nndr']])


# BINARY_CLASSIFICATION_METRICS = [
#     ('accuracy', accuracy_score, {}),
#     ('f1', f1_score, {}),
#     ('roc_auc', roc_auc_score, {}),
# ]
#
# MULTICLASS_CLASSIFICATION_METRICS = [
#     ('accuracy', accuracy_score, {}),
#     ('f1_macro', f1_score, {'average': 'macro'})
# ]
#
# REGRESSION_METRICS = [
#     ('neg_mean_absolute_error', mean_absolute_error, {}),
#     ('r2', r2_score, {})
# ]


# CLASSIFICATION_GENERATORS = [
#     (PrivBNGenerator(is_classification_task=True), {
#         # a noisy distribution is θ-useful if the ratio of average scale of information to average scale of noise is no less than θ
#         # in a k-degree bayesian network theta as a parameter is fixed and a corresponding k is calculated
#         'theta': [10, 15, 20, 30]
#     }),
#     (SMOTEGenerator(is_classification_task=True), {
#         'k_neighbors': [4, 5, 6, 7] # default 5
#     }),
#     (GaussianCopulaGenerator(is_classification_task=True), {
#         'default_distribution': [ # default 'parametric'
#             'univariate', # Let ``copulas`` select the optimal univariate distribution. This may result in non-parametric models being used.
#             'parametric', # Let ``copulas`` select the optimal univariate distribution, but restrict the selection to parametric distributions only.
#             'bounded', # Let ``copulas`` select the optimal univariate distribution, but restrict the selection to bounded distributions only.
#         ]
#     }),
#     (TableGANGenerator(is_classification_task=True), {
#         'batch_size': [10, 20, 50], # default 500
#         'epochs': [1] # default 300
#     }),
#     (CTGANGenerator(is_classification_task=True), {
#         'batch_size': [10, 20, 50], # default 500
#         'epochs': [1] # default 300
#     }),
#     (CopulaGANGenerator(is_classification_task=True), {
#         'batch_size': [10, 20, 50], # default 500
#         'epochs': [1] # default 300
#     }),
#     (TVAEGenerator(is_classification_task=True), {
#         'batch_size': [10, 20, 50], # default 500
#         'epochs': [1] # default 300
#     }),
#     (CWGANGPGenerator(is_classification_task=True), {
#         'batch_size': [10, 20, 50], # default 128
#         'epochs': [1] # default 300
#     }),
#     (MedGANGenerator(is_classification_task=True), {
#         'batch_size': [10, 20, 50], # default 1000
#         'epochs': [1] # default 2000
#     }),
#     (DPCTGANGenerator(is_classification_task=True), {
#         'batch_size': [10, 20, 50], # default 500
#         'epochs': [10] # default 300
#     }),
#     (CTABGANGenerator(is_classification_task=True), {
#         'batch_size': [10, 20, 50], # default 500
#         'epochs': [1] # default 1
#     })
# ]


# def testing():
#     load_adult()
#     load_amazon()
#     load_census_income()
#     load_electricity()
#     load_higgs()
#
#     load_covertype()
#     load_credit_g()
#     load_jungle_chess()
#
#     load_california()
#     load_diamonds()
#     load_king()
#
#     dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns = load_adult()
#
#     deep_ordinal_encoder = DeepOrdinalEncoder(categorical_columns=categorical_columns)
#     deep_ordinal_encoder.fit(X, y)
#     X, y = deep_ordinal_encoder.transform(X, y)
#     categorical_columns = deep_ordinal_encoder.transform_column_titles(categorical_columns)
#     ordinal_columns = deep_ordinal_encoder.transform_column_titles(ordinal_columns)
#
#     encoder = None
#     encoder = CVEncoder(TargetEncoder, cols=categorical_columns)
#     categorical_columns = []
#     ordinal_columns = X.columns
#
#     scaler = RobustScaler()
#     generator = CLASSIFICATION_GENERATORS[5]
#     student = (DecisionTreeClassifier(), {'max_depth': [7]}, BinaryEncoder(cols=categorical_columns))
#     teacher = (
#         CatBoostClassifier(),
#         {
#             'depth': [4, 6, 8, 10],
#             'iterations': [1]
#             # 'iterations': [10, 25, 50]
#         }
#     )
#     metric = BINARY_CLASSIFICATION_METRICS[1]
#
#     run(
#         classification_task=True,
#         dataset=(dataset_name, X, y, categorical_columns, ordinal_columns),
#         encoder=encoder,
#         scaler=scaler,
#         generator=generator,
#         student=student,
#         teacher=teacher,
#         metric=metric,
#         train_size=500,
#         n_samples_list=[0, 500, 1000, 2000, 5000, 7500, 10000],
#         random_state=42
#     )
#     run_encoder_test()
#     run_sampler_test()
#     run_sampler_tuning_test()
#     run_deep_ordinal_encoder_test()
#     run_scaler_test()

# {'depth':[3,1,2,6,4,5,7,8,9,10],
# 'iterations':[250,100,500,1000],
# 'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3],
# 'l2_leaf_reg':[3,1,5,10,100],
# 'border_count':[32,5,10,20,50,100,200],
# 'ctr_border_count':[50,5,10,20,100,200]}
