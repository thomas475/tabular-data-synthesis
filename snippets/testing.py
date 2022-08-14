import pandas as pd
from category_encoders import *
from numpy import zeros
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from framework.encoders import *
import copy
from experiments.datasets import *
from sklearn.model_selection import train_test_split
import random
import os

from framework.samplers import *

from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer


def get_relative_counts(column):
    unique, counts = np.unique(column, return_counts=True)

    count_list = list(zip(unique.T, counts.T))

    result = pd.DataFrame(columns=['value', 'count', 'proportion'])
    for value, count in count_list:
        result = result.append({
            'value': int(value),
            'count': int(count),
            'proportion': round(float(count) / float(len(column)), 3)
        }, ignore_index=True)

    return result


def run():
    for load_set in [load_adult, load_covertype]:
        dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns = load_set()

        deep_ordinal_encoder = DeepOrdinalEncoder(categorical_columns=categorical_columns)
        deep_ordinal_encoder.fit(X, y)
        X, y = deep_ordinal_encoder.transform(X, y)
        categorical_columns = deep_ordinal_encoder.transform_column_titles(categorical_columns)
        ordinal_columns = deep_ordinal_encoder.transform_column_titles(ordinal_columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2500, stratify=y)

        generator = ProportionalCWGANGPGenerator(batch_size=50, epochs=1)
        generator.fit(X_train, y_train, categorical_columns, ordinal_columns)
        X_sampled, y_sampled = generator.sample(5000)

        print(dataset_name)
        original_counts = get_relative_counts(y_train)
        sampled_counts = get_relative_counts(y_sampled)
        print(original_counts)
        print(sampled_counts)
        print(original_counts.compare(sampled_counts))

    from sklearn.model_selection import StratifiedKFold


def generator_fold_test():
    dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns = load_adult()

    deep_ordinal_encoder = DeepOrdinalEncoder(categorical_columns=categorical_columns)
    deep_ordinal_encoder.fit(X, y)
    X, y = deep_ordinal_encoder.transform(X, y)
    categorical_columns = deep_ordinal_encoder.transform_column_titles(categorical_columns)
    ordinal_columns = deep_ordinal_encoder.transform_column_titles(ordinal_columns)

    X = X.head(10499)
    y = y.head(10499)

    samples_per_fold = 500
    n_splits = int(len(X) / samples_per_fold)
    total_selection_size = n_splits * samples_per_fold

    print('totalselsize', total_selection_size)
    print('splits', n_splits)
    input()

    for _, fold_index in StratifiedKFold(n_splits=n_splits, shuffle=False).split(X.head(total_selection_size),
                                                                                 y.head(total_selection_size)):
        print(X.iloc[fold_index])
        print(y.iloc[fold_index].value_counts())


def get_encoder_list(categorical_columns, ordinal_columns):
    encoder_list = [
        BinaryEncoder(cols=categorical_columns),
        # CatBoostEncoder(cols=categorical_columns),
        # CountEncoder(cols=categorical_columns),
        # GLMMEncoder(cols=categorical_columns),
        # CV5GLMMEncoder(cols=categorical_columns),
        # OneHotEncoder(cols=categorical_columns),
        # TargetEncoder(cols=categorical_columns),
        # CV5TargetEncoder(cols=categorical_columns),
    ]

    if ordinal_columns:
        encoder_list.append(
            CollapseEncoder(cols=categorical_columns)
        )

    return encoder_list


def get_encoder_list(categorical_columns, ordinal_columns):
    if categorical_columns:
        encoder_list = [
            BinaryEncoder(cols=categorical_columns),
            CatBoostEncoder(cols=categorical_columns),
            CountEncoder(cols=categorical_columns),
            GLMMEncoder(cols=categorical_columns),
            StratifiedCV5GLMMEncoder(cols=categorical_columns),
            # OneHotEncoder(cols=categorical_columns),
            TargetEncoder(cols=categorical_columns),
            StratifiedCV5TargetEncoder(cols=categorical_columns),
        ]

        if ordinal_columns:
            encoder_list.append(
                CollapseEncoder(cols=categorical_columns)
            )
    else:
        encoder_list = []

    return encoder_list


def test_encoders():
    dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns = load_car()

    dataset = X.copy()
    dataset[y.name] = y
    print(dataset)
    print(categorical_columns)
    print(ordinal_columns)

    deep_ordinal_encoder = DeepOrdinalEncoder(categorical_columns=categorical_columns)
    deep_ordinal_encoder.fit(X, y)
    X, y = deep_ordinal_encoder.transform(X, y)
    categorical_columns = deep_ordinal_encoder.transform_column_titles(categorical_columns)
    ordinal_columns = deep_ordinal_encoder.transform_column_titles(ordinal_columns)

    dataset = X.copy()
    dataset[y.name] = y
    print(dataset)
    print(categorical_columns)
    print(ordinal_columns)

    for encoder in get_encoder_list(categorical_columns, ordinal_columns):
        scaler = RobustScaler()

        # apply the encoder on the categorical columns and the scaler on the numerical columns
        column_transformer = ColumnTransformer(
            [
                ("scaler", scaler, ordinal_columns.copy()),
                ("encoder", encoder, categorical_columns.copy())
            ]
        )

        original_column_order = X.columns

        index = X.index
        X_encoded = column_transformer.fit_transform(X.copy())
        X_encoded = pd.DataFrame(X_encoded)
        X_encoded.index = index

        # our dataset is completely numerical now, so we update the columns
        categorical_columns = []
        ordinal_columns = X.columns
        y_encoded = y.copy()
        y_encoded.name = len(ordinal_columns)

        # convert categorical numerical entries to int
        X_encoded[categorical_columns] = X_encoded[categorical_columns].astype(int)

        categorical_columns = []
        ordinal_columns = X.columns

        print(type(encoder).__name__, 'completed')
        dataset = X_encoded.copy()
        dataset[y_encoded.name] = y_encoded
        print(dataset)
        print(categorical_columns)
        print(ordinal_columns)


from sklearn.compose import ColumnTransformer
from framework.pipelines import AugmentedEstimation
from sklearn.tree import DecisionTreeRegressor

dataset_name, dataset_task, X, y, categorical_columns, ordinal_columns = load_connect_4()

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

encoder = BinaryEncoder(cols=categorical_columns)
scaler = RobustScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=500, test_size=500)

column_transformer = ColumnTransformer(
    [
        ("scaler", scaler, ordinal_columns.copy()),
        ("encoder", encoder, categorical_columns.copy())
    ]
)

original_column_order = list(X_train.columns)

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
ordinal_columns = list(X_train.columns)
y_train = y_train.copy()
y_train.name = len(ordinal_columns)
y_test = y_test.copy()
y_test.name = len(ordinal_columns)

generator = CopulaGANGenerator(is_classification_task=True, batch_size=500, epochs=1)
tuned_student = DecisionTreeRegressor(max_depth=5)
student_encoder = BinaryEncoder(cols=categorical_columns)

train_size = 500

# tune the generator
augmented_student = AugmentedEstimation(
    generator=generator,
    estimator=tuned_student,
    n_samples=train_size,
    categorical_columns=categorical_columns,
    ordinal_columns=ordinal_columns,
    encoder=student_encoder
)

# X_train[categorical_columns] = X_train[categorical_columns].astype('category')
# X_train[ordinal_columns] = X_train[ordinal_columns].astype('float')

augmented_student.fit(X_train, y_train)