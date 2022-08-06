import pandas as pd
from category_encoders import *
from framework.encoders import *
import copy
from experiments.datasets import *
from sklearn.model_selection import train_test_split
import random
import os

from framework.samplers import *

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

for _, fold_index in StratifiedKFold(n_splits=n_splits, shuffle=False).split(X.head(total_selection_size), y.head(total_selection_size)):
    print(X.iloc[fold_index])
    print(y.iloc[fold_index].value_counts())