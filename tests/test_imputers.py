import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

from framework.imputers import *

def drop_test_1():
    adult = pd.read_csv('../data/adult.csv')
    X = adult.drop(columns='income')
    X = X.replace({'?': np.NaN})
    y = adult['income'].map({'<=50K': 0, '>50K': 1})

    X1 = DropImputer().fit_transform(X)
    X2, y2 = DropImputer().fit_transform(X, y)

    assert len(X1) == len(X2)
    assert len(X2) == len(y2)

def drop_no_nan_test_1():
    adult = pd.read_csv('../data/adult.csv')
    X = adult.drop(columns='income')
    X = X.replace({'?': 1})
    y = adult['income'].map({'<=50K': 0, '>50K': 1})

    imputer = DropImputer()

    assert len(X) == len(imputer.fit_transform(X))

def drop_no_nan_test_2():
    adult = pd.read_csv('../data/adult.csv')
    X = adult.drop(columns='income')
    X = X.replace({'?': 1})
    y = adult['income'].map({'<=50K': 0, '>50K': 1})

    imputer = DropImputer()
    X_new, y_new = imputer.fit_transform(X, y)
    assert len(X) == len(X_new)
    assert len(y) == len(y_new)

def simple_test_1():
    adult = pd.read_csv('../data/adult.csv')
    X = adult.drop(columns='income')
    X = X.replace({'?': np.NaN})
    y = adult['income'].map({'<=50K': 0, '>50K': 1})

    SimpleImputer(strategy='most_frequent').fit_transform(X, y)

def simple_no_nan_test_1():
    adult = pd.read_csv('../data/adult.csv')
    X = adult.drop(columns='income')
    X = X.replace({'?': 1})
    y = adult['income'].map({'<=50K': 0, '>50K': 1})

    imputer = SimpleImputer(strategy='most_frequent')
    X_new = imputer.fit_transform(X, y)
    assert len(X) == len(X_new)

drop_test_1()
drop_no_nan_test_1()
drop_no_nan_test_2()
simple_test_1()
simple_no_nan_test_1()
