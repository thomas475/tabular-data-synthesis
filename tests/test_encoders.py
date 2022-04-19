import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import tree

import category_encoders as ce
import framework.encoders as enc

def print_frame(X, y):
    frame = pd.DataFrame(X).copy().reset_index(drop=True)
    frame['target'] = pd.Series(y).copy().reset_index(drop=True)
    print(frame)

encoders = [
    # ce.BackwardDifferenceEncoder(),
    # ce.BaseNEncoder(),
    # ce.BinaryEncoder(),
    # ce.CatBoostEncoder(),
    # ce.CountEncoder(),
    # ce.GLMMEncoder(),
    # ce.HashingEncoder(),
    # ce.HelmertEncoder(),
    # ce.JamesSteinEncoder(),
    # ce.LeaveOneOutEncoder(),
    # ce.MEstimateEncoder(),
    # ce.OneHotEncoder(),
    # ce.OrdinalEncoder(),
    # ce.SumEncoder(),
    # ce.PolynomialEncoder(),
    # ce.TargetEncoder(),
    # ce.WOEEncoder(),
    # ce.QuantileEncoder(),
    # enc.TargetEncoder(),
    enc.CollapseEncoder()
]

adult = pd.read_csv('../data/adult.csv')
X = adult.drop(columns='income')
X = X.replace({'?': np.NaN})
y = adult['income'].map({'<=50K': 0, '>50K': 1})

start = 98
end = start + 5
X = X.loc[start:end - 1, :]
y = y[start:end]

print_frame(X, y)
for encoder in encoders:
    print(encoder)
    print_frame(encoder.fit_transform(X, y), y)
