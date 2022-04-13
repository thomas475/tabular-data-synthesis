import numpy as np
import pandas as pd

from framework.transformers import *

adult = pd.read_csv('../data/adult.csv')
X = adult.drop(columns='income')
X = X.replace({'?': np.NaN})
y = adult['income'].map({'<=50K': 0, '>50K': 1})

injector = TargetInjector()
extractor = TargetExtractor()

print(X)

dataset = injector.fit_transform(X=X, y=y)
print(dataset)

X, y = extractor.fit_transform(X=dataset)
print(X)
print(y)

X_1 = pd.DataFrame(X).copy().iloc[0:10,:]
X_2 = pd.DataFrame(X).copy().iloc[10:20,:]
X_2.columns = range(0, len(X_2.columns))
y_1 = pd.Series(y).copy()[0:10]
y_2 = pd.Series(y).copy()[10:20]
combiner = DatasetCombiner(X_1, y_1)
combined_X = combiner.fit_transform(X=X_2)
print(combined_X)
combined_X, combined_y = combiner.fit_transform(X=X_2, y=y_2)
print(combined_X)
print(combined_y)

X = [1, 2, 3, 4, 5]
y = [0, 1, 1, 0, 1]
continuous_y = [0.1, 1.3, 0.8, 0.4, 0.5]
discretizer = NumericalTargetDiscretizer(y)
print(discretizer.fit_transform(X, continuous_y))
print(discretizer.fit_transform(X, y))