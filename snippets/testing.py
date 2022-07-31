import pandas as pd
from category_encoders import *
from framework.encoders import *
import copy
from experiments.datasets import load_adult
from sklearn.model_selection import train_test_split
import random
import os

name, task, X, y, cat, ordinal = load_adult()

for random_state in [1, 2, 3, 4]:
    os.environ['PYTHONHASHSEED'] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10, stratify=y)

    print(X_train)
