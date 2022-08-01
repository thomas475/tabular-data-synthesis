import pandas as pd
from category_encoders import *
from framework.encoders import *
import copy
from experiments.datasets import load_adult
from sklearn.model_selection import train_test_split
import random
import os

from itertools import product


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))


name, task, X, y, cat, ordinal = load_adult()

dictionary = {
    'iterations': [10, 25, 50, 100, 250, 500]
}

print(list(product_dict(**dictionary)))
