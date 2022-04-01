from sklearn.pipeline import Pipeline
from transformers import *
import category_encoders as ce
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier

encoders = [
    ce.BackwardDifferenceEncoder(),
    ce.BaseNEncoder(),
    ce.BinaryEncoder(),
    ce.CatBoostEncoder(),
    ce.CountEncoder(),
    ce.GLMMEncoder(),
    ce.HashingEncoder(),
    ce.HelmertEncoder(),
    ce.JamesSteinEncoder(),
    ce.LeaveOneOutEncoder(),
    ce.MEstimateEncoder(),
    ce.OneHotEncoder(),
    ce.OrdinalEncoder(),
    ce.SumEncoder(),
    ce.PolynomialEncoder(),
    ce.TargetEncoder(),
    ce.WOEEncoder(),
    ce.QuantileEncoder()
]

scaler = RobustScaler()

samplers = [
    ProportionalSMOTETransformer(),
    UnlabeledSMOTETransformer(),
    ProportionalRACOGTransformer(),
    UnlabeledRACOGTransformer(),
    UnlabeledVanillaGANTransformer()
]
