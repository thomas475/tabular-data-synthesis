import category_encoders as ce
from sklearn.preprocessing import RobustScaler

import framework.encoders as enc

from framework.encoders import *
from framework.samplers import *

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
    ce.QuantileEncoder(),
    enc.TargetEncoder(),
    enc.CollapseEncoder()
]

scaler = RobustScaler()

samplers = [
    ProportionalSMOTESampler(),
    UnlabeledSMOTESampler(),
    ProportionalRACOGSampler(),
    UnlabeledRACOGSampler(),
    UnlabeledVanillaGANSampler()
]