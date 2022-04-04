from framework.pipelines import *

from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.preprocessing import RobustScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from framework.imputers import DropImputer
import framework.encoders as enc
from framework.transformers import DatasetCombiner, Labeler
from framework.samplers import *

imputers = [
    SimpleImputer(strategy='most_frequent'),
    DropImputer()
]

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

scalers = [
    RobustScaler
]

samplers = [
    ProportionalSMOTESampler,
    UnlabeledSMOTESampler,
    ProportionalRACOGSampler,
    UnlabeledRACOGSampler,
    ProportionalVanillaGANSampler,
    UnlabeledVanillaGANSampler,
    ProportionalConditionalGANSampler,
    UnlabeledConditionalGANSampler,
    ProportionalDRAGANSampler,
    UnlabeledDRAGANSampler
]

# pipeline = TeacherLabeledAugmentationPipeline(
#     imputer=SimpleImputer(strategy='most_frequent'),
#     encoder=ce.CatBoostEncoder(),
#     scaler=RobustScaler(),
#     sampler=ProportionalConditionalGANSampler(sample_multiplication_factor=1, epochs=1)
# )

adult = pd.read_csv('../data/adult.csv')
X = adult.drop(columns='income')
X = X.replace({'?': np.NaN})
y = adult['income'].map({'<=50K': 0, '>50K': 1})

# pipeline.fit_transform(X, y)

# =========================================================================== #

# X = X.head(50)
# y = y.head(50)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=42)

teacher_pipeline = RandomForestClassifierTeacherPipeline(
    imputer=SimpleImputer(strategy='most_frequent'),
    encoder=ce.TargetEncoder(),
    scaler=RobustScaler()
)
teacher_pipeline.fit(X_train, y_train)

print("test score: %s" % teacher_pipeline.score(X_test, y_test))

# random_forest_classifier = RandomForestClassifier()
# random_forest_classifier.fit(X, y)

# pipeline = TeacherLabeledAugmentationPipeline(
#     imputer=SimpleImputer(strategy='most_frequent'),
#     encoder=ce.TargetEncoder(),
#     scaler=RobustScaler(),
#     sampler=ProportionalSMOTESampler(sample_multiplication_factor=1),
#     teacher=Labeler(trained_model=random_forest_classifier),
#     combiner=DatasetCombiner(X=X, y=y),
#     student=DecisionTreeClassifier()
# )
#
# pipeline.fit_transform(X, y)