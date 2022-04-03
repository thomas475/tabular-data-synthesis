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

from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=42)

teacher_training_pipeline = TeacherTrainingPipeline(
    imputer=SimpleImputer(strategy='most_frequent'),
    encoder=ce.TargetEncoder(),
    scaler=RobustScaler(),
    teacher=RandomForestClassifier(random_state=42)
)

opt = BayesSearchCV(
    teacher_training_pipeline,
    {
        'teacher__n_estimators': (5,5000),
        'teacher__max_features': ['auto','sqrt'],
        'teacher__max_depth': (2,90),
        'teacher__min_samples_split': (2,10),
        'teacher__min_samples_leaf': (1,7),
        'teacher__bootstrap': ["True","False"]
    },
    n_iter=25,
    cv=5,
    scoring='roc_auc'
)
opt.fit(X_train, y_train)

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))

teacher_training_pipeline.fit(X, y)

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