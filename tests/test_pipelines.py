import pandas as pd

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

from skopt.space import Real, Categorical, Integer

search_spaces = {
    'decision_tree_classifier': {
        "student__max_depth": Integer(0, 6),
        "student__max_features": Integer(1, 9),
        "student__min_samples_leaf": Integer(1, 9),
        "student__criterion": Categorical(["gini", "entropy"])
    },
    'smote': {
        'sampler__k_neighbors': Integer(1, 7)
    },
    'racog': {
        'sampler__burnin': Integer(50, 250),
        'sampler__lag': Integer(1, 5)
    },
    'gan': {
        'sampler__batch_size': Integer(32, 256),
        'sampler__learning_rate': Real(0.0001, 0.1),
        'sampler__noise_dim': Integer(64, 512),
        'sampler__layers_dim': Integer(32, 256)
    }
}

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

imputer = SimpleImputer(strategy='most_frequent')
encoder = ce.TargetEncoder()
scaler = RobustScaler()
sampler = ProportionalRACOGSampler
labeler = Labeler
combiner = DatasetCombiner
student = DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=42)

preprocessor = PreprocessorPipeline(
    imputer=imputer,
    encoder=encoder,
    scaler=scaler,
)

X_train_preprocessed, y_train_preprocessed = preprocessor.fit_transform(X_train, y_train)
X_test_preprocessed, y_test_preprocessed = preprocessor.fit_transform(X_test, y_test)

teacher = RandomForestClassifierTeacherPipeline(
    imputer=imputer,
    encoder=encoder,
    scaler=scaler,
    n_iter=1,
    cv=2
)
teacher.fit(X_train, y_train)
print('teacher auc:', teacher.score(X_test, y_test))

from framework.pipelines import CustomPipeline

baseline_student = BaselineStudentPipeline(
    imputer=imputer,
    encoder=encoder,
    scaler=scaler,
    student=student(),
    search_spaces={
        **search_spaces['decision_tree_classifier']
    },
    n_iter=1,
    cv=2,
    scoring='roc_auc',
    n_jobs=1,
    random_state=42
)
baseline_student.fit(X_train, y_train)
print("baseline student auc:", baseline_student.score(X_test, y_test))

student = TeacherLabeledAugmentedStudentPipeline(
    imputer=imputer,
    encoder=encoder,
    scaler=scaler,
    sampler=sampler(sample_multiplication_factor=4),
    teacher=labeler(trained_model=teacher),
    combiner=combiner(X=X_train_preprocessed, y=y_train_preprocessed),
    student=student(),
    search_spaces={
        **search_spaces['decision_tree_classifier'],
        **search_spaces['racog']
    },
    n_iter=1,
    cv=2,
    scoring='roc_auc',
    n_jobs=1,
    random_state=42
)
student.fit(X_train, y_train)
student.predict(X_test)
print("student auc:", student.score(X_test, y_test))

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