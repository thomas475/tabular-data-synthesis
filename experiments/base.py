from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import category_encoders as ce
from skopt.space import Real, Categorical, Integer

from framework.scheduler import Scheduler
from framework.imputers import MostFrequentImputer
from framework.transformers import *
from framework.pipelines import *
from framework.samplers import *
import framework.encoders as enc

pipelines = [
    TeacherLabeledAugmentedStudentPipeline,
    IndirectGeneratorLabeledAugmentedStudentPipeline,
    DirectGeneratorLabeledAugmentedStudentPipeline
]

imputers = [
    MostFrequentImputer,
]

encoders = [
    ce.BackwardDifferenceEncoder,
    ce.BaseNEncoder,
    ce.BinaryEncoder,
    ce.CatBoostEncoder,
    ce.CountEncoder,
    ce.GLMMEncoder,
    ce.HashingEncoder,
    ce.HelmertEncoder,
    ce.JamesSteinEncoder,
    ce.LeaveOneOutEncoder,
    ce.MEstimateEncoder,
    ce.OneHotEncoder,
    ce.OrdinalEncoder,
    ce.SumEncoder,
    ce.PolynomialEncoder,
    ce.TargetEncoder,
    ce.WOEEncoder,
    ce.QuantileEncoder,
    enc.TargetEncoder,
    enc.CollapseEncoder
]

scalers = [
    RobustScaler
]

samplers = [
    (ProportionalSMOTESampler, {
        'sampler__k_neighbors': Integer(1, 7)
    }),
    (UnlabeledSMOTESampler, {
        'sampler__k_neighbors': Integer(1, 7)
    }),
    (ProportionalRACOGSampler, {
        'sampler__burnin': Integer(50, 250),
        'sampler__lag': Integer(1, 30)
    }),
    (UnlabeledRACOGSampler, {
        'sampler__burnin': Integer(50, 250),
        'sampler__lag': Integer(1, 30)
    }),
    (ProportionalVanillaGANSampler, {
        'sampler__epochs': [1],
        'sampler__batch_size': Integer(32, 256),
        'sampler__learning_rate': Real(0.00001, 0.01),
        'sampler__noise_dim': Integer(64, 512),
        'sampler__layers_dim': Integer(32, 256)
    }),
    (UnlabeledVanillaGANSampler, {
        'sampler__epochs': [1],
        'sampler__batch_size': Integer(32, 256),
        'sampler__learning_rate': Real(0.00001, 0.01),
        'sampler__noise_dim': Integer(64, 512),
        'sampler__layers_dim': Integer(32, 256)
    }),
    (ProportionalConditionalGANSampler, {
        'sampler__epochs': [1],
        'sampler__batch_size': Integer(32, 256),
        'sampler__learning_rate': Real(0.00001, 0.01),
        'sampler__noise_dim': Integer(64, 512),
        'sampler__layers_dim': Integer(32, 256)
    }),
    (UnlabeledConditionalGANSampler, {
        'sampler__epochs': [1],
        'sampler__batch_size': Integer(32, 256),
        'sampler__learning_rate': Real(0.00001, 0.01),
        'sampler__noise_dim': Integer(64, 512),
        'sampler__layers_dim': Integer(32, 256)
    }),
    (ProportionalDRAGANSampler, {
        'sampler__epochs': [1],
        'sampler__discriminator_updates_per_step': Integer(1, 5),
        'sampler__batch_size': Integer(32, 256),
        'sampler__learning_rate': Real(0.00001, 0.001),
        'sampler__noise_dim': Integer(64, 512),
        'sampler__layers_dim': Integer(32, 256)
    }),
    (UnlabeledDRAGANSampler, {
        'sampler__epochs': [1],
        'sampler__discriminator_updates_per_step': Integer(1, 5),
        'sampler__batch_size': Integer(32, 256),
        'sampler__learning_rate': Real(0.00001, 0.001),
        'sampler__noise_dim': Integer(64, 512),
        'sampler__layers_dim': Integer(32, 256)
    })
]

teachers = [
    (RandomForestClassifier, {
        'teacher__n_estimators': (25, 175),
        'teacher__max_features': ['auto', 'sqrt'],
        'teacher__max_depth': (15, 90),
        'teacher__min_samples_split': (2, 10),
        'teacher__min_samples_leaf': (1, 7),
        'teacher__bootstrap': ["True", "False"]
    })
]

labelers = [
    Labeler
]

injectors = [
    TargetInjector
]

extractors = [
    TargetExtractor
]

discretizers = [
    NumericalTargetDiscretizer
]

students = [
    (DecisionTreeClassifier, {
        "student__max_depth": Integer(1, 6),
        "student__criterion": Categorical(["gini", "entropy"])
    })
]

metrics = [
    'roc_auc'
]

sample_multiplication_factors = [
    0, 1, 5, 10, 20, 50, 100
]

n_iter = 50
n_points = 1
train_ratio = 0.75
cv = 10
n_jobs = 1
verbose = 100
random_state = 42

def load_datasets():
    datasets = []

    # load adult dataset
    adult = pd.read_csv('../data/adult.csv')

    # preprocess dataset
    X = adult.drop(columns='income')
    X = X.replace({'?': np.nan})
    X.columns = range(0, len(X.columns))

    # preprocess target
    y = adult['income'].map({'<=50K': 0, '>50K': 1})
    y.name = len(X.columns)

    # choose desired number of samples used from this dataset
    selected_n_samples = len(X)
    total_n_samples = min(selected_n_samples, len(X))

    # set number of samples, train size and test size so that they are divisible by cv
    train_size = int(((total_n_samples * train_ratio) // cv) * cv)
    test_size = int(((total_n_samples * (1 - train_ratio)) // cv) * cv)
    total_n_samples = train_size + test_size

    X = X.head(total_n_samples)
    y = y.head(total_n_samples)

    datasets.append(('adult', X, y, train_size, test_size))

    return datasets

def explore():
    scheduler = Scheduler(
        pipelines=pipelines,
        sample_multiplication_factors=sample_multiplication_factors,
        datasets=load_datasets(),
        imputers=imputers,
        encoders=encoders,
        scalers=scalers,
        samplers=samplers,
        teachers=teachers,
        labelers=labelers,
        injectors=injectors,
        extractors=extractors,
        discretizers=discretizers,
        students=students,
        metrics=metrics,
        n_iter=n_iter,
        n_points=n_points,
        train_ratio=train_ratio,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state
    )

    scheduler.explore()