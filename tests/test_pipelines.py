import pandas as pd

from framework.pipelines import *

from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.preprocessing import RobustScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from framework.imputers import DropImputer
import framework.encoders as enc
from framework.transformers import *
from framework.samplers import *

from skopt.space import Real, Categorical, Integer

search_spaces = {
    'random_forest_classifier': {
        'teacher__n_estimators': (25, 175),
        'teacher__max_features': ['auto', 'sqrt'],
        'teacher__max_depth': (15, 90),
        'teacher__min_samples_split': (2, 10),
        'teacher__min_samples_leaf': (1, 7),
        'teacher__bootstrap': ["True", "False"]
    },
    'decision_tree_classifier': {
        "student__max_depth": Integer(1, 6),
        "student__criterion": Categorical(["gini", "entropy"])
    },
    'smote': {
        'sampler__k_neighbors': Integer(1, 7)
    },
    'racog': {
        'sampler__burnin': Integer(50, 250),
        'sampler__lag': Integer(1, 30)
    },
    'vanilla_gan': {
        'sampler__epochs': Integer(1, 2), # fix
        'sampler__batch_size': Integer(32, 256),
        'sampler__learning_rate': Real(0.00001, 0.01),
        'sampler__noise_dim': Integer(64, 512),
        'sampler__layers_dim': Integer(32, 256)
    },
    'conditional_gan': {
        'sampler__epochs': Integer(1, 2), # fix
        'sampler__batch_size': Integer(32, 256),
        'sampler__learning_rate': Real(0.00001, 0.01),
        'sampler__noise_dim': Integer(64, 512),
        'sampler__layers_dim': Integer(32, 256)
    },
    'dragan': {
        'sampler__epochs': Integer(1, 2), # fix
        'sampler__discriminator_updates_per_step': Integer(1, 5),
        'sampler__batch_size': Integer(32, 256),
        'sampler__learning_rate': Real(0.00001, 0.001),
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


def run_teacher_pipeline_test():
    # pipeline = TeacherLabeledAugmentationPipeline(
    #     imputer=SimpleImputer(strategy='most_frequent'),
    #     encoder=ce.CatBoostEncoder(),
    #     scaler=RobustScaler(),.
    #     sampler=ProportionalConditionalGANSampler(sample_multiplication_factor=1, epochs=1)
    # )

    adult = pd.read_csv('../data/adult.csv')
    X = adult.drop(columns='income')
    X = X.replace({'?': np.nan})
    X.columns = range(0, len(X.columns))
    y = adult['income'].map({'<=50K': 0, '>50K': 1})
    y.name = len(X.columns)

    # pipeline.fit_transform(X, y)

    # =========================================================================== #

    X = X.head(2000)
    y = y.head(2000)

    from sklearn.model_selection import train_test_split

    imputer = SimpleImputer(strategy='most_frequent')
    encoder = ce.TargetEncoder()
    scaler = RobustScaler()
    labeler = Labeler
    combiner = DatasetCombiner
    student_model = DecisionTreeClassifier
    student_type = 'decision_tree_classifier'

    n_iter = 1
    cv = 2
    n_jobs = 1
    n_points = 1
    scorer = 'roc_auc'
    verbose = 100
    random_state = 42

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
        n_iter=n_iter,
        cv=cv,
        n_jobs=n_jobs,
        n_points=n_points,
        scoring=scorer,
        verbose=verbose,
        random_state=random_state
    )
    teacher.fit(X_train, y_train)

    baseline_student = BaselineStudentPipeline(
        imputer=imputer,
        encoder=encoder,
        scaler=scaler,
        student=student_model(random_state=random_state),
        search_spaces={
            **search_spaces['decision_tree_classifier']
        },
        n_iter=n_iter,
        cv=cv,
        n_jobs=n_jobs,
        n_points=n_points,
        scoring=scorer,
        verbose=verbose,
        random_state=random_state
    )
    baseline_student.fit(X_train, y_train)

    results = {}

    for name, sampler_type, sampler in [
        ('prop_smote', 'smote', ProportionalSMOTESampler),
        ('unlb_smote', 'smote', UnlabeledSMOTESampler),
        # ('prop_racog', 'racog', ProportionalRACOGSampler),
        # ('unlb_racog', 'racog', UnlabeledRACOGSampler),
        ('prop_gan', 'vanilla_gan', ProportionalVanillaGANSampler),
        ('unlb_gan', 'vanilla_gan', UnlabeledVanillaGANSampler),
        ('prop_cgan', 'conditional_gan', ProportionalConditionalGANSampler),
        ('unlb_cgan', 'conditional_gan', UnlabeledConditionalGANSampler),
        ('prop_dragan', 'dragan', ProportionalDRAGANSampler),
        ('unlb_dragan', 'dragan', UnlabeledDRAGANSampler)
    ]:
        student = TeacherLabeledAugmentedStudentPipeline(
            imputer=imputer,
            encoder=encoder,
            scaler=scaler,
            sampler=sampler(sample_multiplication_factor=1, random_state=42),
            teacher=labeler(trained_model=teacher),
            combiner=combiner(X=X_train_preprocessed, y=y_train_preprocessed),
            student=student_model(),
            search_spaces={
                **search_spaces[student_type],
                **search_spaces[sampler_type]
            },
            n_iter=n_iter,
            cv=cv,
            n_jobs=n_jobs,
            n_points=n_points,
            scoring=scorer,
            verbose=verbose,
            random_state=random_state
        )
        student.fit(X_train, y_train)

        teacher_auc = teacher.score(X_test, y_test)
        baseline_student_auc = baseline_student.score(X_test, y_test)
        student_auc = student.score(X_test, y_test)

        results[name] = [teacher_auc, baseline_student_auc, student_auc]

    for name in results:
        print(name)
        print('teacher auc:', results[name][0])
        print("baseline student auc:", results[name][1])
        print("student auc:", results[name][2])
        print('\n')


def run_indirect_generator_pipeline_test():
    # pipeline = TeacherLabeledAugmentationPipeline(
    #     imputer=SimpleImputer(strategy='most_frequent'),
    #     encoder=ce.CatBoostEncoder(),
    #     scaler=RobustScaler(),.
    #     sampler=ProportionalConditionalGANSampler(sample_multiplication_factor=1, epochs=1)
    # )

    adult = pd.read_csv('../data/adult.csv')
    X = adult.drop(columns='income')
    X = X.replace({'?': np.nan})
    X.columns = range(0, len(X.columns))
    y = adult['income'].map({'<=50K': 0, '>50K': 1})
    y.name = len(X.columns)

    # pipeline.fit_transform(X, y)

    # =========================================================================== #

    X = X.head(2000)
    y = y.head(2000)

    from sklearn.model_selection import train_test_split

    imputer = SimpleImputer(strategy='most_frequent')
    encoder = ce.TargetEncoder()
    scaler = RobustScaler()
    injector = TargetInjector()
    extractor = TargetExtractor()
    discretizer = NumericalTargetDiscretizer
    combiner = DatasetCombiner
    student_model = DecisionTreeClassifier
    student_type = 'decision_tree_classifier'

    n_iter = 1
    cv = 2
    n_jobs = 1
    n_points = 1
    scorer = 'roc_auc'
    verbose = 100
    random_state = 42

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
        n_iter=n_iter,
        cv=cv,
        n_jobs=n_jobs,
        n_points=n_points,
        scoring=scorer,
        verbose=verbose,
        random_state=random_state
    )
    teacher.fit(X_train, y_train)

    baseline_student = BaselineStudentPipeline(
        imputer=imputer,
        encoder=encoder,
        scaler=scaler,
        student=student_model(random_state=random_state),
        search_spaces={
            **search_spaces['decision_tree_classifier']
        },
        n_iter=n_iter,
        cv=cv,
        n_jobs=n_jobs,
        n_points=n_points,
        scoring=scorer,
        verbose=verbose,
        random_state=random_state
    )
    baseline_student.fit(X_train, y_train)

    results = {}

    for name, sampler_type, sampler in [
        ('unlb_smote', 'smote', UnlabeledSMOTESampler),
        # ('unlb_racog', 'racog', UnlabeledRACOGSampler),
        ('unlb_gan', 'vanilla_gan', UnlabeledVanillaGANSampler),
        ('unlb_cgan', 'conditional_gan', UnlabeledConditionalGANSampler),
        ('unlb_dragan', 'dragan', UnlabeledDRAGANSampler)
    ]:
        student = IndirectGeneratorLabeledAugmentedStudentPipeline(
            imputer=imputer,
            encoder=encoder,
            scaler=scaler,
            injector=injector,
            sampler=sampler(sample_multiplication_factor=1, random_state=42),
            extractor=extractor,
            discretizer=discretizer(y=y_train_preprocessed),
            combiner=combiner(X=X_train_preprocessed, y=y_train_preprocessed),
            student=student_model(),
            search_spaces={
                **search_spaces[student_type],
                **search_spaces[sampler_type]
            },
            n_iter=n_iter,
            cv=cv,
            n_jobs=n_jobs,
            n_points=n_points,
            scoring=scorer,
            verbose=verbose,
            random_state=random_state
        )
        student.fit(X_train, y_train)

        teacher_auc = teacher.score(X_test, y_test)
        baseline_student_auc = baseline_student.score(X_test, y_test)
        student_auc = student.score(X_test, y_test)

        results[name] = [teacher_auc, baseline_student_auc, student_auc]

    for name in results:
        print(name)
        print('teacher auc:', results[name][0])
        print("baseline student auc:", results[name][1])
        print("student auc:", results[name][2])
        print('\n')


def run_direct_generator_pipeline_test():
    # pipeline = TeacherLabeledAugmentationPipeline(
    #     imputer=SimpleImputer(strategy='most_frequent'),
    #     encoder=ce.CatBoostEncoder(),
    #     scaler=RobustScaler(),.
    #     sampler=ProportionalConditionalGANSampler(sample_multiplication_factor=1, epochs=1)
    # )

    adult = pd.read_csv('../data/adult.csv')
    X = adult.drop(columns='income')
    X = X.replace({'?': np.nan})
    X.columns = range(0, len(X.columns))
    y = adult['income'].map({'<=50K': 0, '>50K': 1})
    y.name = len(X.columns)

    # pipeline.fit_transform(X, y)

    # =========================================================================== #

    X = X.head(2000)
    y = y.head(2000)

    from sklearn.model_selection import train_test_split

    imputer = SimpleImputer(strategy='most_frequent')
    encoder = ce.TargetEncoder()
    scaler = RobustScaler()
    combiner = DatasetCombiner
    student_model = DecisionTreeClassifier
    student_type = 'decision_tree_classifier'

    n_iter = 1
    cv = 2
    n_jobs = 1
    n_points = 1
    scorer = 'roc_auc'
    verbose = 100
    random_state = 42

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
        n_iter=n_iter,
        cv=cv,
        n_jobs=n_jobs,
        n_points=n_points,
        scoring=scorer,
        verbose=verbose,
        random_state=random_state
    )
    teacher.fit(X_train, y_train)

    baseline_student = BaselineStudentPipeline(
        imputer=imputer,
        encoder=encoder,
        scaler=scaler,
        student=student_model(random_state=random_state),
        search_spaces={
            **search_spaces['decision_tree_classifier']
        },
        n_iter=n_iter,
        cv=cv,
        n_jobs=n_jobs,
        n_points=n_points,
        scoring=scorer,
        verbose=verbose,
        random_state=random_state
    )
    baseline_student.fit(X_train, y_train)

    results = {}

    for name, sampler_type, sampler in [
        ('prop_smote', 'smote', ProportionalSMOTESampler),
        # ('prop_racog', 'racog', ProportionalRACOGSampler),
        ('prop_gan', 'vanilla_gan', ProportionalVanillaGANSampler),
        ('prop_cgan', 'conditional_gan', ProportionalConditionalGANSampler),
        ('prop_dragan', 'dragan', ProportionalDRAGANSampler),
    ]:
        student = DirectGeneratorLabeledAugmentedStudentPipeline(
            imputer=imputer,
            encoder=encoder,
            scaler=scaler,
            sampler=sampler(sample_multiplication_factor=1, random_state=42),
            combiner=combiner(X=X_train_preprocessed, y=y_train_preprocessed),
            student=student_model(),
            search_spaces={
                **search_spaces[student_type],
                **search_spaces[sampler_type]
            },
            n_iter=n_iter,
            cv=cv,
            n_jobs=n_jobs,
            n_points=n_points,
            scoring=scorer,
            verbose=verbose,
            random_state=random_state
        )
        student.fit(X_train, y_train)

        teacher_auc = teacher.score(X_test, y_test)
        baseline_student_auc = baseline_student.score(X_test, y_test)
        student_auc = student.score(X_test, y_test)

        results[name] = [teacher_auc, baseline_student_auc, student_auc]

    for name in results:
        print(name)
        print('teacher auc:', results[name][0])
        print("baseline student auc:", results[name][1])
        print("student auc:", results[name][2])
        print('\n')


if __name__ == '__main__':
    run_teacher_pipeline_test()
    run_indirect_generator_pipeline_test()
    run_direct_generator_pipeline_test()
