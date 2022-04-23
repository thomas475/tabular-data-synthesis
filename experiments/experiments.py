from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import category_encoders as ce
from skopt.space import Real, Categorical, Integer

import os

from framework.scheduler import Scheduler
from framework.imputers import MostFrequentImputer
from framework.transformers import *
from framework.pipelines import *
from framework.samplers import *
import framework.encoders as enc


class Exploration:
    def run(self):
        pipelines = [
            TeacherLabeledAugmentedStudentPipeline,
            IndirectGeneratorLabeledAugmentedStudentPipeline,
            DirectGeneratorLabeledAugmentedStudentPipeline
        ]
        imputers = [
            MostFrequentImputer,
        ]
        encoders = [
            # ce.BinaryEncoder,
            ce.CatBoostEncoder,
            ce.CountEncoder,
            ce.OrdinalEncoder,
            ce.TargetEncoder,
            enc.CollapseEncoder
        ]
        scalers = [
            RobustScaler
        ]
        samplers = [
            (ProportionalConditionalGANSampler, {
                # 'sampler__sample_multiplication_factor': Integer(0, 20),
                'sampler__epochs': [1],
                'sampler__batch_size': Integer(32, 256),
                'sampler__learning_rate': Real(0.00001, 0.01),
                'sampler__noise_dim': Integer(64, 512),
                'sampler__layers_dim': Integer(32, 256)
            }),
            (UnlabeledConditionalGANSampler, {
                # 'sampler__sample_multiplication_factor': Integer(0, 20),
                'sampler__epochs': [1],
                'sampler__batch_size': Integer(32, 256),
                'sampler__learning_rate': Real(0.00001, 0.01),
                'sampler__noise_dim': Integer(64, 512),
                'sampler__layers_dim': Integer(32, 256)
            }),
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
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20
        ]
        n_iter = 20
        n_points = 2
        train_ratio = 0.75
        cv = 5
        n_jobs = -1
        verbose = 100
        random_state = 42

# ===== LOAD DATASET ======================================================== #

        datasets = []

        # load adult dataset
        adult = pd.read_csv('data/adult.csv')

        # preprocess dataset
        X = adult.drop(columns='income')
        X = X.replace({'?': np.nan})
        X.columns = range(0, len(X.columns))

        # preprocess target
        y = adult['income'].map({'<=50K': 0, '>50K': 1})
        y.name = len(X.columns)

        # choose desired number of samples used from this dataset
        selected_n_samples = 1000
        total_n_samples = min(selected_n_samples, len(X))

        # set number of samples, train size and test size so that they are divisible by cv
        train_size = int(((total_n_samples * train_ratio) // cv) * cv)
        test_size = int(((total_n_samples * (1 - train_ratio)) // cv) * cv)
        total_n_samples = train_size + test_size

        X = X.head(total_n_samples)
        y = y.head(total_n_samples)

        datasets.append(('adult', X, y, train_size, test_size))

# ===== LOAD DATASET ======================================================== #

        experiment_directory = os.path.join(os.getcwd(), 'experiments', 'runs')
        experiment_base_title = 'cgan_adult'

        scheduler = Scheduler(
            experiment_directory=experiment_directory,
            experiment_base_title=experiment_base_title,
            pipelines=pipelines,
            sample_multiplication_factors=sample_multiplication_factors,
            datasets=datasets,
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
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )

        scheduler.explore()


Exploration().run()
