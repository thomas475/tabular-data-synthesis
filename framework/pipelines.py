import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.pipeline import _final_estimator_has, _fit_transform_one
from sklearn.base import clone
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_memory

from sklearn.ensemble import RandomForestClassifier

from skopt import BayesSearchCV

from inspect import signature


class CustomPipeline(Pipeline):
    def __init__(self, steps, predict_ignored_steps=[], memory=None, verbose=False):
        self.predict_ignored_steps = predict_ignored_steps
        super().__init__(steps=steps, memory=memory, verbose=verbose)

    """
    Modified version of sklearn.pipeline.Pipeline that enables transformers to
    update X and y in each pipeline step.
    """

    def _fit(self, X, y=None, **fit_params_steps):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        for (step_idx, name, transformer) in self._iter(
                with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(memory, "location"):
                # joblib >= 0.12
                if memory.location is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            elif hasattr(memory, "cachedir"):
                # joblib < 0.11
                if memory.cachedir is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            else:
                cloned_transformer = clone(transformer)
            # Fit or load from cache the current transformer
            X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                **fit_params_steps[name],
            )

            # NEW CODE
            if isinstance(X, tuple):
                X, y = X

            # print(name)
            # print(pd.concat([pd.DataFrame(X).reset_index(drop=True), pd.Series(y).reset_index(drop=True)], axis=1))
            # print(name)
            # print('len(X):', np.shape(X), ' len(y):', np.shape(y))

            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)

        # NEW CODE
        return X, y

    def fit(self, X, y=None, **fit_params):
        fit_params_steps = self._check_fit_params(**fit_params)
        # NEW CODE
        Xt, y = self._fit(X, y, **fit_params_steps)

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                self._final_estimator.fit(Xt, y, **fit_params_last_step)

        return self

    def _can_transform(self):
        return self._final_estimator == "passthrough" or hasattr(
            self._final_estimator, "transform"
        )

    @available_if(_can_transform)
    def transform(self, X, y=None):
        Xt = X
        for _, _, transform in self._iter():
            # NEW CODE
            y_param = False
            for param in signature(transform.transform).parameters.values():
                if param.name == 'y':
                    y_param = True

            if y_param:
                Xt = transform.transform(Xt, y)
            else:
                Xt = transform.transform(Xt)

            if isinstance(Xt, tuple):
                Xt, y = Xt

        # NEW CODE
        return Xt, y

    def fit_transform(self, X, y=None, **fit_params):
        fit_params_steps = self._check_fit_params(**fit_params)
        # NEW CODE
        Xt, y = self._fit(X, y, **fit_params_steps)

        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt, y
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]

            # NEW CODE
            if hasattr(last_step, "fit_transform"):
                Xt = last_step.fit_transform(Xt, y, **fit_params_last_step)
            else:
                y_param = False
                for param in signature(last_step.transform).parameters.values():
                    if param.name == 'y':
                        y_param = True

                if y_param:
                    Xt = last_step.fit(Xt, y, **fit_params_last_step).transform(Xt, y)
                else:
                    Xt = last_step.fit(Xt, y, **fit_params_last_step).transform(Xt)

            if isinstance(Xt, tuple):
                Xt, y = Xt

            # NEW CODE
            return Xt, y

    @available_if(_final_estimator_has("predict"))
    def predict(self, X, **predict_params):
        Xt = X
        y = None
        for _, name, transform in self._iter(with_final=False):
            if name not in self.predict_ignored_steps:
                # NEW CODE
                y_param = False
                for param in signature(transform.transform).parameters.values():
                    if param.name == 'y':
                        y_param = True

                if y_param:
                    Xt = transform.transform(Xt, y)
                else:
                    Xt = transform.transform(Xt)

                if isinstance(Xt, tuple):
                    Xt, y = Xt

        return self.steps[-1][1].predict(Xt, **predict_params)

    @available_if(_final_estimator_has("fit_predict"))
    def fit_predict(self, X, y=None, **fit_params):
        fit_params_steps = self._check_fit_params(**fit_params)
        # NEW CODE
        Xt, _ = self._fit(X, y, **fit_params_steps)

        fit_params_last_step = fit_params_steps[self.steps[-1][0]]
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            y_pred = self.steps[-1][1].fit_predict(Xt, y, **fit_params_last_step)
        return y_pred

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X, **predict_proba_params):
        Xt = X
        y = None
        for _, name, transform in self._iter(with_final=False):
            if name not in self.predict_ignored_steps:
                # NEW CODE
                y_param = False
                for param in signature(transform.transform).parameters.values():
                    if param.name == 'y':
                        y_param = True

                if y_param:
                    Xt = transform.transform(Xt, y)
                else:
                    Xt = transform.transform(Xt)

                if isinstance(Xt, tuple):
                    Xt, y = Xt

        return self.steps[-1][1].predict_proba(Xt, **predict_proba_params)

    @available_if(_final_estimator_has("score"))
    def score(self, X, y=None, sample_weight=None):
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            # NEW CODE
            y_param = False
            for param in signature(transform.transform).parameters.values():
                if param.name == 'y':
                    y_param = True

            if y_param:
                Xt = transform.transform(Xt, y)
            else:
                Xt = transform.transform(Xt)

            if isinstance(Xt, tuple):
                Xt, y = Xt

        score_params = {}
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight
        return self.steps[-1][1].score(Xt, y, **score_params)


class OptimalModelPipeline:
    def __init__(
            self,
            pipeline,
            search_spaces,
            scoring,
            n_iter=50,
            cv=10,
            n_jobs=-1,
            n_points=1,
            verbose=10,
            random_state=None
    ):
        self._optimal_model = BayesSearchCV(
            estimator=pipeline,
            search_spaces=search_spaces,
            n_jobs=n_jobs,
            n_iter=n_iter,
            n_points=n_points,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            random_state=random_state
        )

    def fit(self, X, y):
        return self._optimal_model.fit(X, y)

    def predict(self, X):
        return self._optimal_model.predict(X)

    def score(self, X, y):
        return self._optimal_model.score(X, y)


class PreprocessorPipeline:
    def __init__(
            self,
            imputer,
            scaler,
            encoder
    ):
        self._pipeline = CustomPipeline(steps=[
            ('imputer', imputer),
            ('encoder', encoder),
            ('scaler', scaler)
        ])

    def fit(self, X, y=None):
        return self._pipeline.fit(X, y)

    def transform(self, X, y=None):
        return self._pipeline.transform(X, y)

    def fit_transform(self, X, y=None):
        return self._pipeline.fit_transform(X, y)


class TeacherLabeledAugmentedStudentPipeline(OptimalModelPipeline):
    def __init__(
            self,
            imputer,
            scaler,
            encoder,
            sampler,
            teacher,
            combiner,
            student,
            search_spaces,
            scoring,
            n_iter=50,
            cv=10,
            n_jobs=-1,
            n_points=1,
            verbose=10,
            random_state=None
    ):
        pipeline = CustomPipeline(
            steps=[
                ('imputer', imputer),
                ('encoder', encoder),
                ('scaler', scaler),
                ('sampler', sampler),
                ('teacher', teacher),
                ('combiner', combiner),
                ('student', student)
            ],
            predict_ignored_steps=[
                'sampler',
                'teacher',
                'combiner'
            ]
        )

        super().__init__(
            pipeline=pipeline,
            search_spaces=search_spaces,
            n_jobs=n_jobs,
            n_iter=n_iter,
            n_points=n_points,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            random_state=random_state
        )


class IndirectGeneratorLabeledAugmentedStudentPipeline(OptimalModelPipeline):
    def __init__(
            self,
            imputer,
            scaler,
            encoder,
            injector,
            sampler,
            extractor,
            discretizer,
            combiner,
            student,
            search_spaces,
            scoring,
            n_iter=50,
            n_points=1,
            cv=10,
            n_jobs=-1,
            verbose=10,
            random_state=None
    ):
        pipeline = CustomPipeline(
            steps=[
                ('imputer', imputer),
                ('encoder', encoder),
                ('scaler', scaler),
                ('injector', injector),
                ('sampler', sampler),
                ('extractor', extractor),
                ('discretizer', discretizer),
                ('combiner', combiner),
                ('student', student)
            ],
            predict_ignored_steps=[
                'injector',
                'sampler',
                'extractor',
                'discretizer',
                'combiner'
            ]
        )

        super().__init__(
            pipeline=pipeline,
            search_spaces=search_spaces,
            n_jobs=n_jobs,
            n_iter=n_iter,
            n_points=n_points,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            random_state=random_state
        )


class DirectGeneratorLabeledAugmentedStudentPipeline(OptimalModelPipeline):
    def __init__(
            self,
            imputer,
            scaler,
            encoder,
            sampler,
            combiner,
            student,
            search_spaces,
            scoring,
            n_iter=50,
            n_points=1,
            cv=10,
            n_jobs=-1,
            verbose=10,
            random_state=None
    ):
        pipeline = CustomPipeline(
            steps=[
                ('imputer', imputer),
                ('encoder', encoder),
                ('scaler', scaler),
                ('sampler', sampler),
                ('combiner', combiner),
                ('student', student)
            ],
            predict_ignored_steps=[
                'sampler',
                'combiner'
            ]
        )

        super().__init__(
            pipeline=pipeline,
            search_spaces=search_spaces,
            n_jobs=n_jobs,
            n_iter=n_iter,
            n_points=n_points,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            random_state=random_state
        )


class BaselineStudentPipeline(OptimalModelPipeline):
    def __init__(
            self,
            imputer,
            encoder,
            scaler,
            student,
            search_spaces,
            scoring,
            n_iter=50,
            n_points=1,
            cv=10,
            n_jobs=-1,
            verbose=10,
            random_state=None
    ):
        pipeline = CustomPipeline(steps=[
            ('imputer', imputer),
            ('encoder', encoder),
            ('scaler', scaler),
            ('student', student)
        ])

        super().__init__(
            pipeline=pipeline,
            search_spaces=search_spaces,
            scoring=scoring,
            n_iter=n_iter,
            n_points=n_points,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )


class TeacherPipeline(OptimalModelPipeline):
    def __init__(
            self,
            imputer,
            encoder,
            scaler,
            teacher,
            search_spaces,
            scoring,
            n_iter=50,
            n_points=1,
            cv=10,
            n_jobs=-1,
            verbose=10,
            random_state=None
    ):
        pipeline = CustomPipeline(steps=[
            ('imputer', imputer),
            ('encoder', encoder),
            ('scaler', scaler),
            ('teacher', teacher)
        ])

        super().__init__(
            pipeline=pipeline,
            search_spaces=search_spaces,
            scoring=scoring,
            n_iter=n_iter,
            n_points=n_points,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )


class RandomForestClassifierTeacherPipeline(TeacherPipeline):
    def __init__(
            self,
            imputer,
            encoder,
            scaler,
            search_spaces={
                'teacher__n_estimators': (25, 175),
                'teacher__max_features': ['auto', 'sqrt'],
                'teacher__max_depth': (15, 90),
                'teacher__min_samples_split': (2, 10),
                'teacher__min_samples_leaf': (1, 7),
                'teacher__bootstrap': ["True", "False"]
            },
            scoring='roc_auc',
            n_iter=25,
            n_points=1,
            cv=5,
            n_jobs=-1,
            verbose=10,
            random_state=None
    ):
        super().__init__(
            imputer=imputer,
            encoder=encoder,
            scaler=scaler,
            teacher=RandomForestClassifier(random_state=random_state),
            search_spaces=search_spaces,
            scoring=scoring,
            n_iter=n_iter,
            n_points=n_points,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )
