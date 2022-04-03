import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import check_memory


class AugmentationPipeline(Pipeline):
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

        fit_transform_one_cached = memory.cache(self._fit_transform_one)

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

            # NEW CODE - unpack X if it is tuple X = (X, y)
            if isinstance(X, tuple):
                X, y = X

            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        return X

    def fit(self, X, y=None, **fit_params):
        """Fit the model.

        Fit all the transformers one after the other and transform the
        data. Finally, fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : object
            Pipeline with fitted steps.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)

        # NEW CODE - unpack X if it is tuple X = (X, y)
        if isinstance(X, tuple):
            X, y = X

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                self._final_estimator.fit(Xt, y, **fit_params_last_step)

        return self


class TeacherLabeledAugmentationPipeline:
    def __init__(
            self,
            imputer,
            encoder,
            scaler,
            sampler,
            teacher,
            combiner,
            student
    ):
        super().__init__()

        self._pipeline = AugmentationPipeline(steps=[
            ('imputer', imputer),
            ('encoder', encoder),
            ('scaler', scaler),
            ('sampler', sampler),
            ('teacher', teacher),
            ('combiner', combiner),
            ('student', student)
        ])

    def fit_transform(self, X, y):
        print(self._pipeline.fit_transform(X, y))


class TeacherTrainingPipeline:
    def __init__(
            self,
            imputer,
            encoder,
            scaler,
            teacher
    ):
        super().__init__()

        self._pipeline = Pipeline(steps=[
            ('imputer', imputer),
            ('encoder', encoder),
            ('scaler', scaler),
            ('teacher', teacher)
        ])

    def fit(self, X, y):
        return self._pipeline.fit(X, y)

    def get_pipeline(self):
        return self._pipeline
