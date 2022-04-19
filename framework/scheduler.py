"""Scheduler for parallel execution of pipeline exploration"""

# Authors: Thomas Frank <https://github.com/thomas475>
# License: MIT

from sklearn.model_selection import train_test_split
import itertools
from joblib import Parallel, delayed
import traceback

import time
from datetime import datetime

from framework.pipelines import *
from framework.samplers import ProportionalSampler, UnlabeledSampler


class Scheduler:
    """
    Scheduler that explores all possible pipelines with the submitted
    transformers and search spaces.

    Sample Multiplication Factors can be floats and integers.

    Datasets must be submitted as a tuple of its name, X, y, train_size and
    test_size. Missing entries in X have to be replaced with np.nan for use in
    the imputers and their column titles should be replaced with numbers. y
    must be mapped to integers if not already done so. Train size and test size
     have to be a multiple of the number of cross validation folds cv.

    Transformers used in the pipelines are submitted either just as a
    transformer or a tuple containing the transformer and a search space. For
    example
        encoders=[
            TargetEncoder
        ]
        samplers=[
            (ProportionalSMOTESampler, {'sampler__k_neighbors': Integer(1, 7)})
        ]
    are both valid inputs but only for the sampler a search space for
    optimization in the pipeline is used.
    """

    def __init__(
            self,
            experiment_base_title=None,
            pipelines=None,
            sample_multiplication_factors=None,
            datasets=None,
            imputers=None,
            encoders=None,
            scalers=None,
            samplers=None,
            teachers=None,
            labelers=None,
            injectors=None,
            extractors=None,
            discretizers=None,
            students=None,
            metrics=None,
            n_iter=50,
            n_points=1,
            cv=10,
            n_jobs=-1,
            verbose=100,
            random_state=None,
    ):
        self._experiment_base_title = experiment_base_title
        self._pipelines = pipelines
        self._sample_multiplication_factors = sample_multiplication_factors
        self._datasets = datasets
        self._imputers = imputers
        self._encoders = encoders
        self._scalers = scalers
        self._samplers = samplers
        self._teachers = teachers
        self._labelers = labelers
        self._injectors = injectors
        self._extractors = extractors
        self._discretizers = discretizers
        self._students = students
        self._metrics = metrics
        self._n_iter = n_iter
        self._n_points = n_points
        self._cv = cv
        self._n_jobs = n_jobs
        self._verbose = verbose
        self._random_state = random_state

    def explore(self):
        self._experiment_title = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self._experiment_base_title is not None:
            self._experiment_title = self._experiment_base_title + '_' + self._experiment_title

        log_messages = []
        savesets = []
        if TeacherLabeledAugmentedStudentPipeline in self._pipelines:
            saveset, logs = self._explore_teacher_labeled_augmented_student_pipeline()
            savesets.append(saveset)
            log_messages.extend(logs)
        if IndirectGeneratorLabeledAugmentedStudentPipeline in self._pipelines:
            saveset, logs = self._explore_indirect_generator_labeled_augmented_student_pipeline()
            savesets.append(saveset)
            log_messages.extend(logs)
        if DirectGeneratorLabeledAugmentedStudentPipeline in self._pipelines:
            saveset, logs = self._explore_direct_generator_labeled_augmented_student_pipeline()
            savesets.append(saveset)
            log_messages.extend(logs)

        savesets_frame = pd.DataFrame()
        for saveset in savesets:
            savesets_frame = pd.concat(
                [
                    savesets_frame,
                    saveset
                ],
                ignore_index=True
            )

        with open(self._experiment_title + '.log', 'w') as log_file:
            log_file.write('\n\n'.join(log_messages))
        savesets_frame.to_csv(self._experiment_title + '.csv', index=False)

    def _explore_teacher_labeled_augmented_student_pipeline(self):
        log_messages = []

        teacher_permutations = itertools.product(
            self._datasets,
            self._imputers,
            self._encoders,
            self._scalers,
            self._teachers,
            self._metrics
        )

        explored_teachers = Parallel(
            n_jobs=-1,
            verbose=self._verbose
        )(
            delayed(self._train_teacher)(
                dataset=dataset,
                imputer=self._remove_search_space(imputer),
                encoder=self._remove_search_space(encoder),
                scaler=self._remove_search_space(scaler),
                teacher=self._remove_search_space(teacher),
                search_spaces=self._get_search_spaces([
                    imputer,
                    encoder,
                    scaler,
                    teacher
                ]),
                metric=metric
            )
            for (
                index, (
                    dataset,
                    imputer,
                    encoder,
                    scaler,
                    teacher,
                    metric
                )
            ) in enumerate(teacher_permutations)
        )

        # convert the list of explored teachers to a dictionary with the identifiers as keys
        explored_teachers = {k: v for element in explored_teachers for k, v in element.items()}

        teacher_saveset = pd.DataFrame()

        # halt function execution if there has been an error while training any of the teachers
        teacher_training_error = False
        for key in explored_teachers:
            if 'error' in explored_teachers[key]['metadata']:
                teacher_training_error = True
                log_message = 'There has been an error with running the following instance: ' + key \
                              + '\n' + explored_teachers[key]['metadata']['error']['exception'] \
                              + '\n' + explored_teachers[key]['metadata']['error']['traceback']
            else:
                teacher_saveset = pd.concat(
                    [
                        teacher_saveset,
                        pd.DataFrame({
                            k: str(v) for k, v in explored_teachers[key]['metadata'].items()
                        }, index=[0])
                    ],
                    ignore_index=True
                )
                log_message = 'Completed running the following instance: ' + key
            log_messages.append(log_message)
            print(log_message)
        if teacher_training_error:
            return teacher_saveset

        student_permutations = itertools.product(
            self._sample_multiplication_factors,
            self._datasets,
            self._imputers,
            self._encoders,
            self._scalers,
            self._samplers,
            self._teachers,
            self._labelers,
            self._students,
            self._metrics
        )

        explored_students = Parallel(
            n_jobs=-1,
            verbose=self._verbose
        )(
            delayed(self._train_teacher_labeled_augmented_student)(
                sample_multiplication_factor=sample_multiplication_factor,
                dataset=dataset,
                imputer=self._remove_search_space(imputer),
                encoder=self._remove_search_space(encoder),
                scaler=self._remove_search_space(scaler),
                sampler=self._remove_search_space(sampler),
                trained_teacher_identifier='_'.join([
                    TeacherPipeline.__name__,
                    str(dataset[0]),
                    self._remove_search_space(imputer).__name__,
                    self._remove_search_space(encoder).__name__,
                    self._remove_search_space(scaler).__name__,
                    self._remove_search_space(teacher).__name__,
                    str(self._get_search_spaces([
                        imputer,
                        encoder,
                        scaler,
                        teacher
                    ])),
                    str(metric),
                    str(self._n_iter),
                    str(self._n_points),
                    str(self._cv),
                    str(self._n_jobs),
                    str(self._random_state)
                ]),
                trained_teacher=explored_teachers[
                    '_'.join([
                        TeacherPipeline.__name__,
                        str(dataset[0]),
                        self._remove_search_space(imputer).__name__,
                        self._remove_search_space(encoder).__name__,
                        self._remove_search_space(scaler).__name__,
                        self._remove_search_space(teacher).__name__,
                        str(self._get_search_spaces([
                            imputer,
                            encoder,
                            scaler,
                            teacher
                        ])),
                        str(metric),
                        str(self._n_iter),
                        str(self._n_points),
                        str(self._cv),
                        str(self._n_jobs),
                        str(self._random_state)
                    ])
                ],
                labeler=self._remove_search_space(labeler),
                student=self._remove_search_space(student),
                search_spaces=self._get_search_spaces([
                    imputer,
                    encoder,
                    scaler,
                    sampler,
                    labeler,
                    student
                ]),
                metric=metric
            )
            for (
                index, (
                    sample_multiplication_factor,
                    dataset,
                    imputer,
                    encoder,
                    scaler,
                    sampler,
                    teacher,
                    labeler,
                    student,
                    metric
                )
            ) in enumerate(student_permutations)
        )

        # convert the list of explored students to a dictionary with the identifiers as keys
        explored_students = {k: v for element in explored_students for k, v in element.items()}

        student_saveset = pd.DataFrame()

        for key in explored_students:
            if 'error' in explored_students[key]:
                log_message = 'There has been an error with running the following instance: ' + key \
                              + '\n' + explored_students[key]['error']['exception'] \
                              + '\n' + explored_students[key]['error']['traceback']
            else:
                student_saveset = pd.concat(
                    [
                        student_saveset,
                        pd.DataFrame({
                            k: str(v) for k, v in explored_students[key].items()
                        }, index=[0])
                    ],
                    ignore_index=True
                )
                log_message = 'Completed running the following instance: ' + key
            log_messages.append(log_message)
            print(log_message)

        saveset = pd.concat(
            [
                teacher_saveset,
                student_saveset
            ],
            ignore_index=True
        )

        return saveset, log_messages

    def _train_teacher(
            self,
            dataset,
            imputer,
            encoder,
            scaler,
            teacher,
            search_spaces,
            metric
    ):
        start_time = time.time()

        dataset_name, X, y, train_size, test_size = dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_size,
            test_size=test_size,
            stratify=y,
            random_state=self._random_state
        )

        # the identifier of this training instance
        identifier = '_'.join([
            TeacherPipeline.__name__,
            str(dataset_name),
            imputer.__name__,
            encoder.__name__,
            scaler.__name__,
            teacher.__name__,
            str(search_spaces),
            str(metric),
            str(self._n_iter),
            str(self._n_points),
            str(self._cv),
            str(self._n_jobs),
            str(self._random_state)
        ])

        output = {}

        try:
            model = TeacherPipeline(
                imputer=self._set_attribute_if_available(imputer(), 'random_state', self._random_state),
                encoder=self._set_attribute_if_available(encoder(), 'random_state', self._random_state),
                scaler=self._set_attribute_if_available(scaler(), 'random_state', self._random_state),
                teacher=self._set_attribute_if_available(teacher(), 'random_state', self._random_state),
                search_spaces=search_spaces,
                scoring=metric,
                n_iter=self._n_iter,
                n_points=self._n_points,
                cv=self._cv,
                n_jobs=self._n_jobs,
                random_state=self._random_state
            )
            model.fit(X_train, y_train)

            output = {
                identifier: {
                    'model': model,
                    'metadata': {
                        'pipeline': TeacherPipeline.__name__,
                        'dataset': dataset_name,
                        'imputer': imputer.__name__,
                        'encoder': encoder.__name__,
                        'scaler': scaler.__name__,
                        'teacher': teacher.__name__,
                        'search_spaces': search_spaces,
                        'metric': metric,
                        'best_params': model.get_params(),
                        'test_score': model.score(X_test, y_test),
                        'n_iter': self._n_iter,
                        'n_points': self._n_points,
                        'cv': self._cv,
                        'n_jobs': self._n_jobs,
                        'random_state': self._random_state,
                        'train_size': len(X_train),
                        'test_size': len(X_test),
                        'runtime': time.time() - start_time
                    }
                }
            }
        except Exception as e:
            output = {
                identifier: {
                    'model': None,
                    'metadata': {
                        'pipeline': TeacherPipeline.__name__,
                        'dataset': dataset_name,
                        'imputer': imputer.__name__,
                        'encoder': encoder.__name__,
                        'scaler': scaler.__name__,
                        'search_spaces': search_spaces,
                        'metric': metric,
                        'n_iter': self._n_iter,
                        'n_points': self._n_points,
                        'cv': self._cv,
                        'n_jobs': self._n_jobs,
                        'random_state': self._random_state,
                        'runtime': time.time() - start_time,
                        'error': {
                            'exception': str(e),
                            'traceback': traceback.format_exc()
                        }
                    }
                }
            }
        finally:
            return output

    def _train_teacher_labeled_augmented_student(
            self,
            sample_multiplication_factor,
            dataset,
            imputer,
            encoder,
            scaler,
            sampler,
            trained_teacher_identifier,
            trained_teacher,
            labeler,
            student,
            search_spaces,
            metric
    ):
        start_time = time.time()

        dataset_name, X, y, train_size, test_size = dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_size,
            test_size=test_size,
            stratify=y,
            random_state=self._random_state
        )

        # the identifier of this training instance
        identifier = '_'.join([
            TeacherLabeledAugmentedStudentPipeline.__name__,
            str(sample_multiplication_factor),
            trained_teacher_identifier,
            sampler.__name__,
            labeler.__name__,
            student.__name__,
            str(search_spaces),
        ])

        output = {}

        try:
            model = TeacherLabeledAugmentedStudentPipeline(
                imputer=self._set_attribute_if_available(imputer(), 'random_state', self._random_state),
                scaler=self._set_attribute_if_available(scaler(), 'random_state', self._random_state),
                encoder=self._set_attribute_if_available(encoder(), 'random_state', self._random_state),
                sampler=self._set_attribute_if_available(sampler(
                    sample_multiplication_factor=sample_multiplication_factor,
                    only_sampled=False
                ), 'random_state', self._random_state),
                teacher=self._set_attribute_if_available(labeler(
                    trained_model=trained_teacher['model'].get_estimator(),
                    ignored_first_n_samples=int(len(X_train) - (len(X_train) / self._cv))
                ), 'random_state', self._random_state),
                student=self._set_attribute_if_available(student(), 'random_state', self._random_state),
                search_spaces=search_spaces,
                scoring=metric,
                n_iter=self._n_iter,
                n_points=self._n_points,
                cv=self._cv,
                n_jobs=self._n_jobs,
                random_state=self._random_state
            )
            model.fit(X_train, y_train)

            output = {
                identifier: {
                    'pipeline': TeacherLabeledAugmentedStudentPipeline.__name__,
                    'sample_multiplication_factor': sample_multiplication_factor,
                    'dataset': dataset_name,
                    'imputer': imputer.__name__,
                    'encoder': encoder.__name__,
                    'scaler': scaler.__name__,
                    'sampler': sampler.__name__,
                    'teacher': trained_teacher['metadata']['teacher'],
                    'teacher_search_spaces': trained_teacher['metadata']['search_spaces'],
                    'labeler': labeler.__name__,
                    'student': student.__name__,
                    'search_spaces': search_spaces,
                    'metric': metric,
                    'best_params': model.get_params(),
                    'test_score': model.score(X_test, y_test),
                    'n_iter': self._n_iter,
                    'n_points': self._n_points,
                    'cv': self._cv,
                    'n_jobs': self._n_jobs,
                    'random_state': self._random_state,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'runtime': time.time() - start_time
                }
            }
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            output = {
                identifier: {
                    'pipeline': TeacherLabeledAugmentedStudentPipeline.__name__,
                    'sample_multiplication_factor': sample_multiplication_factor,
                    'dataset': dataset_name,
                    'imputer': imputer.__name__,
                    'encoder': encoder.__name__,
                    'scaler': scaler.__name__,
                    'sampler': sampler.__name__,
                    'teacher': trained_teacher['metadata']['teacher'],
                    'teacher_search_spaces': trained_teacher['metadata']['search_spaces'],
                    'labeler': labeler.__name__,
                    'student': student.__name__,
                    'search_spaces': search_spaces,
                    'metric': metric,
                    'n_iter': self._n_iter,
                    'n_points': self._n_points,
                    'cv': self._cv,
                    'n_jobs': self._n_jobs,
                    'random_state': self._random_state,
                    'runtime': time.time() - start_time,
                    'error': {
                        'exception': str(e),
                        'traceback': traceback.format_exc()
                    }
                }
            }
        finally:
            return output

    def _explore_indirect_generator_labeled_augmented_student_pipeline(self):
        log_messages = []

        student_permutations = itertools.product(
            self._sample_multiplication_factors,
            self._datasets,
            self._imputers,
            self._encoders,
            self._scalers,
            self._injectors,
            [
                sampler for sampler in self._samplers if issubclass(
                    self._remove_search_space(sampler),
                    UnlabeledSampler
                )
            ],
            self._extractors,
            self._discretizers,
            self._students,
            self._metrics
        )

        explored_students = Parallel(
            n_jobs=-1,
            verbose=self._verbose
        )(
            delayed(self._train_indirect_generator_labeled_augmented_student)(
                sample_multiplication_factor=sample_multiplication_factor,
                dataset=dataset,
                imputer=self._remove_search_space(imputer),
                encoder=self._remove_search_space(encoder),
                scaler=self._remove_search_space(scaler),
                injector=self._remove_search_space(injector),
                sampler=self._remove_search_space(sampler),
                extractor=self._remove_search_space(extractor),
                discretizer=self._remove_search_space(discretizer),
                student=self._remove_search_space(student),
                search_spaces=self._get_search_spaces([
                    imputer,
                    encoder,
                    scaler,
                    injector,
                    sampler,
                    extractor,
                    discretizer,
                    student
                ]),
                metric=metric
            )
            for (
                index, (
                    sample_multiplication_factor,
                    dataset,
                    imputer,
                    encoder,
                    scaler,
                    injector,
                    sampler,
                    extractor,
                    discretizer,
                    student,
                    metric
                )
            ) in enumerate(student_permutations)
        )

        # convert the list of explored students to a dictionary with the identifiers as keys
        explored_students = {k: v for element in explored_students for k, v in element.items()}

        saveset = pd.DataFrame()

        for key in explored_students:
            if 'error' in explored_students[key]:
                log_message = 'There has been an error with running the following instance: ' + key \
                              + '\n' + explored_students[key]['error']['exception'] \
                              + '\n' + explored_students[key]['error']['traceback']
            else:
                saveset = pd.concat(
                    [
                        saveset,
                        pd.DataFrame({
                            k: str(v) for k, v in explored_students[key].items()
                        }, index=[0])
                    ],
                    ignore_index=True
                )
                log_message = 'Completed running the following instance: ' + key
            log_messages.append(log_message)
            print(log_message)

        return saveset, log_messages

    def _train_indirect_generator_labeled_augmented_student(
            self,
            sample_multiplication_factor,
            dataset,
            imputer,
            encoder,
            scaler,
            injector,
            sampler,
            extractor,
            discretizer,
            student,
            search_spaces,
            metric
    ):
        start_time = time.time()

        dataset_name, X, y, train_size, test_size = dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_size,
            test_size=test_size,
            stratify=y,
            random_state=self._random_state
        )

        # the identifier of this training instance
        identifier = '_'.join([
            IndirectGeneratorLabeledAugmentedStudentPipeline.__name__,
            str(sample_multiplication_factor),
            str(dataset_name),
            imputer.__name__,
            encoder.__name__,
            scaler.__name__,
            injector.__name__,
            sampler.__name__,
            extractor.__name__,
            discretizer.__name__,
            student.__name__,
            str(search_spaces),
            str(metric),
            str(self._n_iter),
            str(self._n_points),
            str(self._cv),
            str(self._n_jobs),
            str(self._random_state)
        ])

        output = {}

        try:
            model = IndirectGeneratorLabeledAugmentedStudentPipeline(
                imputer=self._set_attribute_if_available(imputer(), 'random_state', self._random_state),
                scaler=self._set_attribute_if_available(scaler(), 'random_state', self._random_state),
                encoder=self._set_attribute_if_available(encoder(), 'random_state', self._random_state),
                injector=self._set_attribute_if_available(injector(), 'random_state', self._random_state),
                sampler=self._set_attribute_if_available(
                    sampler(
                        sample_multiplication_factor=sample_multiplication_factor,
                        only_sampled=False
                    ), 'random_state', self._random_state),
                extractor=self._set_attribute_if_available(extractor(), 'random_state', self._random_state),
                discretizer=self._set_attribute_if_available(
                    discretizer(
                        y=y_train
                    ), 'random_state', self._random_state),
                student=self._set_attribute_if_available(student(), 'random_state', self._random_state),
                search_spaces=search_spaces,
                scoring=metric,
                n_iter=self._n_iter,
                n_points=self._n_points,
                cv=self._cv,
                n_jobs=self._n_jobs,
                random_state=self._random_state
            )
            model.fit(X_train, y_train)

            output = {
                identifier: {
                    'pipeline': IndirectGeneratorLabeledAugmentedStudentPipeline.__name__,
                    'sample_multiplication_factor': sample_multiplication_factor,
                    'dataset': dataset_name,
                    'imputer': imputer.__name__,
                    'encoder': encoder.__name__,
                    'scaler': scaler.__name__,
                    'injector': injector.__name__,
                    'sampler': sampler.__name__,
                    'extractor': extractor.__name__,
                    'discretizer': discretizer.__name__,
                    'student': student.__name__,
                    'search_spaces': search_spaces,
                    'metric': metric,
                    'best_params': model.get_params(),
                    'test_score': model.score(X_test, y_test),
                    'n_iter': self._n_iter,
                    'n_points': self._n_points,
                    'cv': self._cv,
                    'n_jobs': self._n_jobs,
                    'random_state': self._random_state,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'runtime': time.time() - start_time
                }
            }
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            output = {
                identifier: {
                    'pipeline': IndirectGeneratorLabeledAugmentedStudentPipeline.__name__,
                    'sample_multiplication_factor': sample_multiplication_factor,
                    'dataset': dataset_name,
                    'imputer': imputer.__name__,
                    'encoder': encoder.__name__,
                    'scaler': scaler.__name__,
                    'injector': injector.__name__,
                    'sampler': sampler.__name__,
                    'extractor': extractor.__name__,
                    'discretizer': discretizer.__name__,
                    'student': student.__name__,
                    'search_spaces': search_spaces,
                    'metric': metric,
                    'n_iter': self._n_iter,
                    'n_points': self._n_points,
                    'cv': self._cv,
                    'n_jobs': self._n_jobs,
                    'random_state': self._random_state,
                    'runtime': time.time() - start_time,
                    'error': {
                        'exception': str(e),
                        'traceback': traceback.format_exc()
                    }
                }
            }
        finally:
            return output

    def _explore_direct_generator_labeled_augmented_student_pipeline(self):
        log_messages = []

        student_permutations = itertools.product(
            self._sample_multiplication_factors,
            self._datasets,
            self._imputers,
            self._encoders,
            self._scalers,
            [
                sampler for sampler in self._samplers if issubclass(
                    self._remove_search_space(sampler),
                    ProportionalSampler
                )
            ],
            self._students,
            self._metrics
        )

        explored_students = Parallel(
            n_jobs=-1,
            verbose=self._verbose
        )(
            delayed(self._train_direct_generator_labeled_augmented_student)(
                sample_multiplication_factor=sample_multiplication_factor,
                dataset=dataset,
                imputer=self._remove_search_space(imputer),
                encoder=self._remove_search_space(encoder),
                scaler=self._remove_search_space(scaler),
                sampler=self._remove_search_space(sampler),
                student=self._remove_search_space(student),
                search_spaces=self._get_search_spaces([
                    imputer,
                    encoder,
                    scaler,
                    sampler,
                    student
                ]),
                metric=metric
            )
            for (
                index, (
                    sample_multiplication_factor,
                    dataset,
                    imputer,
                    encoder,
                    scaler,
                    sampler,
                    student,
                    metric
                )
            ) in enumerate(student_permutations)
        )

        # convert the list of explored students to a dictionary with the identifiers as keys
        explored_students = {k: v for element in explored_students for k, v in element.items()}

        saveset = pd.DataFrame()

        for key in explored_students:
            if 'error' in explored_students[key]:
                log_message = 'There has been an error with running the following instance: ' + key \
                              + '\n' + explored_students[key]['error']['exception'] \
                              + '\n' + explored_students[key]['error']['traceback']
            else:
                saveset = pd.concat(
                    [
                        saveset,
                        pd.DataFrame({
                            k: str(v) for k, v in explored_students[key].items()
                        }, index=[0])
                    ],
                    ignore_index=True
                )
                log_message = 'Completed running the following instance: ' + key
            log_messages.append(log_message)
            print(log_message)

        return saveset, log_messages

    def _train_direct_generator_labeled_augmented_student(
            self,
            sample_multiplication_factor,
            dataset,
            imputer,
            encoder,
            scaler,
            sampler,
            student,
            search_spaces,
            metric
    ):
        start_time = time.time()

        dataset_name, X, y, train_size, test_size = dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_size,
            test_size=test_size,
            stratify=y,
            random_state=self._random_state
        )

        # the identifier of this training instance
        identifier = '_'.join([
            DirectGeneratorLabeledAugmentedStudentPipeline.__name__,
            str(sample_multiplication_factor),
            str(dataset_name),
            imputer.__name__,
            encoder.__name__,
            scaler.__name__,
            sampler.__name__,
            student.__name__,
            str(search_spaces),
            str(metric),
            str(self._n_iter),
            str(self._n_points),
            str(self._cv),
            str(self._n_jobs),
            str(self._random_state)
        ])

        output = {}

        try:
            model = DirectGeneratorLabeledAugmentedStudentPipeline(
                imputer=self._set_attribute_if_available(imputer(), 'random_state', self._random_state),
                scaler=self._set_attribute_if_available(scaler(), 'random_state', self._random_state),
                encoder=self._set_attribute_if_available(encoder(), 'random_state', self._random_state),
                sampler=self._set_attribute_if_available(
                    sampler(
                        sample_multiplication_factor=sample_multiplication_factor,
                        only_sampled=False
                    ), 'random_state', self._random_state),
                student=self._set_attribute_if_available(student(), 'random_state', self._random_state),
                search_spaces=search_spaces,
                scoring=metric,
                n_iter=self._n_iter,
                n_points=self._n_points,
                cv=self._cv,
                n_jobs=self._n_jobs,
                random_state=self._random_state
            )
            model.fit(X_train, y_train)

            output = {
                identifier: {
                    'pipeline': DirectGeneratorLabeledAugmentedStudentPipeline.__name__,
                    'sample_multiplication_factor': sample_multiplication_factor,
                    'dataset': dataset_name,
                    'imputer': imputer.__name__,
                    'encoder': encoder.__name__,
                    'scaler': scaler.__name__,
                    'sampler': sampler.__name__,
                    'student': student.__name__,
                    'search_spaces': search_spaces,
                    'metric': metric,
                    'best_params': model.get_params(),
                    'test_score': model.score(X_test, y_test),
                    'n_iter': self._n_iter,
                    'n_points': self._n_points,
                    'cv': self._cv,
                    'n_jobs': self._n_jobs,
                    'random_state': self._random_state,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'runtime': time.time() - start_time
                }
            }
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            output = {
                identifier: {
                    'pipeline': DirectGeneratorLabeledAugmentedStudentPipeline.__name__,
                    'sample_multiplication_factor': sample_multiplication_factor,
                    'dataset': dataset_name,
                    'imputer': imputer.__name__,
                    'encoder': encoder.__name__,
                    'scaler': scaler.__name__,
                    'sampler': sampler.__name__,
                    'student': student.__name__,
                    'search_spaces': search_spaces,
                    'metric': metric,
                    'n_iter': self._n_iter,
                    'n_points': self._n_points,
                    'cv': self._cv,
                    'n_jobs': self._n_jobs,
                    'random_state': self._random_state,
                    'runtime': time.time() - start_time,
                    'error': {
                        'exception': str(e),
                        'traceback': traceback.format_exc()
                    }
                }
            }
        finally:
            return output

    def _remove_search_space(self, step):
        """
        If the submitted step is a two element tuple we discard the second
        element as the search space and return the first element as the
        transformer.
        """
        if isinstance(step, tuple):
            step, _ = step
        return step

    def _get_search_spaces(self, steps):
        """
        The submitted steps are either transformers or tuples consisting of a
        transformer and a search space. This function joins and returns all the
        search spaces submitted in the steps.
        """
        search_spaces = {}
        for step in steps:
            if isinstance(step, tuple):
                _, search_space = step
                search_spaces = {**search_spaces, **search_space}
        return search_spaces

    def _set_attribute_if_available(self, instance, attribute, value):
        if hasattr(instance, attribute):
            setattr(instance, attribute, value)
        return instance
