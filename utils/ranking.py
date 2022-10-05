# extract evaluation results from dataset for each combination of encoder, generator, teacher, metric.

import os

import os
from pathlib import Path
import shutil
import math
import json

from collections import Counter, OrderedDict

import operator

from scipy.stats import friedmanchisquare, ttest_rel, t

import scikit_posthocs as sp

from tkinter.filedialog import askdirectory

import numpy as np
import pandas as pd


# a and b are the arrays to be compared
# n is the number of evaluations
# n1 is the number of samples used for training
# n2 is the number of samples used for testing
def adjusted_ttest(a, b, n, n1, n2):
    # Compute the difference between the results
    diff = [y - x for y, x in zip(a, b)]
    # Compute the mean of differences
    d_bar = np.mean(diff)
    # compute the variance of differences
    sigma2 = np.var(diff, ddof=1)
    # compute the modified variance
    sigma2_mod = sigma2 * (1 / n + n2 / n1)
    # compute the t_static
    with np.errstate(divide='ignore', invalid='ignore'):
        t_static = np.divide(d_bar, np.sqrt(sigma2_mod))
    # Compute p-value and plot the results
    p_value = ((1 - t.cdf(np.abs(t_static), n - 1)) * 2.0)

    return t_static, p_value


# ranks the two samples.
# if the ttest shows that they are not significantly different we return 0.5.
# if the samples are different then we return 1 if the eman of a is bigger and 0 if the mean of a is smaller.
def adjusted_ttest_compare(a, b, n, n1, n2):
    statistic, pvalue = adjusted_ttest(a, b, n, n1, n2)

    if pvalue < 0.05:
        # significantly different
        if np.mean(a) > np.mean(b):
            return 1
        else:
            return 0
    else:
        # not significantly different
        return 0.5


# ranks the two samples.
# if the ttest shows that they are not significantly different we return 0.5.
# if the samples are different then we return 1 if the eman of a is bigger and 0 if the mean of a is smaller.
def ttest_compare(a, b):
    statistic, pvalue = ttest_rel(a, b)
    if pvalue < 0.05:
        # significantly different
        if np.mean(a) > np.mean(b):
            return 1
        else:
            return 0
    else:
        # not significantly different
        return 0.5


def get_first_file_path_in_directory_with_extension(directory_path, extension):
    absolute_file_paths = [os.path.join(directory_path, path) for path in os.listdir(directory_path)]

    first_csv_file = None
    for absolute_file_path in absolute_file_paths:
        if absolute_file_path.endswith(extension):
            first_csv_file = absolute_file_path
            break

    if not first_csv_file:
        print('no csv file found')
        # exit(1)

    return first_csv_file


def get_all_file_paths_in_directory(directory_path, extension):
    absolute_file_paths = [os.path.join(directory_path, path) for path in os.listdir(directory_path)]

    file_list = []
    for absolute_file_path in absolute_file_paths:
        if absolute_file_path.endswith(extension):
            file_list.append(absolute_file_path)

    if not file_list:
        print('no ' + extension + ' file found')
        exit(1)

    return file_list


def store_results(result_directory, result_name, result_set):
    Path(result_directory).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(result_directory, result_name)).mkdir(parents=True, exist_ok=True)

    result_set.to_csv(
        os.path.join(os.path.join(result_directory, result_name), result_name + '.csv'),
        index=False
    )


def format_options(option_list):
    output = []
    index = 0
    for option, _ in option_list:
        output.append('[' + str(index) + '] ' + option)
        index = index + 1
    return '\n'.join(output) + '\n'


def get_user_selection(options):
    user_input = input(format_options(options))

    selection = None
    if user_input.isdigit():
        user_input = int(user_input)

        if user_input < len(options):
            selection = options[user_input]

    if not selection:
        print('invalid input')
        exit(1)

    return selection


def validate_path(path):
    try:
        Path(path).resolve()
    except (OSError, RuntimeError):
        print('invalid path')
        exit(1)


def get_all_directories_in_path(directory_path):
    directory_path_list = []
    for path in os.listdir(directory_path):
        if os.path.isdir(os.path.join(directory_path, path)):
            directory_path_list.append(os.path.join(directory_path, path))
    return directory_path_list


def get_unique_values_of_column(dataset, column):
    return dataset[column].unique()


def preprocess(dataset):
    # drop CollapseEncoder / DropEncoder, because it isn't applied to all datasets
    dataset = dataset[dataset.encoder != 'CollapseEncoder']

    # # remove baseline students that have encodings in their preprocessing
    # dataset = dataset[~((dataset.generator.isnull()) & (dataset.teacher.isnull()) & (dataset.encoder != 'NoneType'))]

    # remove duplicates of baseline student
    baseline_student_results = dataset[((dataset.generator.isnull()) & (dataset.teacher.isnull()))]
    baseline_student_results = baseline_student_results.groupby(
        ['encoder', 'optimization_metric', 'random_state']).first().reset_index()
    dataset = dataset[~((dataset.generator.isnull()) & (dataset.teacher.isnull()))]
    dataset = dataset.append(baseline_student_results).reset_index(drop=True)
    dataset = dataset.fillna(value=np.nan)

    # merge statistical runs
    statistical_results = dataset[dataset.generated_fold.notnull()]
    statistical_results = statistical_results.groupby(
        [
            'dataset', 'encoder', 'scaler', 'teacher', 'optimization_metric', 'performance', 'train_size',
            'test_size', 'random_state', 'student', 'student_encoder', 'generator', 'n_samples'
        ],
        dropna=False
    ).mean().reset_index()
    dataset = dataset[~dataset.generated_fold.notnull()]
    dataset = dataset.append(statistical_results).reset_index(drop=True)

    # remove undersampling of synthetic data
    dataset = dataset[~((dataset.n_samples >= 250) & (dataset.n_samples < 10000) & (dataset.generated_fold.isnull()))]

    # rename encoders
    dataset['encoder'] = dataset['encoder'].replace({
        'NoneType': 'nan',
        'BinaryEncoder': 'Binary',
        'CatBoostEncoder': 'CatBoost',
        'CollapseEncoder': 'Collapse',
        'CountEncoder': 'Count',
        'GLMMEncoder': 'GLMM',
        'CV5GLMMEncoder': 'CV5GLMM',
        'TargetEncoder': 'Target',
        'CV5TargetEncoder': 'CV5Target',
        'StratifiedCV5GLMMEncoder': 'CV5GLMM',
        'StratifiedCV5TargetEncoder': 'CV5Target',
        'MultiClassCatBoostEncoder': 'CatBoost',
        'MultiClassGLMMEncoder': 'GLMM',
        'MultiClassStratifiedCV5GLMMEncoder': 'CV5GLMM',
        'MultiClassTargetEncoder': 'Target',
        'MultiClassStratifiedCV5TargetEncoder': 'CV5Target',
    })

    # rename generators
    dataset['generator'] = dataset['generator'].replace({
        'PrivBNGenerator': 'PrivBayes',
        'GaussianCopulaGenerator': 'GaussianCopula',
        'TableGANGenerator': 'TableGAN',
        'CTGANGenerator': 'CTGAN',
        'CopulaGANGenerator': 'CopulaGAN',
        'MedGANGenerator': 'MedGAN',
        'DPCTGANGenerator': 'DP-CTGAN',
        'ProportionalSMOTEGenerator': 'SMOTE',
        'ProportionalCWGANGPGenerator': 'CWGAN-GP',
        'TVAEGenerator': 'TVAE',
        'CTABGANGenerator': 'CTAB-GAN',
        'WGANGPGenerator': 'WGAN-GP'
    })

    # negate metrics that denote better performance with smaller values.
    dataset['jsd'] = dataset['jsd'] * -1
    dataset['wd'] = dataset['wd'] * -1
    dataset['diff_corr'] = dataset['diff_corr'] * -1

    mean_absolute_error_set = dataset[dataset.optimization_metric == 'neg_mean_absolute_error']
    mean_absolute_error_set['performance'] = mean_absolute_error_set['performance'] * -1
    dataset[dataset.optimization_metric == 'neg_mean_absolute_error'] = mean_absolute_error_set

    return dataset


def group_runs(dataset):
    runs = {}
    optimization_metrics = get_unique_values_of_column(dataset, 'optimization_metric')
    for optimization_metric in optimization_metrics:
        machine_learning_efficacy_results = dataset[
            dataset.generated_fold.isnull() & (dataset.optimization_metric == optimization_metric)]
        statistical_results = dataset[
            dataset.generated_fold.notnull() & (dataset.optimization_metric == optimization_metric)]

        runs[optimization_metric] = {k: list(v) for k, v in machine_learning_efficacy_results.groupby([
            'encoder', 'generator', 'teacher'
        ], dropna=False)['performance']}

        for statistical_metric in ['jsd', 'wd', 'diff_corr', 'dcr', 'nndr']:
            runs[optimization_metric + '_' + statistical_metric] = {k: list(v) for k, v in statistical_results.groupby([
                'encoder', 'generator'
            ], dropna=False)[statistical_metric]}
    return runs


def calculate_ranks(run, dataset):
    ranked_run = {}

    # user_input = 'error'
    # while user_input != '':
    #     print(dataset)
    #     n_evaluations = input('how many evaluations were performed ?\n')
    #     n_train = input('how many train samples ?\n')
    #     n_test = input('how many test samples ?\n')
    #     user_input = input('n:' + n_evaluations + ' n1:' + n_train + ' n2:' + n_test + ' press enter to continue')
    # n_evaluations = int(n_evaluations)
    # n_train = int(n_train)
    # n_test = int(n_test)

    for metric in run:
        mapping = {i: k for i, k in enumerate(run[metric].keys())}
        evaluations = {i: v for i, (k, v) in enumerate(run[metric].items())}

        # pairwise comparison matrix
        pcm = np.ndarray((len(evaluations), len(evaluations)))
        for a in evaluations:
            for b in evaluations:
                pcm[a][b] = ttest_compare(evaluations[a], evaluations[b])
                # pcm[a][b] = adjusted_ttest_compare(evaluations[a], evaluations[b], n_evaluations, n_train, n_test)

        total_rank = {}
        for a in range(len(pcm)):
            total_rank[a] = np.sum(pcm[a])

        sorted_rank_values = sorted(total_rank.values())
        sorted_value_counts = Counter(sorted_rank_values)

        sorted_total_rank = dict(sorted(total_rank.items(), key=lambda x: (x[1], x[0])))
        sorted_total_rank = dict(zip(sorted_total_rank.keys(), range(len(sorted_total_rank))))
        relative_rank = dict(sorted(sorted_total_rank.items(), key=lambda x: (x[0], x[1])))

        # # create a ranking from the values
        # # the rank of duplicate values is the mean of the ranks they would cover after sorting, e.g., if the duplicates
        # # would have rank 1 and 2 they would both get the rank 1.5
        # ranking = np.ndarray((len(sorted_rank_values),))
        # i = 0
        # for value in sorted_value_counts:
        #     counts = sorted_value_counts[value]
        #     ranks_covered_by_value = []
        #     for n in range(counts):
        #         ranks_covered_by_value.append(i + n)
        #     mean_rank = np.mean(ranks_covered_by_value)
        #     for n in range(counts):
        #         ranking[i + n] = mean_rank
        #     i = i + counts
        # rank_mapping = dict(zip(sorted_rank_values, ranking))
        #
        # relative_rank = dict(zip(total_rank.keys(), [rank_mapping[v] for v in total_rank.values()]))

        ranked_run[metric] = dict(zip([mapping[k] for k in relative_rank.keys()], relative_rank.values()))

    return ranked_run


def load_dataset(path):
    return pd.read_csv(path, index_col=False)


def get_first_csv_in_directory(directory_path):
    return get_first_file_path_in_directory_with_extension(directory_path, '.csv')


def calculate_ranks_of_ranks(ranks):
    ordered_ranks = {}
    for dataset in ranks:
        ordered_ranks[dataset] = dict(sorted(ranks[dataset].items()))

    # mapping from numbers back to permutation tuple for later
    mapping = {n: k for n, k in enumerate(ranks[list(ordered_ranks.keys())[0]].keys())}

    # map permuatation identifiers to numbers
    indexed_ordered_ranks = {}
    for dataset in ordered_ranks:
        indexed_ordered_ranks[dataset] = {n: v for n, v in enumerate(ordered_ranks[dataset].values())}

    rankings = []
    for dataset in indexed_ordered_ranks:
        rankings.append(list(indexed_ordered_ranks[dataset].values()))

    data = np.array(rankings)
    statistic, pvalue = friedmanchisquare(*rankings)

    averaged_rankings = np.mean(np.array([np.array(r) for r in rankings]), axis=0)

    print(pvalue)
    if pvalue < 0.05:
        matrix = sp.posthoc_nemenyi_friedman(data)

        # pairwise comparison matrix
        pcm = np.ndarray((len(mapping.keys()), len(mapping.keys())))
        for a in range(len(mapping.keys())):
            for b in range(len(mapping.keys())):
                if matrix[a][b] < 0.05:
                    if averaged_rankings[a] > averaged_rankings[b]:
                        pcm[a][b] = 1
                    else:
                        pcm[a][b] = 0
                else:
                    pcm[a][b] = 0.5

        total_rank = {}
        for a in range(len(pcm)):
            total_rank[a] = np.sum(pcm[a])

        sorted_rank_values = sorted(total_rank.values())
        sorted_value_counts = Counter(sorted_rank_values)

        # create a ranking from the values
        # the rank of duplicate values is the mean of the ranks they would cover after sorting, e.g., if the duplicates
        # would have rank 1 and 2 they would both get the rank 1.5
        ranking = np.ndarray((len(sorted_rank_values),))
        i = 0
        for value in sorted_value_counts:
            counts = sorted_value_counts[value]
            ranks_covered_by_value = []
            for n in range(counts):
                ranks_covered_by_value.append(i + n)
            mean_rank = np.mean(ranks_covered_by_value)
            for n in range(counts):
                ranking[i + n] = mean_rank
            i = i + counts
        rank_mapping = dict(zip(sorted_rank_values, ranking))

        # sorted_total_rank = dict(sorted(total_rank.items(), key=lambda x: (x[1], x[0])))
        # sorted_total_rank = dict(zip(sorted_total_rank.keys(), range(len(sorted_total_rank))))
        # relative_rank = dict(sorted(sorted_total_rank.items(), key=lambda x: (x[0], x[1])))
        # dict(zip([mapping[k] for k in relative_rank.keys()], relative_rank.values()))

        return dict(zip(total_rank.keys(), [rank_mapping[v] for v in total_rank.values()]))

    return


def store_ranks(ranked_runs, base_directory, dataset):
    for metric in ranked_runs:
        Path(os.path.join(base_directory, metric)).mkdir(parents=True, exist_ok=True)
        transformed_run = dict(
            zip([' '.join([str(c) for c in k]) for k in ranked_runs[metric].keys()], ranked_runs[metric].values()))
        with open(os.path.join(base_directory, metric, dataset + '.json'), 'w') as file:
            json.dump(transformed_run, file, indent=4)


def load_ranks(directory):
    ranks = {}
    rank_json_paths = get_all_file_paths_in_directory(directory, '.json')
    for path in rank_json_paths:
        dataset = os.path.basename(path)
        with open(path, 'r') as file:
            rank = json.load(file)
            ranks[dataset] = dict(zip([tuple(k.split(' ')) for k in rank.keys()], rank.values()))
    return ranks


import re


def extract_info(dataset):
    match = re.search("exploration_([A-Za-z]+)_(.*)_\d+_\d+_\d+_\d+_\d+(?:_\w+)*.json", dataset)

    if match:
        task, dataset_name = match.groups()

        return task, dataset_name
    else:
        return None


def remove_datetime(dataset):
    match = re.search("(.*)_\d\d\d\d\d\d\d\d_\d\d\d\d\d\d", dataset)
    if match:
        dataset_name, = match.groups()

        return dataset_name
    else:
        return dataset


def create_tables(ranks, metric_type):
    tables = []
    for dataset in ranks:
        table_start = '\\begin{table}[!h]\n\\tiny\n\\centering\n'
        table_end = '\\caption{' + dataset.replace('_', '\_') + ' - ' + metric_type.replace('_',
                                                                                            '\_') + '}\\end{table}\n'
        subtable_separator = '\\quad\n'
        positive_teacher_indicator = '$\\times$'
        subtables = []
        if len(ranks[dataset][list(ranks[dataset].keys())[0]][0]) == 3:
            for metric in ranks[dataset]:
                subtable_start = '\\begin{tabular}{cccc}\n' \
                                 '\\multicolumn{4}{c}{\\scriptsize\\textbf{' + metric.replace('_', '\_') + '}} \\\\\n' \
                                                                                                           '\\hline\n' \
                                                                                                           '\\ & \\textbf{Encoder} & \\textbf{Generator} & \\textbf{T} \\\\\n'
                subtable_end = '\\hline\n' \
                               '\\end{tabular}\n'
                rows = []
                ranking = 1
                for (encoder, generator, teacher) in ranks[dataset][metric]:
                    teacher_indicator = '$-$'
                    if encoder == 'nan':
                        encoder = '$-$'
                    if generator == 'nan':
                        generator = '$-$'
                    if teacher != 'nan':
                        teacher_indicator = positive_teacher_indicator

                    rows.append(
                        str(ranking) + ' & ' + encoder + ' & ' + generator + ' & ' + teacher_indicator + ' \\\\\n'
                    )
                    ranking = ranking + 1
                subtable_content = ''.join(rows)
                subtables.append(subtable_start + subtable_content + subtable_end)
        else:
            for metric in ranks[dataset]:
                subtable_start = '\\begin{tabular}{ccc}\n' \
                                 '\\multicolumn{3}{c}{\\scriptsize\\textbf{' + metric.replace('_', '\_') + '}} \\\\\n' \
                                                                                                           '\\hline\n' \
                                                                                                           '\\ & \\textbf{Encoder} & \\textbf{Generator} \\\\\n'
                subtable_end = '\\hline\n' \
                               '\\end{tabular}\n'
                rows = []
                ranking = 1
                for (encoder, generator) in ranks[dataset][metric]:
                    if encoder == 'nan':
                        encoder = '$-$'
                    if generator == 'nan':
                        generator = '$-$'

                    rows.append(
                        str(ranking) + ' & ' + encoder + ' & ' + generator + ' \\\\\n'
                    )
                    ranking = ranking + 1
                subtable_content = ''.join(rows)
                subtables.append(subtable_start + subtable_content + subtable_end)
        subtables = subtable_separator.join(subtables)

        tables.append(table_start + subtables + table_end)

    return tables


def remove_teacher_only_runs(rankings):
    if len(list(rankings.keys())[0]) != 3:
        return rankings

    rankings = rankings.copy()
    for rank in list(rankings.keys()):
        if rank[1] == 'nan' and (rank[2] == 'LGBMClassifier' or rank[2] == 'LGBMRegressor'):
            del rankings[rank]

    return rankings


def format_first_n_ranks(directory, n):
    metric_paths = get_all_directories_in_path(directory)

    machine_learning_efficacy_metric_paths = []
    for metric in metric_paths:
        machine_learning_efficacy_metric = True
        for suffix in ['_dcr', '_diff_corr', '_jsd', '_nndr', '_wd']:
            if metric.endswith(suffix):
                machine_learning_efficacy_metric = False
        if machine_learning_efficacy_metric:
            machine_learning_efficacy_metric_paths.append(metric)

    statistic_metric_paths = []
    for metric in metric_paths:
        statistic_metric = False
        for suffix in ['_jsd', '_wd', '_diff_corr']:
            if metric.endswith(suffix):
                statistic_metric = True
        if statistic_metric:
            statistic_metric_paths.append(metric)

    privacy_metric_paths = []
    for metric in metric_paths:
        privacy_metric = False
        for suffix in ['_dcr', '_nndr']:
            if metric.endswith(suffix):
                privacy_metric = True
        if privacy_metric:
            privacy_metric_paths.append(metric)

    ranked_machine_learning_efficacy = {}
    for metric in machine_learning_efficacy_metric_paths:
        ranks = load_ranks(metric)
        for dataset in ranks:
            reverse_sorted_ranks = dict(sorted(ranks[dataset].items(), key=operator.itemgetter(1), reverse=True))
            keys = list(reverse_sorted_ranks.keys())
            for k in keys:
                if k[1] == 'nan' and (k[2] == 'LGBMClassifier' or k[2] == 'LGBMRegressor'):
                    del reverse_sorted_ranks[k]

            dataset_name = remove_datetime(extract_info(dataset)[1])
            if dataset_name not in ranked_machine_learning_efficacy:
                ranked_machine_learning_efficacy[dataset_name] = {}
            ranked_machine_learning_efficacy[dataset_name][os.path.basename(metric)] = list(
                reverse_sorted_ranks.keys())[0:n]

    ranked_statistical_metrics = {}
    for metric in statistic_metric_paths:
        ranks = load_ranks(metric)
        for dataset in ranks:
            reverse_sorted_ranks = dict(sorted(ranks[dataset].items(), key=operator.itemgetter(1), reverse=True))
            keys = list(reverse_sorted_ranks.keys())
            # for k in keys:
            #     if k[0] != 'nan':
            #         del reverse_sorted_ranks[k]

            dataset_name = remove_datetime(extract_info(dataset)[1])
            if dataset_name not in ranked_statistical_metrics:
                ranked_statistical_metrics[dataset_name] = {}
            ranked_statistical_metrics[dataset_name][os.path.basename(metric)] = list(reverse_sorted_ranks.keys())[0:n]

    ranked_privacy_metrics = {}
    for metric in privacy_metric_paths:
        ranks = load_ranks(metric)
        for dataset in ranks:
            reverse_sorted_ranks = dict(sorted(ranks[dataset].items(), key=operator.itemgetter(1), reverse=True))
            keys = list(reverse_sorted_ranks.keys())
            # for k in keys:
            #     if k[0] != 'nan':
            #         del reverse_sorted_ranks[k]

            dataset_name = remove_datetime(extract_info(dataset)[1])
            if dataset_name not in ranked_privacy_metrics:
                ranked_privacy_metrics[dataset_name] = {}
            ranked_privacy_metrics[dataset_name][os.path.basename(metric)] = list(reverse_sorted_ranks.keys())[0:n]

    ranked_statistical_metrics_for_optimization_metric = {}
    for dataset in ranked_statistical_metrics:
        ranked_statistical_metrics_for_optimization_metric[dataset] = {}
        for metric in ranked_statistical_metrics[dataset]:
            for prefix in ['f1', 'r2']:
                if metric.startswith(prefix):
                    ranked_statistical_metrics_for_optimization_metric[dataset][metric] = \
                        ranked_statistical_metrics[dataset][metric]

    ranked_privacy_metrics_for_optimization_metric = {}
    for dataset in ranked_privacy_metrics:
        ranked_privacy_metrics_for_optimization_metric[dataset] = {}
        for metric in ranked_privacy_metrics[dataset]:
            for prefix in ['f1', 'r2']:
                if metric.startswith(prefix):
                    ranked_privacy_metrics_for_optimization_metric[dataset][metric] = \
                        ranked_privacy_metrics[dataset][metric]

    # tables = create_tables(ranked_machine_learning_efficacy, 'machine learning efficacy')
    # print('\n\n'.join(tables))
    # tables = create_tables(ranked_statistical_metrics, 'statistical similarity')
    # print('\n\n'.join(tables))
    tables = create_tables(ranked_privacy_metrics, 'privacy preservability')
    print('\n\n'.join(tables))


def get_top_n(ranking, n):
    sorted_ranking = {k: v for k, v in sorted(ranking.items(), key=lambda item: item[1], reverse=True)}
    return list(sorted_ranking.keys())[0:n]


def find_top_n_occurences_by_component(directory, n, grouping, requested_metrics):
    metric_paths = get_all_directories_in_path(directory)

    requested_metric_paths = []
    for metric in metric_paths:
        requested = False
        for desired_metric in requested_metrics:
            if metric.endswith(desired_metric):
                requested = True
        if requested:
            requested_metric_paths.append(metric)
    metric_paths = requested_metric_paths

    top_n_rankings = {}
    for metric in metric_paths:
        rankings = load_ranks(metric)
        top_n_rankings[metric] = {}
        for dataset in rankings:
            top_n_rankings[metric][dataset] = get_top_n(remove_teacher_only_runs(rankings[dataset]), n)

    tuple_index = None
    if grouping == 'encoder':
        tuple_index = 0
    elif grouping == 'generator':
        tuple_index = 1

    if tuple_index == None:
        print('invalid component')
        exit(1)

    top_n_occurences = {}
    for metric in top_n_rankings:
        top_n_occurences[metric] = {}
        for dataset in top_n_rankings[metric]:
            for entry in top_n_rankings[metric][dataset]:
                if entry[tuple_index] not in top_n_occurences[metric]:
                    top_n_occurences[metric][entry[tuple_index]] = 0
                top_n_occurences[metric][entry[tuple_index]] = top_n_occurences[metric][entry[tuple_index]] + 1

    top_n_occurences = {os.path.basename(k): v for k, v in top_n_occurences.items()}

    metrics = list(top_n_occurences.keys())
    datasets = []
    for metric in top_n_occurences:
        datasets.extend(list(top_n_occurences[metric].keys()))
    datasets = list(set(datasets))

    result_matrix = np.ndarray((len(datasets), len(metrics)))
    for i in range(len(datasets)):
        for j in range(len(metrics)):
            if datasets[i] in top_n_occurences[metrics[j]]:
                result_matrix[i][j] = top_n_occurences[metrics[j]][datasets[i]]
            else:
                result_matrix[i][j] = 0

    sum_per_metric = np.sum(result_matrix, axis=0)
    sum_per_dataset = np.sum(result_matrix, axis=1)
    metrics = [v.replace('neg_mean_absolute_error', 'mae') for v in metrics]
    metrics = [v.replace('_', '\_') for v in metrics]
    datasets = ['$-$' if v == 'nan' else v for v in datasets]
    datasets = [v.replace('_', '\_') for v in datasets]

    table_start = '\\begin{table}[!h]\n\\small\n\\centering\n' \
                  '\\begin{tabular}{c|' + 'c' * len(metrics) + '|c}\n' \
                  '\\ & ' + ' & '.join(['' + m + '' for m in metrics]) + ' & \\textbf{total} \\\\\n' \
                  '\\hline\n'
    table_end = '\\end{tabular}\n'\
                '\\caption{' + 'Number of times each INSERT was ranked in the top 10 for all regression datasets and metrics.' + '}\n' \
                '\\end{table}\n'

    rows = []
    for i in range(len(result_matrix)):
        row = '' + datasets[i] + '' + ' & ' + ' & '.join([str(int(n)) for n in result_matrix[i]]) + ' & \\textbf{' + str(int(sum_per_dataset[i])) + '} \\\\\n'
        rows.append(row)
    rows.append('\\hline\n'
                '\\textbf{total} & ' + ' & '.join(['\\textbf{' + str(int(n)) + '}' for n in sum_per_metric]) + ' & \\textbf{ ' + str(int(np.sum(sum_per_metric))) + '} \\\\\n')

    table = table_start + ''.join(rows) + table_end
    print(table)

def find_top_n_occurences(directory, n, encoders, generators, teachers, requested_metrics, not_in=False):
    metric_paths = get_all_directories_in_path(directory)

    requested_metric_paths = []
    for metric in metric_paths:
        requested = False
        for desired_metric in requested_metrics:
            if metric.endswith(desired_metric):
                requested = True
        if requested:
            requested_metric_paths.append(metric)
    metric_paths = requested_metric_paths

    top_n_rankings = {}
    for metric in metric_paths:
        rankings = load_ranks(metric)
        top_n_rankings[metric] = {}
        for dataset in rankings:
            top_n_rankings[metric][dataset] = get_top_n(remove_teacher_only_runs(rankings[dataset]), n)

    top_n_occurences = {}
    for metric in top_n_rankings:
        top_n_occurences[metric] = {}
        for dataset in top_n_rankings[metric]:
            top_n_occurences[metric][dataset] = 0
            if not_in:
                if len(top_n_rankings[metric][dataset][0]) == 3:
                    for encoder, generator, teacher in top_n_rankings[metric][dataset]:
                        if encoder not in encoders and generator not in generators and teacher not in teachers:
                            top_n_occurences[metric][dataset] = top_n_occurences[metric][dataset] + 1
                else:
                    for encoder, generator in top_n_rankings[metric][dataset]:
                        if encoder not in encoders and generator not in generators:
                            top_n_occurences[metric][dataset] = top_n_occurences[metric][dataset] + 1
            else:
                if len(top_n_rankings[metric][dataset][0]) == 3:
                    for encoder, generator, teacher in top_n_rankings[metric][dataset]:
                        if encoder in encoders or generator in generators or teacher in teachers:
                            top_n_occurences[metric][dataset] = top_n_occurences[metric][dataset] + 1
                else:
                    for encoder, generator in top_n_rankings[metric][dataset]:
                        if encoder in encoders or generator in generators:
                            top_n_occurences[metric][dataset] = top_n_occurences[metric][dataset] + 1

    top_n_occurences = {os.path.basename(k): v for k, v in top_n_occurences.items()}
    for metric in top_n_occurences:
        top_n_occurences[metric] = {remove_datetime(extract_info(k)[1]): v for k, v in top_n_occurences[metric].items()}

    metrics = list(top_n_occurences.keys())
    datasets = list(top_n_occurences[list(top_n_occurences.keys())[0]].keys())

    result_matrix = np.ndarray((len(datasets), len(metrics)))
    for i in range(len(datasets)):
        for j in range(len(metrics)):
            result_matrix[i][j] = top_n_occurences[metrics[j]][datasets[i]]

    sum_per_metric = np.sum(result_matrix, axis=0)
    sum_per_dataset = np.sum(result_matrix, axis=1)
    metrics = [v.replace('neg_mean_absolute_error', 'mae') for v in metrics]
    metrics = [v.replace('_', '\_') for v in metrics]
    datasets = [v.replace('_', '\_') for v in datasets]

    table_start = '\\begin{table}[!h]\n\\small\n\\centering\n' \
                  '\\begin{tabular}{c|' + 'c' * len(metrics) + '|c}\n' \
                  '\\ & ' + ' & '.join(['' + m + '' for m in metrics]) + ' & \\textbf{total} \\\\\n' \
                  '\\hline\n'
    table_end = '\\end{tabular}\n'\
                '\\caption{' + 'Number of times a INSERT was in the top 10 for each dataset and metric.' + '}\\end{table}\n'

    rows = []
    for i in range(len(result_matrix)):
        row = '' + datasets[i] + '' + ' & ' + ' & '.join([str(int(n)) for n in result_matrix[i]]) + ' & \\textbf{' + str(int(sum_per_dataset[i])) + '} \\\\\n'
        rows.append(row)
    rows.append('\\hline\n'
                '\\textbf{total} & ' + ' & '.join(['\\textbf{' + str(int(n)) + '}' for n in sum_per_metric]) + ' & \\textbf{ ' + str(int(np.sum(sum_per_metric))) + '} \\\\\n')

    table = table_start + ''.join(rows) + table_end
    print(table)




def aggregate_first_n_ranks(directory, n):
    metric_paths = get_all_directories_in_path(directory)

    machine_learning_efficacy_metric_paths = []
    for metric in metric_paths:
        machine_learning_efficacy_metric = True
        for suffix in ['_dcr', '_diff_corr', '_jsd', '_nndr', '_wd']:
            if metric.endswith(suffix):
                machine_learning_efficacy_metric = False
        if machine_learning_efficacy_metric:
            machine_learning_efficacy_metric_paths.append(metric)

    statistic_metric_paths = []
    for metric in metric_paths:
        statistic_metric = False
        for suffix in ['_jsd', '_wd', '_diff_corr']:
            if metric.endswith(suffix):
                statistic_metric = True
        if statistic_metric:
            statistic_metric_paths.append(metric)

    privacy_metric_paths = []
    for metric in metric_paths:
        privacy_metric = False
        for suffix in ['_dcr', '_nndr']:
            if metric.endswith(suffix):
                privacy_metric = True
        if privacy_metric:
            privacy_metric_paths.append(metric)

    ranked_machine_learning_efficacy = {}
    for metric in machine_learning_efficacy_metric_paths:
        ranks = load_ranks(metric)
        for dataset in ranks:
            reverse_sorted_ranks = dict(sorted(ranks[dataset].items(), key=operator.itemgetter(1), reverse=True))
            keys = list(reverse_sorted_ranks.keys())
            for k in keys:
                if k[1] == 'nan' and (k[2] == 'LGBMClassifier' or k[2] == 'LGBMRegressor'):
                    del reverse_sorted_ranks[k]

            dataset_name = remove_datetime(extract_info(dataset)[1])
            if dataset_name not in ranked_machine_learning_efficacy:
                ranked_machine_learning_efficacy[dataset_name] = {}
            ranked_machine_learning_efficacy[dataset_name][os.path.basename(metric)] = list(
                reverse_sorted_ranks.keys())[0:n]

    ranked_statistical_metrics = {}
    for metric in statistic_metric_paths:
        ranks = load_ranks(metric)
        for dataset in ranks:
            reverse_sorted_ranks = dict(sorted(ranks[dataset].items(), key=operator.itemgetter(1), reverse=True))
            keys = list(reverse_sorted_ranks.keys())
            # for k in keys:
            #     if k[0] != 'nan':
            #         del reverse_sorted_ranks[k]

            dataset_name = remove_datetime(extract_info(dataset)[1])
            if dataset_name not in ranked_statistical_metrics:
                ranked_statistical_metrics[dataset_name] = {}
            ranked_statistical_metrics[dataset_name][os.path.basename(metric)] = list(reverse_sorted_ranks.keys())[0:n]

    ranked_privacy_metrics = {}
    for metric in privacy_metric_paths:
        ranks = load_ranks(metric)
        for dataset in ranks:
            reverse_sorted_ranks = dict(sorted(ranks[dataset].items(), key=operator.itemgetter(1), reverse=True))
            keys = list(reverse_sorted_ranks.keys())
            # for k in keys:
            #     if k[0] != 'nan':
            #         del reverse_sorted_ranks[k]

            dataset_name = remove_datetime(extract_info(dataset)[1])
            if dataset_name not in ranked_privacy_metrics:
                ranked_privacy_metrics[dataset_name] = {}
            ranked_privacy_metrics[dataset_name][os.path.basename(metric)] = list(reverse_sorted_ranks.keys())[0:n]

    ranked_statistical_metrics_for_optimization_metric = {}
    for dataset in ranked_statistical_metrics:
        ranked_statistical_metrics_for_optimization_metric[dataset] = {}
        for metric in ranked_statistical_metrics[dataset]:
            for prefix in ['f1', 'r2']:
                if metric.startswith(prefix):
                    ranked_statistical_metrics_for_optimization_metric[dataset][metric] = \
                        ranked_statistical_metrics[dataset][metric]

    ranked_privacy_metrics_for_optimization_metric = {}
    for dataset in ranked_privacy_metrics:
        ranked_privacy_metrics_for_optimization_metric[dataset] = {}
        for metric in ranked_privacy_metrics[dataset]:
            for prefix in ['f1', 'r2']:
                if metric.startswith(prefix):
                    ranked_privacy_metrics_for_optimization_metric[dataset][metric] = \
                        ranked_privacy_metrics[dataset][metric]

    print(ranked_machine_learning_efficacy)


def string_combinations(list_1, list_2):
    if not list_1:
        return list_2
    if not list_2:
        return list_1

    results = []
    for a in list_1:
        for b in list_2:
            results.append(a + b)
    return results


import matplotlib.pyplot as plt
import seaborn as sns

def get_run_time_plots(run_directory):
    directory_paths = get_all_directories_in_path(run_directory)
    run_times_per_dataset = pd.DataFrame()
    for directory_path in directory_paths:
        dataset_csv_path = get_first_csv_in_directory(directory_path)
        if not dataset_csv_path:
            continue
        dataset = load_dataset(dataset_csv_path)
        dataset = preprocess(dataset)
        run_times_per_dataset = run_times_per_dataset.append(dataset.groupby(['dataset', 'generator', 'random_state']).mean().reset_index()[['dataset', 'generator', 'run_time', 'random_state']])
    print(run_times_per_dataset)

    f, ax = plt.subplots(figsize=(7, 6))
    # Plot the orbital period with horizontal boxes
    plot = sns.boxplot(x="run_time", y="generator", data=dataset,
                whis=[0, 100], width=.6, palette="vlag")
    plot.set_xscale("log")
    # ax.xaxis.grid(True)
    # ax.set(ylabel="")
    sns.despine(trim=True, left=True)
    fig = plot.get_figure()
    fig.savefig(os.path.join(run_directory, os.path.basename(run_directory) + "_run_times_log.png"), bbox_inches='tight')

    plt.clf()

    f, ax = plt.subplots(figsize=(7, 6))
    # Plot the orbital period with horizontal boxes
    plot = sns.boxplot(x="run_time", y="generator", data=dataset,
                       whis=[0, 100], width=.6, palette="vlag")
    # plot.set_xscale("log")
    # ax.xaxis.grid(True)
    # ax.set(ylabel="")
    sns.despine(trim=True, left=True)
    fig = plot.get_figure()
    fig.savefig(os.path.join(run_directory, os.path.basename(run_directory) + "_run_times.png"), bbox_inches='tight')

    exit(1)


def run(interrupt=True):
    # run_directory = 'C:\\Users\\Thomas\\Desktop\\test_results\\regression'
    print('select the root folder\n')
    run_directory = askdirectory(title='Select Folder')
    if run_directory == '':
        print('invalid path')
        exit(1)

    # directory_paths = get_all_directories_in_path(run_directory)
    # for directory_path in directory_paths:
    #     dataset = load_dataset(get_first_csv_in_directory(directory_path))
    #     dataset = preprocess(dataset)
    #     grouped_runs = group_runs(dataset)
    #     ranked_runs = calculate_ranks(grouped_runs, os.path.basename(directory_path))
    #     store_ranks(ranked_runs, os.path.join(run_directory, 'rankings'), os.path.basename(directory_path))

    # metrics = get_all_directories_in_path(os.path.join(run_directory, 'rankings'))
    # for metric in metrics:
    #     ranks = load_ranks(os.path.join(run_directory, 'rankings', metric))
    #     ranks_of_ranks = calculate_ranks_of_ranks(ranks)
    #     if ranks_of_ranks:
    #         with open(os.path.join(run_directory, 'rankings', metric + '.json'), 'w') as file:
    #             json.dump(ranks_of_ranks, file, indent=4)

    # format_first_n_ranks(run_directory, 20)
    # aggregate_first_n_ranks(run_directory, 20)
    # find_top_n_occurences(run_directory, 10, [], ['nan'], [],
    #                       ['neg_mean_absolute_error', 'r2', 'accuracy', 'roc_auc', 'f1', 'f1_macro'], not_in=True)
    find_top_n_occurences_by_component(run_directory, 10, 'encoder',
                                       ['neg_mean_absolute_error', 'r2', 'accuracy', 'roc_auc', 'f1', 'f1_macro'])
    # find_top_n_occurences_by_component(run_directory, 10, 'generator', ['neg_mean_absolute_error', 'r2', 'accuracy', 'roc_auc', 'f1', 'f1_macro'])
    # get_run_time_plots(run_directory)
    # find_top_n_occurences_by_component(run_directory, 10, 'generator', string_combinations(['f1', 'f1_macro', 'r2'], ['_jsd']))
    # find_top_n_occurences_by_component(run_directory, 10, 'generator', string_combinations(['f1', 'f1_macro', 'r2'], ['_jsd', '_wd', '_diff_corr']))
    # find_top_n_occurences_by_component(run_directory, 10, 'generator', string_combinations(['f1', 'f1_macro', 'r2'], ['_dcr', '_nndr']))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='A script for ranking the generated results.'
    )
    parser.add_argument('--interrupt', action='store_true')
    parser.add_argument('--no-interrupt', dest='interrupt', action='store_false')
    parser.set_defaults(feature=True)

    args = parser.parse_args()

    run(args.interrupt)

# dataset = loaded_datasets[absolute_path]['csv']
# dataset = dataset[~dataset['generator'].isin(user_input)]
# Path(result_directory).mkdir(parents=True, exist_ok=True)
# Path(os.path.join(result_directory, result_name)).mkdir(parents=True, exist_ok=True)
#
# dataset.to_csv(
#     os.path.join(result_directory, result_name, os.path.basename(absolute_path) + '.csv'),
#     index=False
# )
