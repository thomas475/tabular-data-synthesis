import os

import os
from pathlib import Path
import shutil

import re
import json

from tkinter import Tk
from tkinter.filedialog import askdirectory

import numpy as np
import pandas as pd


def group_runs(directory_path, split_by_suffix=False):
    directory_path_list = []
    for path in os.listdir(directory_path):
        if os.path.isdir(os.path.join(directory_path, path)):
            directory_path_list.append((path, os.path.join(directory_path, path)))

    groups = {}
    for relative_directory_path, absolute_directory_path in directory_path_list:
        match = re.search("(.*)_\d\d\d\d\d\d\d\d_\d\d\d\d\d\d((?:_\d+)*)((?:_\w+)*)", relative_directory_path)

        if match:
            dataset_name, random_states, suffix = match.groups()

            random_states = random_states.split('_')
            random_states.remove('')
            random_states = sorted(random_states)

            suffixes = suffix.split('_')
            suffixes.remove('')
            suffixes = sorted(suffixes)

            if split_by_suffix:
                identifier = '_'.join([dataset_name] + suffixes)
            else:
                identifier = dataset_name

            if identifier not in groups:
                groups[identifier] = []

            groups[identifier].append(
                {
                    'dataset_name': dataset_name,
                    'path': absolute_directory_path,
                    'random_states': random_states,
                    'suffixes': suffixes
                }
            )
    return groups


def get_file_path_in_directory(directory_path, extension):
    absolute_file_paths = [os.path.join(directory_path, path) for path in os.listdir(directory_path)]

    first_csv_file = None
    for absolute_file_path in absolute_file_paths:
        if absolute_file_path.endswith(extension):
            first_csv_file = absolute_file_path
            break

    if not first_csv_file:
        print('no csv file found')
        exit(1)

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


def store_results(result_directory, result_name, result_set, log_file_paths):
    Path(result_directory).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(result_directory, result_name)).mkdir(parents=True, exist_ok=True)

    result_set.to_csv(
        os.path.join(os.path.join(result_directory, result_name), result_name + '.csv'),
        index=False
    )
    for log_file_path in log_file_paths:
        shutil.copy(
            log_file_path,
            os.path.join(result_directory, result_name, os.path.basename(log_file_path))
        )


def combine_random_state(directory_path, interrupt=False):
    result_directory = os.path.join(directory_path, 'processed')

    groups = group_runs(directory_path, split_by_suffix=True)
    print('found the following groups:\n' + '\n'.join(groups.keys()) + '\n')

    for group in groups:
        dataset_name, absolute_paths, random_states, suffixes = extract_group_data(groups, group)

        if interrupt:
            user_input = ''
            while user_input not in ['y', 'N']:
                user_input = input('combine randomstates of ' + group + ' ? [y/N]\n'
                                   + '\n'.join(absolute_paths) + '\n')
        else:
            print('combining randomstates of ' + group + '\n' + '\n'.join(absolute_paths) + '\n')
            user_input = 'y'

        if user_input == 'y':
            result_set = pd.DataFrame()
            log_file_paths = []
            for absolute_path in absolute_paths:
                result_set = result_set.append(
                    pd.read_csv(
                        get_file_path_in_directory(absolute_path, '.csv'),
                        index_col=False
                    ),
                    ignore_index=True
                )
                log_file_paths.extend(get_all_file_paths_in_directory(absolute_path, '.log'))

            result_name = '_'.join([dataset_name] + random_states + suffixes)

            store_results(result_directory, result_name, result_set, log_file_paths)


def extract_group_data(groups, group):
    dataset_name = groups[group][0]['dataset_name']
    absolute_paths = sorted([entry['path'] for entry in groups[group]])
    random_states = []
    for random_state_list in [entry['random_states'] for entry in groups[group]]:
        for random_state in random_state_list:
            random_states.append(random_state)
    random_states = sorted(list(set(random_states)))
    suffixes = sorted(groups[group][0]['suffixes'])

    return dataset_name, absolute_paths, random_states, suffixes


def merge_runs(directory_path, interrupt=False):
    result_directory = os.path.join(directory_path, 'processed')

    groups = group_runs(directory_path)
    print('found the following groups:\n' + '\n'.join(groups.keys()) + '\n')

    for group in groups:
        dataset_name, absolute_paths, random_states, suffixes = extract_group_data(groups, group)

        if interrupt:
            user_input = ''
            while user_input not in ['y', 'N']:
                user_input = input('combine randomstates of ' + group + ' ? [y/N]\n'
                                   + '\n'.join(absolute_paths) + '\n')
        else:
            print('combining randomstates of ' + group + '\n' + '\n'.join(absolute_paths) + '\n')
            user_input = 'y'

        if user_input == 'y':
            result_set = pd.DataFrame()
            log_file_paths = []
            for absolute_path in absolute_paths:
                result_set = result_set.append(
                    pd.read_csv(
                        get_file_path_in_directory(absolute_path, '.csv'),
                        index_col=False
                    ),
                    ignore_index=True
                )
                log_file_paths.extend(get_all_file_paths_in_directory(absolute_path, '.log'))

            result_name = '_'.join([dataset_name] + random_states + suffixes)

            store_results(result_directory, result_name, result_set, log_file_paths)


def remove_generator(directory_path, interrupt=True):
    result_directory = os.path.join(directory_path, 'processed')

    groups = group_runs(directory_path, split_by_suffix=True)
    print('found the following groups:\n' + '\n'.join(groups.keys()) + '\n')

    all_generators = [
        np.nan,
        'TableGANGenerator',
        'CTGANGenerator',
        'CopulaGANGenerator',
        'TVAEGenerator',
        'MedGANGenerator',
        'DPCTGANGenerator',
        'CTABGANGenerator',
        'ProportionalCWGANGPGenerator',
        'WGANGPGenerator',
    ]

    if not interrupt:
        user_input = input(
            'which generators do you want to remove from all groups if available ?'
            + ' valid inputs are for example "", "1" or "1 2"\n'
            + '\n'.join([
                '[' + str(index) + '] ' + str(generator) for index, generator in enumerate(all_generators)
            ]) + '\n'
        )
        if user_input == '':
            user_input = []
        else:
            user_input = user_input.split(' ')
            if False in [entry.isdigit() for entry in user_input]:
                print('invalid input')
                exit(1)
            user_input = [int(number) for number in user_input]
            for number in user_input:
                if number > len(all_generators):
                    print('invalid input')
                    exit(1)
            user_input = [all_generators[i] for i in user_input]

    for group in groups:
        dataset_name, absolute_paths, random_states, suffixes = extract_group_data(groups, group)

        loaded_datasets = {}
        generators = []
        log_file_paths = []
        for absolute_path in absolute_paths:
            loaded_datasets[absolute_path] = {}
            loaded_datasets[absolute_path]['csv'] = pd.read_csv(
                get_file_path_in_directory(absolute_path, '.csv'),
                index_col=False
            )
            loaded_datasets[absolute_path]['log'] = get_all_file_paths_in_directory(
                absolute_path,
                '.log'
            )
            generators.extend(list(loaded_datasets[absolute_path]['csv']['generator'].unique()))
        generators = list(set(generators))

        if interrupt:
            user_input = input(
                'which generators do you want to remove from ' + group
                + ' ? valid inputs are for example "", "1" or "1 2"\n'
                + '\n'.join([
                    '[' + str(index) + '] ' + str(generator) for index, generator in enumerate(generators)
                ]) + '\n'
            )
            if user_input == '':
                user_input = []
            else:
                user_input = user_input.split(' ')
                if False in [entry.isdigit() for entry in user_input]:
                    print('invalid input')
                    exit(1)
                user_input = [int(number) for number in user_input]
                for number in user_input:
                    if number > len(generators):
                        print('invalid input')
                        exit(1)
                user_input = [all_generators[i] for i in user_input]

        for absolute_path in absolute_paths:
            dataset = loaded_datasets[absolute_path]['csv']
            dataset = dataset[~dataset['generator'].isin(user_input)]

            # result_name = os.path.basename(absolute_path) + ''.join(['_no' + generator for generator in user_input])
            result_name = os.path.basename(absolute_path)

            Path(result_directory).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(result_directory, result_name)).mkdir(parents=True, exist_ok=True)

            dataset.to_csv(
                os.path.join(result_directory, result_name, os.path.basename(absolute_path) + '.csv'),
                index=False
            )
            for logs in loaded_datasets[absolute_path]['log']:
                shutil.copy(
                    logs,
                    os.path.join(result_directory, result_name, os.path.basename(logs))
                )


def remove_random_state(directory_path, interrupt=True):
    result_directory = os.path.join(directory_path, 'processed')

    groups = group_runs(directory_path, split_by_suffix=True)
    print('found the following groups:\n' + '\n'.join(groups.keys()) + '\n')

    all_randomstates = [
        1, 2, 3, 4, 5, 6, 8, 10, 16
    ]

    if not interrupt:
        user_input = input(
            'which randomstates do you want to remove from all groups if available ?'
            + ' valid inputs are for example "", "1" or "1 2"\n'
            + '\n'.join([
                '[' + str(index) + '] ' + str(random_state) for index, random_state in enumerate(all_randomstates)
            ]) + '\n'
        )
        if user_input == '':
            user_input = []
        else:
            user_input = user_input.split(' ')
            if False in [entry.isdigit() for entry in user_input]:
                print('invalid input')
                exit(1)
            user_input = [int(number) for number in user_input]
            for number in user_input:
                if number > len(all_randomstates):
                    print('invalid input')
                    exit(1)
            user_input = [all_randomstates[i] for i in user_input]

    for group in groups:
        dataset_name, absolute_paths, random_states, suffixes = extract_group_data(groups, group)

        loaded_datasets = {}
        randomstates = []
        log_file_paths = []
        for absolute_path in absolute_paths:
            loaded_datasets[absolute_path] = {}
            loaded_datasets[absolute_path]['csv'] = pd.read_csv(
                get_file_path_in_directory(absolute_path, '.csv'),
                index_col=False
            )
            loaded_datasets[absolute_path]['log'] = get_all_file_paths_in_directory(
                absolute_path,
                '.log'
            )
            randomstates.extend(list(loaded_datasets[absolute_path]['csv']['random_state'].unique()))
        randomstates = list(set(randomstates))

        if interrupt:
            user_input = input(
                'which randomstates do you want to remove from ' + group
                + ' ? valid inputs are for example "", "1" or "1 2"\n'
                + '\n'.join([
                    '[' + str(index) + '] ' + str(random_state) for index, random_state in enumerate(randomstates)
                ]) + '\n'
            )
            if user_input == '':
                user_input = []
            else:
                user_input = user_input.split(' ')
                if False in [entry.isdigit() for entry in user_input]:
                    print('invalid input')
                    exit(1)
                user_input = [int(number) for number in user_input]
                for number in user_input:
                    if number > len(randomstates):
                        print('invalid input')
                        exit(1)
                user_input = [int(randomstates[i]) for i in user_input]

        for absolute_path in absolute_paths:
            dataset = loaded_datasets[absolute_path]['csv']
            dataset = dataset[~dataset['random_state'].isin(user_input)]

            # result_name = os.path.basename(absolute_path) + ''.join(['_no' + random_state for random_state in user_input])
            result_name = os.path.basename(absolute_path)

            Path(result_directory).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(result_directory, result_name)).mkdir(parents=True, exist_ok=True)

            dataset.to_csv(
                os.path.join(result_directory, result_name, os.path.basename(absolute_path) + '.csv'),
                index=False
            )
            for logs in loaded_datasets[absolute_path]['log']:
                shutil.copy(
                    logs,
                    os.path.join(result_directory, result_name, os.path.basename(logs))
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


def run(interrupt=True):
    print('what do you want to do ?\n')

    _, routine = get_user_selection([
        ('combine randomstates', combine_random_state),
        ('merge runs', merge_runs),
        ('remove generator', remove_generator),
        ('remove randomstates', remove_random_state)
    ])

    print('select the root folder\n')
    run_directory = askdirectory(title='Select Folder')
    if run_directory == '':
        print('invalid path')
        exit(1)

    routine(run_directory, interrupt)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='A script for processing the generated results.'
    )
    parser.add_argument('--interrupt', action='store_true')
    parser.add_argument('--no-interrupt', dest='interrupt', action='store_false')
    parser.set_defaults(feature=True)

    args = parser.parse_args()

    run(args.interrupt)
