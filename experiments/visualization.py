import seaborn as sns
import pandas as pd
import os
from pathlib import Path
import glob


class Visualizer:
    def __init__(self, source_directory, target_directory):
        self._source_directory = source_directory
        self._target_directory = target_directory
        self._dataset = None

    def load_datasets(self):
        csv_files = glob.glob(os.path.join(self._source_directory, '*.csv'))

        csv_dataframes = []
        for file in csv_files:
            csv_dataframes.append(pd.read_csv(file))

        self._dataset = pd.concat(csv_dataframes, ignore_index=True).reset_index(drop=True)

    def run(self):
        Path(self._target_directory).mkdir(parents=True, exist_ok=True)

        base_title = 'samplers'
        datasets = self._dataset.groupby(by=['dataset'])
        for name, dataset in datasets:
            name = [name] if isinstance(name, str) else name
            dataset = dataset.dropna(subset=['sampler'])
            if dataset.empty:
                continue
            self._save_box_plot(
                title=base_title + ' ' + '_'.join(name),
                x=dataset['sampler'],
                y=dataset['test_score'],
                show_outliers=True
            )
            self._save_box_plot(
                title=base_title + ' ' + '_'.join(name),
                x=dataset['sampler'],
                y=dataset['test_score'],
                show_outliers=False
            )

        base_title = 'encoders'
        datasets = self._dataset.groupby(by=['dataset'])
        for name, dataset in datasets:
            name = [name] if isinstance(name, str) else name
            dataset = dataset.dropna(subset=['encoder'])
            if dataset.empty:
                continue
            self._save_box_plot(
                title=base_title + ' ' + '_'.join(name),
                x=dataset['encoder'],
                y=dataset['test_score'],
                show_outliers=True
            )
            self._save_box_plot(
                title=base_title + ' ' + '_'.join(name),
                x=dataset['encoder'],
                y=dataset['test_score'],
                show_outliers=False
            )

        base_title = 'pipelines'
        datasets = self._dataset.groupby(by=['dataset'])
        for name, dataset in datasets:
            name = [name] if isinstance(name, str) else name
            dataset = dataset.dropna(subset=['pipeline'])
            if dataset.empty:
                continue
            self._save_box_plot(
                title=base_title + ' ' + '_'.join(name),
                x=dataset['pipeline'],
                y=dataset['test_score'],
                show_outliers=True
            )
            self._save_box_plot(
                title=base_title + ' ' + '_'.join(name),
                x=dataset['pipeline'],
                y=dataset['test_score'],
                show_outliers=False
            )

        base_title = 'augmentation'
        datasets = self._dataset.groupby(by=['dataset', 'sampler'])
        for name, dataset in datasets:
            name = [name] if isinstance(name, str) else name
            dataset = dataset.dropna(subset=['sample_multiplication_factor'])
            if dataset.empty:
                continue
            self._save_line_plot(
                title=base_title + ' ' + '_'.join(name),
                x=dataset['sample_multiplication_factor'],
                y=dataset['test_score']
            )

    def _save_box_plot(self, title, x, y, show_outliers=True):
        if show_outliers:
            plot = sns.boxplot(
                x=x,
                y=y
            )
        else:
            plot = sns.boxplot(
                x=x,
                y=y,
                showfliers=False
            )
            ylims = plot.get_ylim()

        plot = sns.stripplot(
            x=x,
            y=y,
            size=4,
            color=".3",
            linewidth=0
        )
        plot.set_xticklabels(plot.get_xticklabels(), rotation=40)
        plot.set(title=title)

        if not show_outliers:
            plot.set(ylim=ylims)
            title = title + '_no_outliers'

        plot.figure.savefig(
            os.path.join(self._target_directory, title + '.png'),
            bbox_inches='tight'
        )

        plot.figure.clf()

    def _save_scatter_plot(self, title, x, y):
        plot = sns.scatterplot(
            x=x,
            y=y
        )

        # plot.set_xticklabels(plot.get_xticklabels(), rotation=40)
        plot.set(title=title)

        plot.figure.savefig(
            os.path.join(self._target_directory, title + '.png'),
            bbox_inches='tight'
        )

        plot.figure.clf()

    def _save_line_plot(self, title, x, y):
        plot = sns.lineplot(
            x=x,
            y=y
        )

        # plot.set_xticklabels(plot.get_xticklabels(), rotation=40)
        plot.set(title=title)

        plot.figure.savefig(
            os.path.join(self._target_directory, title + '.png'),
            bbox_inches='tight'
        )

        plot.figure.clf()


run_folders = [
    'proposal_run_20220422_193545',
    'proposal_run_20220424_151611',
]
for folder in run_folders:
    source_directory = os.path.join(os.getcwd(), 'experiments', 'runs', folder)
    target_directory = os.path.join(source_directory, 'results')

    visualizer = Visualizer(
        source_directory=source_directory,
        target_directory=target_directory
    )
    visualizer.load_datasets()
    visualizer.run()
