import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

    def compact_run(self):
        Path(self._target_directory).mkdir(parents=True, exist_ok=True)

        dataset = self._dataset.replace({
            'TeacherLabeledAugmentedStudentPipeline': 'TeacherLabeled',
            'DirectGeneratorLabeledAugmentedStudentPipeline': 'DirGeneratorLabeled',
            'IndirectGeneratorLabeledAugmentedStudentPipeline': 'IndirGeneratorLabeled',
            'ProportionalSMOTESampler': 'PropSMOTE',
            'UnlabeledSMOTESampler': 'UnlbSMOTE',
            'ProportionalConditionalGANSampler': 'PropCGAN',
            'UnlabeledConditionalGANSampler': 'UnlbCGAN',
        })

        sns.set_style("darkgrid")

        dataset_without_teacher_pipelines = dataset.drop(
            dataset[dataset.pipeline == 'TeacherPipeline'].index
        )

        dataset_without_gan_samplers = dataset.drop(
            dataset[dataset.sampler == 'PropCGAN'].index
        )
        dataset_without_gan_samplers = dataset_without_gan_samplers.drop(
            dataset_without_gan_samplers[dataset_without_gan_samplers.sampler == 'UnlbCGAN'].index
        )

        dataset_without_gan_samplers_without_teacher_pipelines = dataset_without_gan_samplers.drop(
            dataset_without_gan_samplers[dataset_without_gan_samplers.pipeline == 'TeacherPipeline'].index
        )

        baselines = {}
        datasets = dataset.groupby(by=['dataset', 'encoder'])
        for name, dataset in datasets:
            student_baseline = dataset[
                dataset.sample_multiplication_factor == 0
            ]['test_score'].reset_index(drop=True)[0]
            teacher_baseline = dataset[
                dataset.pipeline == 'TeacherPipeline'
            ]['test_score'].reset_index(drop=True)[0]

            baseline = {
                'student': student_baseline,
                'teacher': teacher_baseline
            }

            dataset, encoder = name
            if dataset in baselines:
                baselines[dataset][encoder] = baseline
            else:
                baselines[dataset] = {}
                baselines[dataset][encoder] = baseline

        datasets = dataset_without_teacher_pipelines.groupby(by=['dataset'])
        for name, dataset in datasets:
            name = [name] if isinstance(name, str) else name

            self._save_catplot(
                title='encoders ' + '_'.join(name),
                dataset=dataset,
                x='test_score',
                y='encoder',
                kind='box',
                hue='pipeline',
                show_outliers=False
            )

            self._save_catplot_with_baseline(
                title='samplers ' + '_'.join(name),
                dataset=dataset,
                y='sampler',
                col='encoder',
                baselines=baselines[name[0]]
            )

        datasets = dataset_without_teacher_pipelines.groupby(by=['dataset', 'encoder'])
        for name, dataset in datasets:
            name = [name] if isinstance(name, str) else name

            self._save_catplot_with_baseline(
                title='samplers ' + '_'.join(name),
                dataset=dataset,
                y='sampler',
                baselines=baselines[name[0]][name[1]]
            )

        datasets = dataset_without_gan_samplers_without_teacher_pipelines.groupby(by=['dataset'])
        for name, dataset in datasets:
            name = [name] if isinstance(name, str) else name

            self._save_relplot(
                title='augmentation ' + '_'.join(name),
                dataset=dataset,
                kind='line',
                x='sample_multiplication_factor',
                y='test_score',
                col='encoder',
            )

            self._save_catplot_with_baseline(
                title='pipelines ' + '_'.join(name),
                dataset=dataset,
                y='pipeline',
                col='encoder',
                baselines=baselines[name[0]]
            )

        datasets = dataset_without_gan_samplers_without_teacher_pipelines.groupby(by=['dataset', 'encoder'])
        for name, dataset in datasets:
            name = [name] if isinstance(name, str) else name

            self._save_relplot(
                title='augmentation ' + '_'.join(name),
                dataset=dataset,
                kind='line',
                x='sample_multiplication_factor',
                y='test_score'
            )

            self._save_relplot_with_baseline(
                title='augmentation_with_baseline ' + '_'.join(name),
                dataset=dataset,
                kind='line',
                x='sample_multiplication_factor',
                y='test_score',
                baseline=baselines[name[0]][name[1]]['student']
            )

            self._save_catplot_with_baseline(
                title='pipelines ' + '_'.join(name),
                dataset=dataset,
                y='pipeline',
                baselines=baselines[name[0]][name[1]]
            )

        self._save_relplot(
            title='augmentation',
            dataset=dataset_without_gan_samplers_without_teacher_pipelines,
            kind='line',
            x='sample_multiplication_factor',
            y='test_score',
            col='dataset',
            row='encoder'
        )

        self._save_catplot(
            title='encoders',
            dataset=dataset_without_teacher_pipelines,
            x='test_score',
            y='encoder',
            kind='box',
            hue='pipeline',
            col='dataset',
            show_outliers=False
        )

        self._save_catplot(
            title='general performance',
            dataset=dataset_without_gan_samplers_without_teacher_pipelines,
            x='test_score',
            y='dataset',
            kind='box',
            hue='encoder',
            show_outliers=False
        )

    def detailed_run(self):
        Path(self._target_directory).mkdir(parents=True, exist_ok=True)

        base_title = 'samplers'
        datasets = self._dataset.groupby(by=['dataset', 'encoder'])
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
        datasets = self._dataset.groupby(by=['dataset', 'sampler'])
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
        datasets = self._dataset.groupby(by=['dataset', 'encoder'])
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
        plt.close()

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
        plt.close()

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
        plt.close()

    def _save_catplot(
            self,
            title,
            dataset,
            kind,
            x=None,
            y=None,
            hue=None,
            row=None,
            col=None,
            show_outliers=True
    ):
        if show_outliers:
            plot = sns.catplot(
                data=dataset,
                x=x,
                y=y,
                kind=kind,
                hue=hue,
                row=row,
                col=col
            )
        else:
            plot = sns.catplot(
                data=dataset,
                x=x,
                y=y,
                kind=kind,
                hue=hue,
                row=row,
                col=col,
                sym=''
            )

        plot.fig.subplots_adjust(top=0.9)
        plot.fig.suptitle(title)

        plot.figure.savefig(
            os.path.join(self._target_directory, title + '.png'),
            bbox_inches='tight'
        )

        plot.figure.clf()
        plt.close()

    def _save_catplot_with_baseline(
            self,
            title,
            dataset,
            y,
            baselines,
            col=None
    ):
        plot = sns.catplot(
            data=dataset,
            x='test_score',
            y=y,
            kind='box',
            col=col,
            sym=''
        )

        if col is not None:
            for encoder in plot.axes_dict:
                plot.axes_dict[encoder].axvline(
                    baselines[encoder]['student'],
                    color='red',
                    ls='--',
                    linewidth=3,
                    alpha=0.6
                )
                plot.axes_dict[encoder].axvline(
                    baselines[encoder]['teacher'],
                    color='black',
                    ls='--',
                    linewidth=3,
                    alpha=0.6
                )
        else:
            plot.refline(
                x=baselines['student'],
                color='red',
                linestyle='--',
                linewidth=3,
                alpha=0.6
            )
            plot.refline(
                x=baselines['teacher'],
                color='black',
                linestyle='--',
                linewidth=3,
                alpha=0.6
            )

        student_linetype = Line2D([0, 1], [0, 1], linestyle='--', color='red', linewidth=3, alpha=0.6)
        teacher_linetype = Line2D([0, 1], [0, 1], linestyle='--', color='black', linewidth=3, alpha=0.6)
        plot.fig.legend(
            handles=[student_linetype, teacher_linetype],
            labels=['student', 'teacher'],
            title='baselines',
            loc='center left',
            bbox_to_anchor=(1, 0.5)
        )

        plot.fig.subplots_adjust(top=0.9)
        plot.fig.suptitle(title)

        plot.figure.savefig(
            os.path.join(self._target_directory, title + '.png'),
            bbox_inches='tight'
        )

        plot.figure.clf()
        plt.close()

    def _save_relplot(
            self,
            title,
            dataset,
            kind,
            x=None,
            y=None,
            hue=None,
            row=None,
            col=None
    ):
        plot = sns.relplot(
            data=dataset,
            x=x,
            y=y,
            kind=kind,
            hue=hue,
            row=row,
            col=col
        )

        plot.fig.subplots_adjust(top=0.9)
        plot.fig.suptitle(title)

        plot.figure.savefig(
            os.path.join(self._target_directory, title + '.png'),
            bbox_inches='tight'
        )

        plot.figure.clf()
        plt.close()

    def _save_relplot_with_baseline(
            self,
            title,
            dataset,
            kind,
            baseline,
            x=None,
            y=None,
            hue=None,
            row=None,
            col=None
    ):
        plot = sns.relplot(
            data=dataset,
            x=x,
            y=y,
            kind=kind,
            hue=hue,
            row=row,
            col=col
        )

        plot.refline(
            y=baseline,
            color='red',
            linestyle='--',
            linewidth=3,
            alpha=0.6
        )
        plot.refline(
            y=baseline,
            color='black',
            linestyle='--',
            linewidth=3,
            alpha=0.6
        )

        student_linetype = Line2D([0, 1], [0, 1], linestyle='--', color='red', linewidth=3, alpha=0.6)
        plot.fig.legend(
            handles=[student_linetype],
            labels=['student'],
            title='baselines',
            loc='center left',
            bbox_to_anchor=(1, 0.5)
        )

        plot.fig.subplots_adjust(top=0.9)
        plot.fig.suptitle(title)

        plot.figure.savefig(
            os.path.join(self._target_directory, title + '.png'),
            bbox_inches='tight'
        )

        plot.figure.clf()
        plt.close()


run_folders = [
    'proposal_run_20220422_193545'
]
for folder in run_folders:
    source_dir = os.path.join(os.getcwd(), 'experiments', 'runs', folder)
    target_dir = os.path.join(source_dir, 'results')

    visualizer = Visualizer(
        source_directory=source_dir,
        target_directory=target_dir
    )
    visualizer.load_datasets()
    visualizer.compact_run()
