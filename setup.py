# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='tabular-data-augmentation',
    version='0.1.0',
    description='A framework for comparing different tabular data augmentation pipelines and accompanying experiments '
                'using it.',
    long_description=readme,
    author='Thomas Frank',
    author_email='thomas-frank01@gmx.de',
    url='https://github.com/thomas475/tabular-data-augmentation',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)