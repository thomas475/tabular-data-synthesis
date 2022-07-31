"""SDGym - Synthetic data Gym.

SDGym is a framework to benchmark the performance of synthetic data generators for
tabular data.
"""

__author__ = 'MIT Data To AI Lab'
__copyright__ = 'Copyright (c) 2018, MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__license__ = 'MIT'
__version__ = '0.5.1.dev0'

import logging

from framework.generators.sdgym import synthesizers

# Clear the logging wrongfully configured by tensorflow/absl
list(map(logging.root.removeHandler, logging.root.handlers))
list(map(logging.root.removeFilter, logging.root.filters))

__all__ = [
    'synthesizers',
]
