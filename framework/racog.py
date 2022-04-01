"""Wrapper for the R implementation of RACOG (Rapidly converging Gibbs algorithm)"""

# Authors: Thomas Frank <thomas-frank01@gmx.de>
# License: MIT

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri

class RACOG:
    """Wrapper for the R implementation of RACOG (Rapidly converging Gibbs algorithm).

    Approximates minority distribution using Gibbs Sampler. Dataset must be
    discretized and numeric. In each iteration, it builds a new sample using a
    Markov chain. It discards first burnin iterations, and from then on, each
    lag iterations, it validates the example as a new minority example. It
    generates d (iterations-burnin)/lag where d is minority examples number.

    Parameters
    ----------

    burnin : int, default=100
        It determines how many examples generated for a given one are going to
        be discarded firstly.

    lag : int, default=20
        Number of iterations between new generated example for a minority one.

    References
    ----------

    .. [1] https://rdrr.io/cran/imbalance/man/racog.html

    .. [2] B. Das, N. C. Krishnan and D. J. Cook, "RACOG and wRACOG: Two
           Probabilistic Oversampling Techniques," in IEEE Transactions on
           Knowledge and Data Engineering, vol. 27, no. 1, pp. 222-234,
           1 Jan. 2015, doi: 10.1109/TKDE.2014.2324567.

    """
    def __init__(
        self,
        burnin=100,
        lag=20
    ):
        self._burnin = burnin
        self._lag = lag

    def resample(
        self,
        dataset,
        num_instances,
        class_attr
    ):
        """Perform Gibbs sampling on the given dataset.

        Parameters
        ----------
        dataset : pandas.DataFrame of shape (n_samples, n_features)
            Dataframe to treat. All columns, except classAttr one, have to be
            numeric or coercible to numeric.

        num_instances : int
            Number of new minority examples to generate.

        class_attr : str
            Indicates the class attribute from dataset. Must exist in it.

        Returns
        -------
        resampled_dataset : pandas.DataFrame of shape (num_instances, n_features)
            Dataset containing the generated synthetic samples.
        """
        original_column_titles = dataset.columns

        with localconverter(ro.default_converter + pandas2ri.converter):
            dataset_R = ro.conversion.py2rpy(dataset)

        imbalance = importr('imbalance')
        resampled_dataset_R = imbalance.racog(
            dataset=dataset_R,
            numInstances=num_instances,
            burnin=self._burnin,
            lag=self._lag,
            classAttr=class_attr
        )

        with localconverter(ro.default_converter + pandas2ri.converter):
            resampled_dataset = ro.conversion.rpy2py(resampled_dataset_R)

        resampled_dataset.columns = original_column_titles

        return resampled_dataset