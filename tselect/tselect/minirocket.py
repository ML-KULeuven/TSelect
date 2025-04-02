from typing import List, Callable

import pandas as pd
from sklearn.base import TransformerMixin

from tselect import SEED
from tselect.abstract_extractor import AbstractExtractor
from tselect.utils import reset_first_level_index, pad_until_length
from sktime.transformations.panel.rocket import (
    MiniRocketMultivariateVariable,
)

from tselect.config import Config, get_default_config


class MiniRocketExtractor(AbstractExtractor, TransformerMixin):
    """
    A class to extract features from time series using MiniRocket. It is a wrapper around sktime's
    MiniRocketMultivariate class. Additionally, it can derive new signals from the original ones ("fusion"), and
    filter out irrelevant ("irrelevant filter") and redundant signals ("redundant filter").
    """

    def __init__(self, series_fusion: bool = False,
                 tselect_config: Config = get_default_config(),
                #  irrelevant_filter: bool = False,
                #  redundant_filter: bool = False,
                #  auc_percentage: float = 0.75,
                #  auc_threshold: float = 0.5,
                #  corr_threshold: float = 0.7,
                # feature_extractor: Callable = None,
                #  test_size: float = 0.2,
                 views: List[int] = None,
                 add_tags=lambda x: x,
                 compatible=lambda x: x,
                 num_kernels: int = 10_000,
                 max_dilations_per_kernel: int = 32,
                 n_jobs: int = 1,
                 random_state: int = SEED):
        """
        The constructor for MiniRocketExtractor class.

        Parameters
        ----------
        series_fusion : bool, optional, default False
            Whether to derive new signals from the original ones ("fusion").
        irrelevant_filter : bool, optional, default False
            Whether to filter out irrelevant signals ("irrelevant filter").
        redundant_filter : bool, optional, default False
            Whether to filter out redundant signals ("redundant filter").
        auc_percentage : float, optional, default 0.75
            The percentage of the time series that will remain after the irrelevant filter. If the auc_threshold is
            0.75, the 75% time series with the highest AUC will remain.
        auc_threshold : float, optional, default 0.5
            The threshold for the irrelevant filter. If the auc_threshold is 0.5, all series with an AUC lower than
            0.5 will be removed, regardless of the specified auc_percentage. After all signals with an AUC lower than
            this threshold are removed, the auc_percentage will be applied.
        corr_threshold : float, optional, default 0.7
            The threshold used for clustering rank correlations. All predictions with a rank correlation above this
             threshold are considered correlated.
        test_size : float, optional, default 0.2
            The test size to use for filtering out irrelevant series based on their AUC score. The test size is the
            percentage of the data that is used for computing the AUC score. The remaining data is used for training.
            If None, the train size is derived from max(100, 0.25*nb_instances). The test size are then the remaining
            instances.
        views : list of int, optional, default None
             The different views of the data. This parameter is used to convert to the internal TSFuse Collection
             format, that groups the dimensions of the data in the unique sensors. For more information on this,
             we refer to https://github.com/arnedb/tsfuse
        add_tags: Callable, default=lambda x: x
            A function that adds tags to the data. This parameter is used to convert to the internal TSFuse Collection
            and adds tags to the Collection, describing what sensors were used for the different dimensions.
            For more information on this, we refer to https://github.com/arnedb/tsfuse
        compatible: Callable, default=lambda x: x
            A function that adds tags to the data. This parameter is used to convert to the internal TSFuse Collection
            and describes what dimensions can be combined to derive new series. For more information on this,
            we refer to https://github.com/arnedb/tsfuse
        num_kernels : int, optional, default 10_000
            The number of kernels to use for the MiniRocket transformation.
        max_dilations_per_kernel : int, optional, default 32
            The maximum number of dilations per kernel to use for the MiniRocket transformation.
        n_jobs : int, optional, default 1
            The number of jobs to use MiniRocket.
        random_state : int, optional, default SEED
            The random state used throughout the class.
        """
        super().__init__(series_fusion, tselect_config, views, add_tags, compatible, random_state)
        self.minirocket = MiniRocketMultivariateVariable(num_kernels=num_kernels,
                                                         max_dilations_per_kernel=max_dilations_per_kernel,
                                                         n_jobs=n_jobs,
                                                         random_state=random_state)

    def transform_model(self, X: pd.DataFrame):
        """
        Transforms the data using MiniRocket.

        Parameters
        ----------
        X : pd.DataFrame
            The data to transform.

        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        if X.index.levels[1].shape[0] < 9:
            X = pad_until_length(X, 9)
        
        return self.minirocket.transform(X)

    def fit_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits MiniRocket to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data to fit MiniRocket to.
        y : pd.Series
            The target values.
        """
        if X.index.levels[1].shape[0] < 9:
            X = pad_until_length(X, 9)

        X_mini, y_mini = reset_first_level_index(X, y)
        self.minirocket.fit(X_mini, y_mini)
        return None
