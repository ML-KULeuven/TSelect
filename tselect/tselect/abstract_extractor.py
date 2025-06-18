import time
from abc import ABC, abstractmethod
from typing import Union, Dict, Callable

import pandas as pd

from tselect import FusionFilter
from tselect.utils import *
from tselect.utils.constants import SEED, Keys
from tsfuse.data import Collection

from tselect.config import Config, get_default_config


class AbstractExtractor(ABC):
    """
    An abstract class for filters. It contains the basic functionality that all filters should have.
    """

    def __init__(self, series_fusion: bool = True,
                 tselect_config: Config = get_default_config(),
                 # irrelevant_filter=True,
                 # redundant_filter=True,
                 # auc_percentage: float = 0.75,
                 # auc_threshold: float = 0.5,
                 # corr_threshold: float = 0.7,
                 # feature_extractor: Callable =None,
                 # test_size: float = None,
                 views: List[int] = None,
                 add_tags=lambda x: x,
                 compatible=lambda x: x,
                 random_state: int = SEED, ):
        """
        The constructor for AbstractExtractor class.

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
        test_size : float, optional, default None
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
        random_state : int, optional, default SEED
            The random state used throughout the class.
        """
        self.fusion_filter: FusionFilter = FusionFilter(series_fusion=series_fusion,
                                                        tselect_config=tselect_config,
                                                        # irrelevant_filter=irrelevant_filter,
                                                        # redundant_filter=redundant_filter,
                                                        # auc_percentage=auc_percentage,
                                                        # auc_threshold=auc_threshold,
                                                        # corr_threshold=corr_threshold,
                                                        # feature_extractor=feature_extractor,
                                                        # test_size=test_size,
                                                        views=views,
                                                        add_tags=add_tags,
                                                        compatible=compatible,
                                                        random_state=random_state)

    @property
    def series_filter(self):
        return self.fusion_filter.series_filter

    def set_series_filter(self, series_filter):
        self.fusion_filter.series_filter = series_filter

    @abstractmethod
    def transform_model(self, X):
        """
        Transform the data by applying the model. This function is called in the transform function and should be
        implemented by the child class.

        Parameters
        ----------
        X : pd.DataFrame
            The data to transform in the MultiIndex Pandas format.

        Returns
        -------
        X : pd.DataFrame
            The transformed data.
        """
        pass

    def transform(self, X: Union[pd.DataFrame, Dict[Union[str, int], Collection]]):
        """
        Transform the data by applying fusion, filtering and the model.

        Parameters
        ----------
        X : Union[pd.DataFrame, Dict[Union[str, int], Collection]]
            The data to transform in the MultiIndex Pandas format or the TSFuse format.

        Returns
        -------
        X : pd.DataFrame
            The transformed data in the MultiIndex Pandas format.
        """
        X_pd = self.fusion_filter.transform(X, return_format='dataframe')
        return self.transform_model(X_pd)

    @abstractmethod
    def fit_model(self, X, y):
        """
        Fit the model. This function is called in the fit function and should be implemented by the child class.

        Parameters
        ----------
        X : pd.DataFrame
            The data to fit in the MultiIndex Pandas format.
        y : pd.Series
            The target variable.

        Returns
        -------
        None, but the model should be fitted.
        """
        pass

    def fit(self, X: Union[pd.DataFrame, Dict[Union[str, int], Collection]], y, metadata=None):
        """
        Fit the model by applying fusion, filtering and fitting the model.

        Parameters
        ----------
        X : Union[pd.DataFrame, Dict[Union[str, int], Collection]]
            The data to fit in the MultiIndex Pandas format or the TSFuse format.
        y : pd.Series
            The target variable.
        metadata : Dict[str, List[float]], optional, default None
            A dictionary containing the metadata of the experiment. If None, no metadata is collected.

        Returns
        -------
        None, but the model and filter should be fitted.
        """
        X_pd = self.fusion_filter.fit(X, y, metadata, return_format='dataframe')

        print("     Executing model")
        start = time.process_time()
        self.fit_model(X_pd, y)
        if metadata:
            metadata[Keys.time_series_to_attr].append(time.process_time() - start)
        return None
