import time
import warnings
from typing import Union, Dict

import pandas as pd
from sklearn.base import TransformerMixin

from tselect.channel_selectors.tselect import TSelect
from tselect.utils import *
from tselect.utils.constants import SEED, Keys
from tselect.utils.scaler import MinMaxScaler3D
from tsfuse.computation import Input
from tsfuse.construction.mlj20 import TSFuseExtractor
from tsfuse.data import dict_collection_to_pd_multiindex, Collection, pd_multiindex_to_dict_collection


class FusionFilter(TransformerMixin):
    """
    An abstract class for filters. It contains the basic functionality that all filters should have.
    """

    def __init__(self, series_fusion: bool = True,
                 irrelevant_filter=True,
                 redundant_filter=True,
                 auc_percentage: float = 0.75,
                 auc_threshold: float = 0.5,
                 corr_threshold: float = 0.7,
                 test_size: float = None,
                 views: List[int] = None,
                 add_tags=lambda x: x,
                 compatible=lambda x: x,
                 random_state: int = SEED, ):
        """
        The constructor for AbstractFilter class.

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
        self.series_fusion = series_fusion
        self.irrelevant_filter = irrelevant_filter
        self.redundant_filter = redundant_filter
        self.series_filtering = irrelevant_filter or redundant_filter
        self.auc_percentage = auc_percentage
        self.auc_threshold = auc_threshold
        self.corr_threshold = corr_threshold
        self.test_size = test_size
        self.views = views
        self.add_tags = add_tags
        self.compatible = compatible
        self.random_state = random_state

        self.tsfuse_extractor = TSFuseExtractor(transformers='full', compatible=compatible, random_state=SEED)
        self.__init_filter__()
        if self.series_filtering:
            self.tsfuse_extractor.series_filter = self.series_filter

        self.included_inputs = []
        self.nodes_translation = {}

    def __init_filter__(self):
        self.series_filter = TSelect(irrelevant_filter=self.irrelevant_filter, redundant_filter=self.redundant_filter,
                                     random_state=SEED, auc_percentage=self.auc_percentage,
                                     filtering_threshold_corr=self.corr_threshold,
                                     filtering_test_size=self.test_size) if self.series_filtering else None

    def transform_fusion(self, X_tsfuse: Dict[Union[str, int], Collection]) -> Dict[Union[str, int], Collection]:
        """
        Transform the data by fusing the series. This function is called in the transform function and should be
        implemented by the child class if non-default behavior is required.

        Parameters
        ----------
        X_tsfuse : Dict[Union[str, int], Collection]
            The data to transform in the TSFuse format.

        Returns
        -------
        X_tsfuse : Dict[Union[str, int], Collection]
            The transformed data in the TSFuse format.
        """
        dict_collection = self.tsfuse_extractor.transform(X_tsfuse, return_dataframe=False)
        dict_collection = {self.nodes_translation[k]: v for k, v in dict_collection.items()}
        inputs = {f'Input({i.name})': X_tsfuse[i.name] for i in self.included_inputs}
        dict_collection.update(inputs)
        if isinstance(self.series_filter, TSelect):
            dict_collection = self.series_filter.scaler.transform(dict_collection)
        return dict_collection

    def transform_filter(self, X: Union[pd.DataFrame, Dict[Union[str, int], Collection]]):
        """
        Transform the data by filtering the series. This function is called in the transform function and should be
        implemented by the child class if non-default behavior is required.

        Parameters
        ----------
        X : pd.DataFrame or Dict[Union[str, int], Collection]
            The data to transform in the MultiIndex Pandas or TSFuse format.

        Returns
        -------
        X : pd.DataFrame or Dict[Union[str, int], Collection]
            The transformed data in the MultiIndex Pandas or TSFuse format.
        """
        if isinstance(X, pd.DataFrame):
            rename_columns_pd(X, self.nodes_translation)
        elif isinstance(X, dict):
            X = rename_keys_dict(X, self.nodes_translation)
        return self.series_filter.transform(X)

    def transform(self, X: Union[pd.DataFrame, Dict[Union[str, int], Collection]], return_format='dataframe'):
        """
        Transform the data by applying fusion, filtering and the model.

        Parameters
        ----------
        X : Union[pd.DataFrame, Dict[Union[str, int], Collection]]
            The data to transform in the MultiIndex Pandas format or the TSFuse format.
        return_format : str, optional, default 'dataframe'
            The return format of the data. Can be 'dataframe' or 'tsfuse'.

        Returns
        -------
        X : pd.DataFrame
            The transformed data in the MultiIndex Pandas format.
        """
        if isinstance(X, pd.DataFrame):
            X_pd = X
            X_tsfuse = None
            input_format = 'dataframe'
        else:
            X_pd = None
            X_tsfuse = X
            input_format = 'tsfuse'
        if self.series_fusion:
            if not X_tsfuse:
                X_tsfuse = get_tsfuse_format(X, views=self.views, add_tags=self.add_tags)
            X_tsfuse = self.transform_fusion(X_tsfuse)

        if self.series_filter and (not self.series_fusion):
            # If the series are fused, the tsfuse extractor takes care of only returning the filtered series
            X = self.transform_filter(X)
            if isinstance(X, pd.DataFrame):
                X_pd = X
            else:
                X_tsfuse = X

        return self.transform_to_output_format(X_pd, X_tsfuse, input_format, return_format)

    def fit(self, X: Union[pd.DataFrame, Dict[Union[str, int], Collection]], y, metadata=None,
            return_format='dataframe'):
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
        return_format : str, optional, default 'dataframe'
            The return format of the data. Can be 'dataframe' or 'tsfuse'.

        Returns
        -------
        X : pd.DataFrame or Dict[Union[str, int], Collection]
            The fitted data in the MultiIndex Pandas format or TSFuse format, depending on the specified
            `return_format`.
        """
        if isinstance(X, pd.DataFrame):
            X_pd = X
            X_tsfuse = None
            input_format = 'pd'
        else:
            X_pd = None
            X_tsfuse = X
            input_format = 'tsfuse'

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.series_fusion:
                print("     Series to series")
                start_fusion = time.process_time()
                X_tsfuse = get_tsfuse_format(X, views=self.views, add_tags=self.add_tags)
                data = self.tsfuse_extractor.initialize(X_tsfuse, y)
                self.tsfuse_extractor.series_to_series(data, metadata, select_non_redundant=False)
                X_tsfuse = self.tsfuse_extractor.get_selected_series(data)
                X_tsfuse = {str(k): v for k, v in X_tsfuse.items()}
                print("         Number of fused signals: ", len(self.tsfuse_extractor.series_))
                print("         Fusion time: ", time.process_time() - start_fusion)

            if self.series_filter is not None:
                print("     Filtering series")
                start = time.process_time()
                if X_tsfuse is None:
                    self.series_filter.fit(X_pd, y, metadata)
                    X_pd = X_pd[self.series_filter.selected_channels]
                else:
                    self.series_filter.fit(X_tsfuse, y, metadata)
                    X_tsfuse = {k: v for k, v in X_tsfuse.items() if k in self.series_filter.selected_channels}
                if self.series_fusion:
                    self.tsfuse_extractor.set_subset_selected_series(self.series_filter.selected_channels)
                metadata[Keys.time_series_filtering].append(time.process_time() - start)
                print("         Number of series after filtering: ", len(self.series_filter.selected_channels))
                print("         Total filtering time: ", time.process_time() - start)

            # After filtering because fewer nodes should be added then
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.series_fusion:
                    for n in self.tsfuse_extractor.series_:
                        n._is_output = True
                        new_node = self.tsfuse_extractor.graph_.add_node(n)
                        if isinstance(n, (Input, int, str)):
                            self.included_inputs.append(n)
                            self.nodes_translation[new_node.name] = n
                        else:
                            self.nodes_translation[new_node] = n

            return self.transform_to_output_format(X_pd, X_tsfuse, input_format, return_format)

    def transform_to_output_format(self, X_pd, X_tsfuse, input_format, return_format):
        if return_format == 'tsfuse':
            if input_format == 'tsfuse' or self.series_fusion:
                return X_tsfuse
            else:
                return pd_multiindex_to_dict_collection(X_pd, views=self.views, add_tags=self.add_tags)

        elif return_format == 'dataframe':
            if input_format == 'tsfuse' or self.series_fusion:
                X_pd = dict_collection_to_pd_multiindex(X_tsfuse, index=X_pd.index if X_pd is not None else None)
            return X_pd
        else:
            raise ValueError(f"Unknown return format {return_format}, should be 'dataframe' or 'tsfuse'.")
