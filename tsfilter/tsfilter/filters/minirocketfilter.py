import time
from _operator import itemgetter
from math import ceil
from typing import Union, Dict

import numpy as np
import pandas as pd
from deprecated import deprecated
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sktime.datatypes._panel._convert import from_3d_numpy_to_multi_index
from sktime.transformations.panel.rocket import MiniRocket

from tsfilter.utils.constants import SEED
from tsfilter.utils import *

from tsfuse.data import Collection, dict_collection_to_numpy3d
from tsfuse.transformers import SinglePassStatistics
from tsfuse.utils import encode_onehot
from tsfilter.rank_correlation.rank_correlation import *

# FILTERING_MODE = 'minirocket'
FILTERING_MODE = 'statistics'


class MiniRocketFilter(TransformerMixin):
    """
    A class for selecting a relevant and non-redundant set of signals, using MiniRocket as a feature extractor. There is
    also support for using Catch22 as a feature extractor, but this is not recommended, as it is much slower than
    MiniRocket. No feature extractor is also supported, but this is not recommended, as the predictive performance is
    much worse than when using MiniRocket.
    Filtering is done in two steps: first, irrelevant series are filtered out based on their AUC score. Second,
    redundant series are filtered out based on their rank correlation. The rank correlation is computed using the
    Spearman rank correlation.
    """

    def __init__(self,
                 irrelevant_filter=True,
                 redundant_filter=True,
                 num_kernels: int = 100,
                 max_dilations_per_kernel: int = 32,
                 n_jobs: int = 1,
                 task: str = 'auto',
                 random_state: int = SEED,
                 filtering_threshold_auc: float = 0.5,
                 auc_percentage: float = 0.75,
                 filtering_threshold_corr: float = 0.7,
                 filtering_test_size: float = None,
                 optimized: bool = False,
                 ):
        """
        Parameters
        ----------
        irrelevant_filter: bool, default=True
            Whether to filter out irrelevant series based on their AUC score
        redundant_filter: bool, default=True
            Whether to filter out redundant series based on their rank correlation
        num_kernels: int, default=100
            The number of kernels to use in MiniRocket
        max_dilations_per_kernel: int, default=32
            The maximum number of dilations per kernel to use in MiniRocket
        n_jobs: int, default=1
            The number of jobs to use in MiniRocket
        task: str, default='auto'
            The task to perform. Can be either 'auto', 'classification' or 'regression'. If 'auto', the task is inferred
            from the data.
        random_state: int, default=SEED
            The random state used throughout the class.
        filtering_threshold_auc: float, default=0.5
            The threshold to use for filtering out irrelevant series based on their AUC score. All signals below this
            threshold are removed.
        auc_percentage: float, default=0.75
            The percentage of series to keep based on their AUC score. This parameter is only used if
            irrelevant_filter=True. If auc_percentage=0.75, the 75% series with the highest AUC score are kept.
        filtering_threshold_corr: float, default=0.7
             The threshold used for clustering rank correlations. All predictions with a rank correlation above this
             threshold are considered correlated.
        filtering_test_size: float, default=None
            The test size to use for filtering out irrelevant series based on their AUC score. The test size is the
            percentage of the data that is used for computing the AUC score. The remaining data is used for training.
            If None, the train size is derived from max(100, 0.25*nb_instances). The test size are then the remaining
            instances.
        optimized: bool, default=False
            Whether to use the optimized version for computing rank correlations. This version is faster, but slightly
            hurts the predictive performance.
        """
        self.minirocket = MiniRocket(num_kernels=num_kernels, max_dilations_per_kernel=max_dilations_per_kernel,
                                     n_jobs=n_jobs, random_state=random_state)
        self.irrelevant_filter = irrelevant_filter
        self.redundant_filter = redundant_filter
        self.task = task
        self.random_state = random_state
        self.filtering_threshold_auc = filtering_threshold_auc
        self.filtering_threshold_corr = filtering_threshold_corr
        self.removed_series_auc = set()
        self.removed_series_corr = set()
        self.acc_col = {}
        self.auc_col = {}
        self.test_size = filtering_test_size
        self.clusters = None
        self.rank_correlation = None
        self.filtered_series = None
        self.selected_col_nb = None
        self._sorted_auc: Optional[List[Union[str, int]]] = None
        self.optimized = optimized
        self.auc_percentage = auc_percentage
        self.features = None
        self.times: dict = {"Extracting features": 0, "Training model": 0, "Computing AUC": 0}
        self.scaler = None
        self.columns = None
        self.map_columns_np = None
        self.index = None

    def transform(self, X: Union[pd.DataFrame, Dict[Union[str, int], Collection]]) \
            -> Union[pd.DataFrame, Dict[Union[str, int], Collection]]:
        """
        Transform the data to the selected series.

        Parameters
        ----------
        X: Union[pd.DataFrame, Dict[Union[str, int], Collection]]
            The data to transform. Can be either a pandas DataFrame or a TSFuse Collection.

        Returns
        -------
        pd.DataFrame or Dict[Union[str, int], Collection]
            The transformed data in the same format as the input data.
        """
        if isinstance(X, pd.DataFrame):
            return X[self.filtered_series]
        elif isinstance(X, dict):
            return {k: v for k, v in X.items() if k in self.filtered_series}

    def fit(self, X: Union[pd.DataFrame, Dict[Union[str, int], Collection]], y, metadata=None, force=False,
            mode=FILTERING_MODE) -> None:
        """
        Fit the filter to the data.

        Parameters
        ----------
        X: Union[pd.DataFrame, Dict[Union[str, int], Collection]]
            The data to fit the filter to. Can be either a pandas DataFrame or a TSFuse Collection.
        y: pd.Series
            The target variable
        metadata: dict, default=None
            The metadata to update with the results of the filter. If no metadata is provided, no metadata is updated.
        force: bool
            Whether to force the filter to be retrained, even if it has already been trained.
        mode: str, default='minirocket'
            The mode to use for filtering. Can be either 'raw', 'minirocket' or 'catch22'. If 'raw', the raw data is
            used. If 'minirocket', the data is transformed using MiniRocket. If 'catch22', the data is transformed
            using Catch22.

        Returns
        -------
        None, for performance reasons, the filter is only fitted to the data, but the data is not transformed.
        """
        y = copy.deepcopy(y)
        X_np = None
        X_tsfuse = None
        if isinstance(X, pd.DataFrame):
            X_np = self.preprocessing(X)
            self.columns = X.columns
            self.map_columns_np = {col: i for i, col in enumerate(X.columns)}
            self.index = X.index
        elif isinstance(X, dict):
            X_tsfuse = self.preprocessing_dict(X)
            self.tsfuse_format = True
            self.columns = list(X.keys())
            self.index = X_tsfuse[self.columns[0]].index

        if self.filtered_series is not None and not force:
            return None
        ranks, highest_removed_auc = self.train_models(X_np, X_tsfuse, y, mode)
        if self.irrelevant_filter:
            start = time.process_time()
            ranks = self.filter_auc_percentage(data_to_filter=ranks, p=self.auc_percentage)

            if len(ranks) == 0:
                raise ValueError(f"No series passed the AUC filtering, please decrease the threshold. The highest AUC"
                                 f" was {highest_removed_auc}.")
            elif len(ranks) == 1 and self.redundant_filter:
                print("     Only one series passed the AUC filtering, no need to compute rank correlations")
                self.rank_correlation = dict()
                self.clusters = [list(ranks.keys())]
                self.filtered_series = list(ranks.keys())
                self.update_metadata(metadata)
                # X_filtered = X_np[:, self.map_columns_np[self.filtered_series[0]], :]
                # return numpy3d_to_multiindex(X_filtered, column_names=self.filtered_series, index=self.index)
                return None
            print("         Time AUC filtering: ", time.process_time() - start)

        if self.redundant_filter:
            self.redundant_filtering(ranks)
        else:
            self.filtered_series = list(ranks.keys())

        self.update_metadata(metadata)
        return None

    def preprocessing(self, X: pd.DataFrame) -> np.ndarray:
        from tsfilter import MinMaxScaler3D
        self.scaler = MinMaxScaler3D()
        X_np = self.scaler.fit_transform(X)
        if np.isnan(X_np).any():
            interpolate_nan_3d(X_np, inplace=True)
        return X_np

    def preprocessing_dict(self, X: Dict[Union[str, int], Collection]) -> Dict[Union[str, int], Collection]:
        from tsfilter.utils.scaler import MinMaxScalerCollections
        self.scaler = MinMaxScalerCollections()
        scaled_X = self.scaler.fit_transform(X, inplace=True)
        for key in X.keys():
            if np.isnan(scaled_X[key].values).any():
                ffill_nan(scaled_X[key].values, inplace=True)
        return scaled_X

    def train_models(self, X_np, X_tsfuse, y: pd.Series, mode: str = FILTERING_MODE) -> (dict, float):
        """
        Train the models for each dimension and compute the AUC score for each dimension.

        Parameters
        ----------
        X_np: np.ndarray or None
            The data to fit the filter to in numpy 3D format.
        X_tsfuse: Dict[Union[str, int], Collection] or None
            The data to fit the filter to in TSFuse format.
        y: pd.Series
            The target variable
        mode: str, default='minirocket'
            The mode to use for filtering. Can be either 'raw', 'minirocket' or 'catch22'. If 'raw', the raw data is
            used for training. If 'minirocket', the data is transformed using MiniRocket. If 'catch22', the data is
            transformed using Catch22.

        Returns
        -------
        dict
            A dictionary with the ranks for each dimension
        float
            The highest AUC score that was removed because it was below the AUC threshold
        """
        start = time.process_time()
        if X_np is None:
            X = X_tsfuse
            tsfuse_format = True
        else:
            X = X_np
            tsfuse_format = False
        ranks = {}
        train_ix, test_ix = self.train_test_split(X)
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        highest_removed_auc = 0
        if mode == 'minirocket':
            # Pad until length 9 (required for MiniRocket)
            if not tsfuse_format and X.shape[1] < 9:
                X = pad_until_length_np(X, 9)
            elif tsfuse_format and X_tsfuse[list(X.keys())[0]].shape[1] < 9:
                X = pad_until_length_tsfuse(X, 9)

        for i, col in enumerate(self.columns):
            start2 = time.process_time()
            features_train, features_test = self.extract_features(X, col, train_ix, test_ix, y_train, mode, i,
                                                                  tsfuse_format=tsfuse_format)
            self.times["Extracting features"] += time.process_time() - start2
            # print("         |   Time extracting features: ", time.process_time() - start2)

            start2 = time.process_time()
            clf = LogisticRegression(random_state=self.random_state)
            clf.fit(features_train, y_train)
            # print("         |   Time training_model: ", time.process_time() - start2)
            self.times["Training model"] += time.process_time() - start2
            if np.isnan(features_test).any():
                print("###################Nan values in the test set, removing them###################")
                nan_rows = np.isnan(features_test).any(axis=1)
                features_test = features_test[~nan_rows, :]
                y_test_no_nan = y_test.iloc[~nan_rows]
            else:
                y_test_no_nan = y_test
            predict_proba = clf.predict_proba(features_test)
            if np.unique(y_test_no_nan).shape[0] != predict_proba.shape[1]:
                raise ValueError("Not all classes are present in the test set, increase the test size to be able to "
                                 "compute the AUC")

            start2 = time.process_time()
            auc_col = roc_auc_score(encode_onehot(y_test_no_nan), predict_proba)
            self.times["Computing AUC"] += time.process_time() - start2
            # print("         |   Time computing AUC: ", time.process_time() - start2)

            if self.irrelevant_filter:
                # Test AUC series high enough
                if auc_col < self.filtering_threshold_auc:
                    self.removed_series_auc.add(col)
                    if auc_col > highest_removed_auc:
                        highest_removed_auc = auc_col
                    continue

            self.acc_col[col] = clf.score(features_test, y_test_no_nan)
            self.auc_col[col] = auc_col
            ranks[col] = probabilities2rank(predict_proba)
        print("         Total: Time AUC per series: ", time.process_time() - start)
        print("             | Time extracting features: ", self.times["Extracting features"])
        print("             | Time training model: ", self.times["Training model"])
        print("             | Time computing AUC: ", self.times["Computing AUC"])
        return ranks, highest_removed_auc

    def redundant_filtering(self, ranks: dict):
        """
        Filter out redundant series based on their rank correlation.

        Parameters
        ----------
        ranks: dict
            A dictionary with the rank for each dimension in the format {(signal1, signal2): rank}
        """
        start = time.process_time()
        if self.optimized:
            self.rank_correlation, included_series = \
                pairwise_rank_correlation_opt(ranks, self.sorted_auc, corr_threshold=self.filtering_threshold_corr)

        else:
            self.rank_correlation = pairwise_rank_correlation(ranks)
            included_series = set(ranks.keys())
        print("         Time computing rank correlations: ", time.process_time() - start)
        start = time.process_time()
        self.clusters = cluster_correlations(self.rank_correlation, included_series,
                                             threshold=self.filtering_threshold_corr)
        print("         Time clustering: ", time.process_time() - start)
        start = time.process_time()
        self.filtered_series = self.choose_from_clusters()
        print("         Time choose from cluster: ", time.process_time() - start)

    def update_metadata(self, metadata):
        """
        Update the metadata with the results of the filter.
        """
        if metadata:
            metadata[Keys.series_filtering][Keys.accuracy_score].append(self.acc_col)
            metadata[Keys.series_filtering][Keys.auc_score].append(self.auc_col)
            metadata[Keys.series_filtering][Keys.rank_correlation].append(self.rank_correlation)
            metadata[Keys.series_filtering][Keys.removed_series_auc].append(self.removed_series_auc)
            metadata[Keys.series_filtering][Keys.removed_series_corr].append(self.removed_series_corr)
            metadata[Keys.series_filtering][Keys.series_filter].append(self)

    def extract_features(self, X, col, train_ix, test_ix, y_train, mode, i, tsfuse_format=False) -> (
            np.ndarray, np.ndarray):
        """
        Extract the features for a single dimension.

        Parameters
        ----------
        X: 3D numpy array or dictionary of Collections
            The data to fit the filter to.
        col: str
            The name of the dimension to extract the features from
        train_ix: list
            The indices of the training set
        test_ix: list
            The indices of the test set
        y_train: pd.Series
            The target variable of the training set
        mode: str
            The mode to use for filtering. Can be either 'raw', 'minirocket' or 'catch22'. If 'raw', the raw data is
            used for training. If 'minirocket', the data is transformed using MiniRocket. If 'catch22', the data is
            transformed using Catch22.
        i: int
            The index of the dimension to extract the features from, needed for the raw or catch22 mode.
        tsfuse_format: bool, default=False
            Whether the data `X` is in TSFuse format or not.

        Returns
        -------
        np.ndarray
            The features of the training set
        np.ndarray
            The features of the test set
        """
        if mode == 'raw':
            if tsfuse_format:
                X = dict_collection_to_numpy3d(X)
            features_train, features_test = X[train_ix, :, i], X[test_ix, :, i]
        elif mode == 'catch22':
            if tsfuse_format:
                X = dict_collection_to_numpy3d(X)
            features = catch22_features_numpy(X[:, :, i], catch24=True)
            features_train = features[train_ix, :]
            features_test = features[test_ix, :]
        elif mode == 'minirocket':
            if tsfuse_format:
                X_selected = X[col].values
            else:
                X_selected = X[:, i, :]
            X_selected = X_selected.reshape(X_selected.shape[0], 1, X_selected.shape[1])
            X_selected = remove_trailing_nans_np(X_selected)

            # Pad until length 9 if trailing nans cause length to decrease (required for MiniRocket)
            if X_selected.shape[1] < 9:
                X_selected = pad_until_length_np(X_selected, 9)

            X_selected = from_3d_numpy_to_multi_index(X_selected, column_names=self.columns)
            X_train, X_test = X_selected.loc[train_ix], X_selected.loc[test_ix]
            features_train_pd = self.minirocket.fit_transform(X_train, y_train)
            features_train = features_train_pd.to_numpy()
            features_test_pd = self.minirocket.transform(X_test)
            features_test = features_test_pd.to_numpy()
        elif mode == 'statistics':
            if not tsfuse_format:
                X = get_tsfuse_format(X)
            stats = SinglePassStatistics().transform(X[col]).values[:, :, 0]
            features_train = stats[train_ix, :]
            features_test = stats[test_ix, :]

        else:
            raise ValueError("Mode must be either 'raw', 'minirocket', 'catch22' or 'statistics'")
        # Drop all NaN columns
        if np.isnan(features_train).any():
            nan_cols = np.isnan(features_train).all(axis=0)
            features_train = features_train[:, ~nan_cols]
            features_test = features_test[:, ~nan_cols]
        # Drop rows where NaN still exist
        if np.isnan(features_train).any():
            ffill_nan(features_train)
            ffill_nan(features_test)
        scaler = MinMaxScaler()
        features_train = scaler.fit_transform(features_train)
        features_test = scaler.transform(features_test)

        return features_train, features_test

    @deprecated
    def train_test_split_pd(self, X: pd.DataFrame) -> (list, list):
        """
        Split the data into a training and test set.

        Parameters
        ----------
        X: pd.DataFrame
            The data to split

        Returns
        -------
        list
            The indices of the training set
        list
            The indices of the test set
        """
        nb_instances = get_nb_instances_multiindex(X)
        test_size = self.compute_test_size(nb_instances)
        print("         Test size: ", test_size)
        train_ix_all, test_ix_all = train_test_split(list(range(nb_instances)),
                                                     test_size=test_size,
                                                     random_state=self.random_state)
        index_values = sorted(list({i for i, _ in X.index.values}))
        train_ix = sorted([index_values[i] for i in train_ix_all])
        test_ix = sorted([index_values[i] for i in test_ix_all])
        return train_ix, test_ix

    def train_test_split(self, X: Union[np.ndarray, Dict[Union[str, int], Collection]]) -> (list, list):
        """
        Split the data into a training and test set.

        Parameters
        ----------
        X: np.ndarray
            The data to split

        Returns
        -------
        list
            The indices of the training set
        list
            The indices of the test set
        """
        if isinstance(X, dict):
            nb_instances = X[list(X.keys())[0]].shape[0]
        else:
            nb_instances = X.shape[0]
        test_size = self.compute_test_size(nb_instances)
        print("         Test size: ", test_size)
        train_ix_all, test_ix_all = train_test_split(list(range(nb_instances)),
                                                     test_size=test_size,
                                                     random_state=self.random_state)
        return train_ix_all, test_ix_all

    def compute_test_size(self, nb_instances):
        if self.test_size:
            test_size = self.test_size
        elif nb_instances < 100:
            test_size = 0.25
        else:
            number_train = max(100, round(0.25 * nb_instances))
            train_size = number_train / nb_instances
            test_size = 1 - train_size
        return test_size

    def choose_from_clusters(self) -> list:
        """
        Choose the series to keep from the clusters. From each cluster, the series with the highest AUC score is kept.

        Returns
        -------
        list
            The series that were chosen.
        """
        chosen = []
        for cluster in self.clusters:
            cluster = list(cluster)
            all_auc = itemgetter(*cluster)(self.auc_col)
            max_ix = np.argmax(all_auc)
            chosen.append(cluster[max_ix])
            self.removed_series_corr.update(set(cluster[:max_ix] + cluster[max_ix + 1:]))
        return chosen

    @property
    def sorted_auc(self) -> list:
        """
        Return the series sorted by their AUC score.

        Returns
        -------
        list
            The series sorted by their AUC score.
        """
        if self._sorted_auc is None:
            self._sorted_auc = sorted(self.auc_col, key=lambda k: self.auc_col[k], reverse=True)
        return self._sorted_auc

    def filter_auc_percentage(self, p: float = 0.75, data_to_filter: dict = None) -> Optional[dict]:
        """
        Filter out the series with the lowest AUC score. The percentage of series to keep equals p.

        Parameters
        ----------
        p: float, default=0.75
            The percentage of series to keep. If p=0.75, the 75% series with the highest AUC score are kept.
        data_to_filter: dict, default=None
            The data to filter. If None, the data is not filtered.

        Returns
        -------
        dict
            The filtered data
        """
        assert 0 <= p <= 1

        i = len(self.auc_col.keys()) - ceil(len(self.auc_col.keys()) * p)  # how many items should be removed
        if i == 0:
            return data_to_filter
        threshold = self.auc_col[self.sorted_auc[-i]]  # the AUC threshold below which we will delete items
        self.filter_auc_threshold(threshold)
        if data_to_filter is not None:
            return {k: data_to_filter[k] for k in self.sorted_auc if k in data_to_filter.keys()}
        return None

    def filter_auc_threshold(self, threshold) -> None:
        """
        Filter out the series with an AUC score below the threshold.

        Parameters
        ----------
        threshold: float
            The threshold to use for filtering out irrelevant series based on their AUC score. All signals below this
            threshold are removed.
        """
        if self.auc_col[self.sorted_auc[-1]] > threshold:
            return
        i = len(self.sorted_auc) - 1
        for i in range(len(self.sorted_auc) - 1, -1, -1):
            if self.auc_col[self.sorted_auc[i]] > threshold:
                break
            self.removed_series_auc.add(self.sorted_auc[i])
        self._sorted_auc = self.sorted_auc[:i + 1]
        self.auc_col = {k: self.auc_col[k] for k in self.sorted_auc}
