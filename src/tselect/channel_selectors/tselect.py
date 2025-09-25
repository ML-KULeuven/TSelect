import copy
import random
import time
import warnings
from _operator import itemgetter
from math import ceil
from typing import Union, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tselect.utils.constants import SEED
from tselect.utils import *

from tsfuse.data import Collection
from tsfuse.transformers import SinglePassStatistics
from tsfuse.utils import encode_onehot
from tselect.rank_correlation.rank_correlation import *

from tselect.config import Config, get_default_config


class TSelect(TransformerMixin):
    """
    A class for selecting a relevant and non-redundant set of signals.
    Filtering is done in two steps: first, irrelevant series are filtered out based on their AUC score. Second,
    redundant series are filtered out based on their rank correlation. The rank correlation is computed using the
    Spearman rank correlation.
    """

    def __init__(self,
                 irrelevant_selector: bool = True,
                 irrelevant_hard_threshold: float = 0.5,
                 irrelevant_percentage_to_keep: float = 0.6,
                 redundant_selector: bool = True,
                 redundant_correlation_threshold: float = 0.7,
                 validation_size: float = None,
                 random_state: int = SEED,
                 config: Config = None
                 ):
        """
        Parameters
        ----------
        irrelevant_selector: bool, default=True
            Whether to only include the relevant channels and remove the irrelevant onces.
        irrelevant_hard_threshold: float, default=0.5
            The hard threshold used to remove irrelevant channels. Everything below this threshold is considered to be worse
            than random and is removed.
        irrelevant_percentage_to_keep: float, default=0.6
            The percentage of best-performing channels to keep.
        redundant_selector: bool, default=True
            Whether to only include non-redundant channels and remove the redundant ones.
        redundant_correlation_threshold: float, default=0.7
            The threshold used to cluster the rank correlations. All channels with a rank correlation above this threshold
            are considered to be correlated.
        validation_size: float, default=None
            The size of the validation set used to compute the evaluation metric. If None, the validation size is derived from
            max(100, 0.25*nb_instances). The train set then includes the remaining instances.
        random_state: int, default=SEED
            The random state used throughout the class.
        config: Config, optional
            The configuration object is mainly introduced for easier development of TSelect. It contains all
            hyperparameters for the TSelect channel selector. If a Config object is given, all other given parameters are
            ignored and the values from the Config object are used. Users of TSelect can ignore this parameter
        """
        if config is None:
            config = Config(irrelevant_filter=irrelevant_selector,
                            filtering_threshold_auc=irrelevant_hard_threshold,
                            auc_percentage=irrelevant_percentage_to_keep,
                            redundant_filter=redundant_selector,
                            filtering_threshold_corr=redundant_correlation_threshold,
                            filtering_test_size=validation_size,
                            random_state=random_state)

        self.config = config
        self.removed_series_auc = set()
        self.removed_series_corr = set()
        self.acc_col = {}
        self.auc_col = {}
        self.clusters = None
        self.rank_correlation = None
        self.selected_channels = None
        self.selected_col_nb = None
        self._sorted_auc: Optional[List[Union[str, int]]] = None
        self.features = None
        self.times: dict = {"Extracting features": 0, "Training model": 0, "Computing AUC": 0, "Predictions": 0,
                            "Removing uninformative signals": 0, "Computing ranks": 0, "Multiple models weighing": 0}
        self.scaler = None
        self.columns = None
        self.map_columns_np = None
        self.index = None
        self.models = {"Models": {}, "Scaler": {}, "DroppedNanCols": {}}

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
            return X[self.selected_channels]
        elif isinstance(X, dict):
            return {k: v for k, v in X.items() if k in self.selected_channels}

    def fit(self, X: Union[pd.DataFrame, Dict[Union[str, int], Collection]], y, metadata=None, force=False) -> None:
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
            self.columns = list(X.keys())
            self.index = X_tsfuse[self.columns[0]].index

        if self.selected_channels is not None and not force:
            return None
        ranks, highest_removed_auc = self.train_models(X_np, X_tsfuse, y)

        if self.config.irrelevant_filter and not self.config.irrelevant_better_than_random:
            start = time.process_time()
            ranks_filtered = self.filter_auc_percentage(data_to_filter=ranks, p=self.config.auc_percentage)

            if len(ranks_filtered) == 0:
                # The unfiltered ranks will be kept.
                warnings.warn(f"No series passed the AUC filtering, please decrease the threshold. The highest AUC"
                              f" was {highest_removed_auc}. For this run, all signals that passed the absolute AUC "
                              f"threshold are kept.")

            elif len(ranks_filtered) == 1 and self.config.redundant_filter:
                # print("     Only one series passed the AUC filtering, no need to compute rank correlations")
                self.rank_correlation = dict()
                self.clusters = [list(ranks_filtered.keys())]
                self.selected_channels = list(ranks_filtered.keys())
                self.update_metadata(metadata)
                return None
            else:
                ranks = ranks_filtered
            if self.config.print_times:
                print("         Time AUC filtering: ", time.process_time() - start)

        if self.config.redundant_filter:
            self.redundant_filtering(ranks)
        else:
            self.selected_channels = list(ranks.keys())

        self.update_metadata(metadata)
        return None

    def preprocessing(self, X: pd.DataFrame) -> np.ndarray:
        """
        Preprocess the data before fitting the filter.

        Parameters
        ----------
        X: pd.DataFrame
            The data to preprocess

        Returns
        -------
        np.ndarray
            The preprocessed data

        """
        from tselect import MinMaxScaler3D
        self.scaler = MinMaxScaler3D()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_np = self.scaler.fit_transform(X)
        if np.isnan(X_np).any():
            interpolate_nan_3d(X_np, inplace=True)
        return X_np

    def preprocessing_dict(self, X: Dict[Union[str, int], Collection]) -> Dict[Union[str, int], Collection]:
        """
        Preprocess the data before fitting the filter if the data is in TSFuse format.

        Parameters
        ----------
        X: Dict[Union[str, int], Collection]
            The data to preprocess

        Returns
        -------
        Dict[Union[str, int], Collection]
            The preprocessed data in TSFuse format

        """
        from tselect.utils.scaler import MinMaxScalerCollections
        self.scaler = MinMaxScalerCollections()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled_X = self.scaler.fit_transform(X, inplace=True)
        for key in X.keys():
            if np.isnan(scaled_X[key].values).any():
                ffill_nan(scaled_X[key].values, inplace=True)
        return scaled_X

    def train_models(self, X_np, X_tsfuse, y: pd.Series) -> (dict, float):
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
        predictions_removed_signals = {}
        all_features = {}

        for i, col in enumerate(self.columns):
            start2 = time.process_time()
            features_train, features_test = self.extract_features(X, col, train_ix, test_ix, i, y=y,
                                                                  tsfuse_format=tsfuse_format)
            self.times["Extracting features"] += time.process_time() - start2

            start2 = time.process_time()
            clf = LogisticRegression(random_state=self.config.random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(features_train, y_train)
            self.models["Models"][col] = clf
            self.times["Training model"] += time.process_time() - start2
            start2 = time.process_time()
            if np.isnan(features_test).any():
                replace_nans_by_col_mean(features_test)

            predict_proba = clf.predict_proba(features_test)
            if np.unique(y_test).shape[0] != predict_proba.shape[1]:
                raise ValueError("Not all classes are present in the test set, increase the test size to be able to "
                                 "compute the AUC")
            self.times["Predictions"] += time.process_time() - start2

            start2 = time.process_time()
            auc_col = roc_auc_score(encode_onehot(y_test), predict_proba)
            self.times["Computing AUC"] += time.process_time() - start2

            all_features[col] = {'train': features_train, 'test': features_test}
            start2 = time.process_time()
            if self.config.irrelevant_filter:
                if self.config.irrelevant_better_than_random:
                    to_keep = self.irrelevant_selector_random(features_train, features_test, y_train, y_test, auc_col)
                else:
                    # Test AUC series high enough
                    to_keep = not (auc_col < self.config.filtering_threshold_auc)
                if not to_keep:
                    self.removed_series_auc.add((col, auc_col))
                    predictions_removed_signals[col] = predict_proba
                    if auc_col > highest_removed_auc:
                        highest_removed_auc = auc_col
                    continue
            self.times["Removing uninformative signals"] += time.process_time() - start2
            self.auc_col[col] = auc_col

            start2 = time.process_time()
            ranks[col] = probabilities2rank(predict_proba)
            self.times["Computing ranks"] += time.process_time() - start2

        # If no series passed the AUC threshold filtering, keep all series for now
        if len(ranks) == 0:
            warnings.warn(f"No series passed the AUC filtering, please decrease the threshold. The highest AUC"
                          f" was {highest_removed_auc}. For this run, no series below the absolute AUC threshold "
                          f"(default=0.5) were removed.")
            for col, auc in self.removed_series_auc:
                self.auc_col[col] = auc
                start2 = time.process_time()
                ranks[col] = probabilities2rank(predictions_removed_signals[col])
                self.times["Computing ranks"] += time.process_time() - start2

        # Compute additional models if multiple model weighing is on
        if self.config.multiple_models_weighing:
            start2 = time.process_time()
            self.irrelevant_selector_multiple_models([0.34, 0.5], [3, 2], all_features,
                                                     y_train, y_test)
            self.times["Multiple models weighing"] += time.process_time() - start2

        if self.config.print_times:
            print("         Total: Time AUC per series: ", time.process_time() - start)
            print("             | Time extracting features: ", self.times["Extracting features"])
            print("             | Time training model: ", self.times["Training model"])
            print("             | Time predictions: ", self.times["Predictions"])
            print("             | Time computing AUC: ", self.times["Computing AUC"])
            print("             | Time removing uninformative signals: ", self.times["Removing uninformative signals"])
            print("             | Time computing ranks: ", self.times["Computing ranks"])
            print("             | Time computing multiple models: ", self.times["Multiple models weighing"])
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
        self.rank_correlation, included_series = \
            pairwise_rank_correlation_opt(ranks)

        if self.config.print_times:
            print("         Time computing rank correlations: ", time.process_time() - start)
        start = time.process_time()
        if self.config.hierarchical_clustering:
            self.clusters = cluster_correlations_agglo_hierarchical(self.rank_correlation, included_series,
                                                                    correlation_threshold=self.config.filtering_threshold_corr)
        elif self.config.spectral_clustering:
            self.clusters = cluster_correlations_spectral_clustering(self.rank_correlation, included_series,
                                                                     random_state=self.config.random_state)
        else:
            self.clusters = cluster_correlations(self.rank_correlation, included_series,
                                             threshold=self.config.filtering_threshold_corr)
        if self.config.print_times:
            print("         Time clustering: ", time.process_time() - start)
        start = time.process_time()
        self.selected_channels = self.choose_from_clusters()
        if self.config.print_times:
            print("         Time choose from cluster: ", time.process_time() - start)

    def update_metadata(self, metadata):
        """
        Update the metadata with the results of the filter.
        """
        if metadata:
            metadata[Keys.series_filtering][Keys.acc_score].append(self.acc_col)
            metadata[Keys.series_filtering][Keys.auc_score].append(self.auc_col)
            metadata[Keys.series_filtering][Keys.rank_correlation].append(self.rank_correlation)
            metadata[Keys.series_filtering][Keys.removed_series_auc].append(self.removed_series_auc)
            metadata[Keys.series_filtering][Keys.removed_series_corr].append(self.removed_series_corr)
            metadata[Keys.series_filtering][Keys.series_filter].append(self)

    def extract_features(self, X: Union[dict, np.ndarray], col, train_ix, test_ix, i, y=None, tsfuse_format=False) -> (
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
        if self.config.feature_extractor is None:
            if not tsfuse_format:
                X_i = Collection(X[:, i, :].reshape(X.shape[0], 1, X.shape[2]), from_numpy3d=True)
            else:
                X_i = X[col]
            stats = SinglePassStatistics().transform(X_i).values[:, :, 0]
            features_train = stats[train_ix, :]
            features_test = stats[test_ix, :]
        else:
            if not tsfuse_format:
                X_i = X[:, i, :]
            else:
                X_i = X[col].values
            features_train, features_test = self.config.feature_extractor(X_i, train_ix, test_ix, y)
        # Drop all NaN columns
        if np.isnan(features_train).any():
            nan_cols = np.isnan(features_train).all(axis=0)
            if col not in self.models["DroppedNanCols"].keys():
                self.models["DroppedNanCols"][col] = nan_cols
            features_train = features_train[:, ~nan_cols]
            features_test = features_test[:, ~nan_cols]
        # Drop rows where NaN still exist
        if np.isnan(features_train).any():
            replace_nans_by_col_mean(features_train)
            replace_nans_by_col_mean(features_test)
        scaler = MinMaxScaler()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features_train = scaler.fit_transform(features_train)
            features_test = scaler.transform(features_test)
        if col not in self.models["Scaler"].keys():
            self.models["Scaler"][col] = scaler

        return features_train, features_test

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

        if self.config.print_times:
            print("         Test size: ", test_size)
        train_ix_all, test_ix_all = train_test_split(list(range(nb_instances)),
                                                     test_size=test_size,
                                                     random_state=self.config.random_state)
        return train_ix_all, test_ix_all

    def compute_test_size(self, nb_instances):
        """
        Compute the test size based on the number of instances.

        Parameters
        ----------
        nb_instances: int
            The number of instances in the data

        Returns
        -------
        float
            The test size
        """
        if self.config.test_size:
            test_size = self.config.test_size
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
        print("     AUC percentage: ", p)

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
            self.removed_series_auc.add((self.sorted_auc[i], self.auc_col[self.sorted_auc[i]]))
        self._sorted_auc = self.sorted_auc[:i + 1]
        self.auc_col = {k: self.auc_col[k] for k in self.sorted_auc}

    def irrelevant_selector_multiple_models(self, group_sizes: List[float], group_amounts: List[int],
                                            features_channel: Dict[str, Dict[str, np.ndarray]], y_train: pd.Series,
                                            y_test: pd.Series):
        # Divide channels in (1) 3 groups (2) 2 groups
        groups = self.make_groups_multiple_models(group_amounts, group_sizes)

        # Train a model on each of the groups
        ind_scores = {ch: [] for ch in self.auc_col.keys()}
        nb_features_per_channel = features_channel[list(self.auc_col.keys())[0]]["train"].shape[1]
        for big_group in groups:
            for group in big_group:
                model = LogisticRegression(random_state=self.config.random_state, penalty='l1', solver='saga')
                features_train = np.concatenate([features_channel[channel]["train"] for channel in group], axis=1)
                model.fit(features_train, y_train)
                self.models["Models"][frozenset(group)] = model

                features_test = np.concatenate([features_channel[channel]["test"] for channel in group], axis=1)
                predictions = model.predict_proba(features_test)
                auc = roc_auc_score(encode_onehot(y_test), predictions)
                importances = np.max(np.abs(model.coef_), axis=0)
                importances = importances / (np.max(importances) - np.min(importances))
                for channel_index, channel in enumerate(group):
                    channel_importance = np.max(importances[channel_index * nb_features_per_channel:
                                                            (channel_index + 1) * nb_features_per_channel])
                    ind_scores[channel].append(auc * channel_importance)

        # Weigh the individual score with the group score
        for channel in self.auc_col.keys():
            if len(ind_scores[channel]) == 0:
                continue
            self.auc_col[channel] = (self.auc_col[channel] + np.mean(ind_scores[channel])) / 2

    def make_groups_multiple_models(self, group_amounts, group_sizes):
        assert len(group_sizes) == len(group_amounts)
        all_channels = list(self.auc_col.keys())
        seed = SEED
        groups: List[List[list]] = []
        for i, size in enumerate(group_sizes):
            assert 0 < size <= 1, ("The group sizes should be expressed as a percentage of the number of "
                                   "channels and must lie in the interval (0,1]")
            amount = group_amounts[i]
            random.seed(seed)
            seed += 1
            groups.append([])
            random.shuffle(all_channels)
            nb_channels_per_group = ceil(size * len(all_channels))
            nb_repetitions = ceil(nb_channels_per_group * amount / len(all_channels))
            channels_extended = []
            for j in range(nb_repetitions):
                temp_channels = random.sample(all_channels, len(all_channels))
                channels_extended.extend(temp_channels)

            for j in range(amount):
                g = list(set(channels_extended[j * nb_channels_per_group:(j + 1) * nb_channels_per_group]))
                # If there is a group with only one channel, we don't need to train a model
                if len(g) > 1:
                    groups[i].append(g)

            random.shuffle(all_channels)

        return groups

    def irrelevant_selector_random(self, features_train, features_test, y_train, y_test, reference_auc):
        """
        For a single channel, train a model on a random permutation of the target variable and check if the AUC score is
        below the reference AUC score. If it is, the channel is considered irrelevant and False is returned.
        Parameters
        ----------
        features_train: np.ndarray
            The features of the training set
        features_test: np.ndarray
            The features of the test set
        y_train: pd.Series
            The target variable of the training set
        y_test: pd.Series
            The target variable of the test set
        reference_auc:
            The reference AUC score. This should be the ROCAUC of the unshuffled target variable.

        Returns
        -------
        bool
            Whether the channel should be kept or not. If False, the channel is considered irrelevant.

        """
        # Shuffle features train
        np.random.seed(self.config.random_state)
        random_indices = np.random.permutation(y_train.shape[0])
        y_train_random = y_train.iloc[random_indices]

        # Train model
        model = LogisticRegression(random_state=self.config.random_state)
        model.fit(features_train, y_train_random)
        predictions = model.predict_proba(features_test)
        auc = roc_auc_score(encode_onehot(y_test), predictions)
        # print(auc)
        if auc > reference_auc:
            print("Random was better with auc: ", auc, " compared to reference auc: ", reference_auc)
        return auc < reference_auc
