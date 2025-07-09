import gc
import random
import time
import warnings
from _operator import itemgetter
from collections import defaultdict
from math import ceil
from typing import Union, Dict

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from TSelect.tselect.tselect.utils.constants import SEED
from TSelect.tselect.tselect.utils import *
from TSelect.tselect.tselect.utils.metrics import auroc_score

from TSelect.tsfuse.tsfuse.data import Collection
from TSelect.tsfuse.tsfuse.transformers import SinglePassStatistics
from TSelect.tselect.tselect.rank_correlation.rank_correlation import *


class TSelect(TransformerMixin):
    """
    A class for selecting a relevant and non-redundant set of signals.
    Filtering is done in two steps: first, irrelevant series are removed based on individually evaluating each channel. Second,
    redundant series are filtered out based on their rank correlation. The rank correlation is computed using the
    Spearman rank correlation.
    """

    def __init__(self,
                 irrelevant_filter=True,
                 redundant_filter=True,
                 evaluation_metric=auroc_score,
                 higher_is_better=True,
                 random_state: int = SEED,
                 irrelevant_selector_threshold: float = 0.5,
                 irrelevant_selector_percentage: float = 0.75,
                 filtering_threshold_corr: float = 0.7,
                 multiple_model_weighing: bool = False,
                 irrelevant_better_than_random: bool = False,
                 filtering_test_size: float = None,
                 print_times: bool = False
                 ):
        """
        Parameters
        ----------
        irrelevant_filter: bool, default=True
            Whether to remove irrelevant channels based on their individual evaluation
        redundant_filter: bool, default=True
            Whether to filter out redundant series based on their rank correlation
        evaluation_metric: callable, default=roc_score
            The evaluation metric to estimate the relevance of each channel. The evaluation metric should be a callable
            that takes the true labels and the predicted probabilities as input and returns a float.
            The default evaluation metric is the ROC AUC score.
        higher_is_better: bool, default=True
            Whether a higher value of the evaluation metric indicates a better performance.
        random_state: int, default=SEED
            The random state used throughout the class.
        irrelevant_selector_threshold: float, default=0.5
            The threshold to use for removing irrelevant channels based on their individual evaluation by computing the
            given metric. All signals worse this threshold are removed.
        irrelevant_selector_percentage: float, default=0.6
            The percentage of series to keep based  on their individual evaluation by computing the
            given metric. This parameter is only used if irrelevant_filter=True. If irrelevant_selector_percentage=0.6,
            the 60% channels with the best score are kept.
        filtering_threshold_corr: float, default=0.7
             The threshold used for clustering rank correlations. All predictions with a rank correlation above this
             threshold are considered correlated.
        irrelevant_better_than_random: bool, default=False
            Whether to filter out irrelevant series by comparing them with a model trained on a randomly shuffled
            target. If the channel performs Better Than Random (BTR), it is kept.
        filtering_test_size: float, default=None
            The test size to use for removing irrelevant series based on their individual evaluation by computing the
            given metric. The test size is the percentage of the data that is used for computing the score.
            The remaining data is used for training. If None, the train size is derived from
            max(100, 0.25*nb_instances). The test size are then the remaining instances.
        """
        self.irrelevant_filter = irrelevant_filter
        self.redundant_filter = redundant_filter
        self.evaluation_metric = evaluation_metric
        self.higher_is_better = higher_is_better
        self.random_state = random_state
        self.irrelevant_threshold = irrelevant_selector_threshold
        self.filtering_threshold_corr = filtering_threshold_corr
        self.removed_series_too_low_metric = set()
        self.removed_series_corr = set()
        self.acc_col = {}
        self.evaluation_metric_per_channel = {}
        self.test_size = filtering_test_size
        self.clusters = None
        self.rank_correlation = None
        self.selected_channels: List[str | int] = None
        self.selected_col_nb = None
        self._sorted_scores: Optional[List[Union[str, int]]] = None
        self.irrelevant_selector_keep_best_x = irrelevant_selector_percentage
        self.multiple_models_weighing = multiple_model_weighing
        self.irrelevant_better_than_random = irrelevant_better_than_random
        self.features = None
        self.times: dict = {"Extracting features": 0, "Training model": 0, "Computing evaluation metric": 0, "Predictions": 0,
                            "Removing irrelevant channels": 0, "Computing ranks": 0, "Multiple models weighing": 0}
        self.scaler = None
        self.columns = None
        self.map_columns_np = None
        self.index = None
        self.models = {"Models": {}, "Scaler": {}, "DroppedNanCols": {}}
        self.print_times = print_times

    def transform(self, X: Union[pd.DataFrame, Dict[Union[str, int], Collection], np.ndarray]):
        """
        Transform the data to the selected series.

        Parameters
        ----------
        X: Union[pd.DataFrame, Dict[Union[str, int], Collection]]
            The data to transform. Can be either a pandas DataFrame or a TSFuse Collection.

        Returns
        -------
        pd.DataFrame, Dict[Union[str, int], Collection] or np.ndarray
            The transformed data in the same format as the input data.
        """
        if isinstance(X, pd.DataFrame):
            return X[self.selected_channels]
        elif isinstance(X, np.ndarray):
            return X[:, :, [self.map_columns_np[col] for col in self.selected_channels]]
        elif isinstance(X, dict):
            return {k: v for k, v in X.items() if k in self.selected_channels}

    def fit(self, X, y, X_val=None, y_val=None, metadata=None, force=False) -> None:
        """
        Fit the filter to the data.

        Parameters
        ----------
        X: pd.DataFrame, Dict[Union[str, int], Collection], np.ndarray
            The data to fit the filter to. Can be either a pandas DataFrame or a TSFuse Collection.
        y: pd.Series
            The target variable
        X_val: Union[pd.DataFrame, Dict[Union[str, int], Collection], np.ndarray] or None
            The validation data to fit the filter to. If provided, X_val should have the same type as X. If X_val is
            None, the data is split into a training and test set.
        y_val: pd.Series, np.ndarray, or None
            The validation target variable. This is only used if X_val is not None.
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
            if X_val is not None:
                assert isinstance(X_val, pd.DataFrame)
                X_val = self.preprocessing(X_val)

        elif isinstance(X, np.ndarray):
            X_np = self.preprocessing(X.transpose(0, 2, 1))
            self.columns = list(range(X.shape[2]))
            self.map_columns_np = {col: i for i, col in enumerate(self.columns)}
            self.index = range(X.shape[0])
            if X_val is not None:
                assert isinstance(X_val, np.ndarray)
                X_val = self.preprocessing(X_val)
        elif isinstance(X, dict):
            X_tsfuse = self.preprocessing_dict(X)
            self.columns = list(X.keys())
            self.index = X_tsfuse[self.columns[0]].index
            if X_val is not None:
                assert isinstance(X_val, dict)
                X_val = self.preprocessing_dict(X_val)

        if self.selected_channels is not None and not force:
            return None
        ranks, best_removed_score = self.train_models(X_np, X_tsfuse, y, X_val=X_val, y_val=y_val)

        if self.irrelevant_filter and not self.irrelevant_better_than_random:
            start = time.process_time()
            ranks_filtered = self.irrelevant_selector_percentage(data_to_filter=ranks, p=self.irrelevant_selector_keep_best_x)

            if len(ranks_filtered) == 0:
                # The unfiltered ranks will be kept.
                warnings.warn(f"No series passed the threshold in the irrelevant selector, please make the threshold "
                              f"less strict. The best removed score was {best_removed_score}. For this run, all signals "
                                f"that passed the absolute threshold are kept.")

            elif len(ranks_filtered) == 1 and self.redundant_filter:
                self.rank_correlation = dict()
                self.clusters = [list(ranks_filtered.keys())]
                self.selected_channels = list(ranks_filtered.keys())
                self.update_metadata(metadata)
                return None
            else:
                ranks = ranks_filtered
            if self.print_times:
                print("         Time irrelevant selector: ", time.process_time() - start)

        if self.redundant_filter:
            self.redundant_filtering(ranks)
        else:
            self.selected_channels = list(ranks.keys())

        self.update_metadata(metadata)
        return None


    def fit_generator(self, gen_train, gen_val, channel_names: List[str]=None, metadata=None, force=False) -> None:
        """
        Fit the channel selector to the data from a generator.

        Parameters
        ----------
        gen_train: generator
            A generator that yields batches of training data in the format (X, y), where X is a pandas DataFrame,
            a numpy array or a tf.Tensor and y is a pandas Series or a numpy array.
        gen_val: generator or None
            A generator that yields batches of validation data. If None, the training data is split into a training and
            validation set.
        channel_names: List[str], default=None
            The names of the channels. If None, the channels are numbered from 0 to nb_channels - 1.
        metadata: dict, default=None
            The metadata to update with the results of the filter. If no metadata is provided, no metadata is updated.
        force: bool
            Whether to force the filter to be retrained, even if it has already been trained.

        Returns
        -------
        None, for performance reasons, the filter is only fitted to the data, but the data is not transformed.
        """
        self.columns = channel_names
        if not force and self.selected_channels is not None:
            return None

        ranks, best_removed_score = self.train_models_generator(gen_train, gen_val)

        if self.irrelevant_filter and not self.irrelevant_better_than_random:
            start = time.process_time()
            ranks_filtered = self.irrelevant_selector_percentage(data_to_filter=ranks, p=self.irrelevant_selector_keep_best_x)

            if len(ranks_filtered) == 0:
                # The unfiltered ranks will be kept.
                warnings.warn(f"No series passed the threshold in the irrelevant selector, please make the threshold "
                              f"less strict. The best removed score was {best_removed_score}. For this run, all signals "
                                f"that passed the absolute threshold are kept.")

            elif len(ranks_filtered) == 1 and self.redundant_filter:
                self.rank_correlation = dict()
                self.clusters = [list(ranks_filtered.keys())]
                self.selected_channels = list(ranks_filtered.keys())
                self.update_metadata(metadata)
                return None
            else:
                ranks = ranks_filtered
            if self.print_times:
                print("         Time irrelevant selector: ", time.process_time() - start)

        if self.redundant_filter:
            self.redundant_filtering(ranks)
        else:
            self.selected_channels = list(ranks.keys())

        self.update_metadata(metadata)
        return None


    def preprocessing(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
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
        from TSelect.tselect.tselect import MinMaxScaler3D
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
        from TSelect.tselect.tselect.utils.scaler import MinMaxScalerCollections
        self.scaler = MinMaxScalerCollections()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled_X = self.scaler.fit_transform(X, inplace=True)
        for key in X.keys():
            if np.isnan(scaled_X[key].values).any():
                ffill_nan(scaled_X[key].values, inplace=True)
        return scaled_X

    def train_models(self, X_np, X_tsfuse, y: Union[pd.Series, np.ndarray], X_val=None, y_val=None) -> (dict, float):
        """
        Train the models for each dimension and compute the given metric score for each dimension.

        Parameters
        ----------
        X_np: np.ndarray or None
            The data to fit the filter to in numpy 3D format.
        X_tsfuse: Dict[Union[str, int], Collection] or None
            The data to fit the filter to in TSFuse format.
        y: pd.Series
            The target variable
        X_val: np.ndarray or None
            The validation data to fit the filter to in the same format as X_np is it is not None or in the same format
            as X_tsfuse if it is not None. If X_val is None, the data is split into a training and test set.
        y_val: pd.Series, np.ndarray or None
            The validation target variable. This is only used if X_val is not None.

        Returns
        -------
        dict
            A dictionary with the ranks for each dimension
        float
            The best score that was removed because it was below the given threshold
        """
        start = time.process_time()
        if X_np is None:
            X = X_tsfuse
            tsfuse_format = True
        else:
            X = X_np
            tsfuse_format = False
        ranks = {}
        train_ix = None
        test_ix = None
        if X_val is None:
            train_ix, test_ix = self.train_test_split(X)
            if isinstance(y, pd.Series):
                y = y.values
            y_train, y_test = y[train_ix], y[test_ix]
        else:
            y_train = y.values if isinstance(y, pd.Series) else y
            y_test = y_val.values if isinstance(y_val, pd.Series) else y_val
        best_removed_score = 0
        predictions_removed_signals = {}
        all_features = {}

        for i, col in enumerate(self.columns):
            start2 = time.process_time()
            features_train, features_test = self.extract_features(X, col, i, train_ix=train_ix, test_ix=test_ix,
                                                                    X_val=X_val,
                                                                  tsfuse_format=tsfuse_format)
            self.times["Extracting features"] += time.process_time() - start2

            start2 = time.process_time()
            clf = LogisticRegression(random_state=self.random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(features_train, y_train)
            # self.models["Models"][col] = clf
            self.times["Training model"] += time.process_time() - start2
            start2 = time.process_time()
            if np.isnan(features_test).any():
                replace_nans_by_col_mean(features_test)

            predict_proba = clf.predict_proba(features_test)
            if np.unique(y_test).shape[0] != predict_proba.shape[1]:
                raise ValueError("Not all classes are present in the test set, increase the test size to be able to "
                                 "evaluate individual channels")
            self.times["Predictions"] += time.process_time() - start2

            start2 = time.process_time()
            score = self.evaluation_metric(y_test, predict_proba)
            self.times["Computing evaluation metric"] += time.process_time() - start2

            all_features[col] = {'train': features_train, 'test': features_test}
            start2 = time.process_time()
            if self.irrelevant_filter:
                if self.irrelevant_better_than_random:
                    to_keep = self.irrelevant_selector_random(features_train, features_test, y_train, y_test, score)
                else:
                    # Test evaluation score high enough
                    to_keep = not (score < self.irrelevant_threshold) if self.higher_is_better \
                        else not (score > self.irrelevant_threshold)
                if not to_keep:
                    self.removed_series_too_low_metric.add((col, score))
                    predictions_removed_signals[col] = predict_proba
                    if score > best_removed_score if self.higher_is_better else score < best_removed_score:
                        best_removed_score = score
                    continue
            self.times["Removing irrelevant channels"] += time.process_time() - start2
            self.evaluation_metric_per_channel[col] = score

            start2 = time.process_time()
            ranks[col] = probabilities2rank(predict_proba)
            self.times["Computing ranks"] += time.process_time() - start2

        # If no series passed the threshold filtering, keep all series for now
        if len(ranks) == 0:
            warnings.warn(f"No series passed the absolute threshold, please make the threshold less strict. The best score"
                          f" was {best_removed_score}. For this run, no series worse than the absolute threshold "
                          f"(default=0.5) were removed.")
            for col, score in self.removed_series_too_low_metric:
                self.evaluation_metric_per_channel[col] = score
                start2 = time.process_time()
                ranks[col] = probabilities2rank(predictions_removed_signals[col])
                self.times["Computing ranks"] += time.process_time() - start2

        # Compute additional models if multiple model weighing is on
        if self.multiple_models_weighing:
            start2 = time.process_time()
            self.irrelevant_selector_multiple_models([0.34, 0.5], [3, 2], all_features,
                                                     y_train, y_test)
            self.times["Multiple models weighing"] += time.process_time() - start2

        if self.print_times:
            print("         Total: Time evaluating each channel: ", time.process_time() - start)
            print("             | Time extracting features: ", self.times["Extracting features"])
            print("             | Time training model: ", self.times["Training model"])
            print("             | Time predictions: ", self.times["Predictions"])
            print("             | Time computing evaluation metric: ", self.times["Computing evaluation metric"])
            print("             | Time removing uninformative signals: ", self.times["Removing irrelevant channels"])
            print("             | Time computing ranks: ", self.times["Computing ranks"])
            print("             | Time computing multiple models: ", self.times["Multiple models weighing"])
        return ranks, best_removed_score


    def train_models_generator(self, gen_train, gen_val):
        start = time.process_time()
        x_train, y_train = self.extract_features_generator(gen_train, validation=False)
        self.times["Extracting features"] += time.process_time() - start
        start = time.process_time()

        for col in self.columns:
            clf = LogisticRegression(random_state=self.random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(x_train[col], y_train)
            self.models["Models"][col] = clf
            self.times["Training model"] += time.process_time() - start

        del x_train
        del y_train
        gc.collect()

        start = time.process_time()
        x_val, y_val = self.extract_features_generator(gen_val, validation=True)
        self.times["Extracting features"] += time.process_time() - start

        ranks = {}
        best_removed_score = 0
        predictions_removed_signals = {}

        for col in self.columns:
            predict_proba = self.models["Models"][col].predict_proba(x_val[col])
            if np.unique(y_val).shape[0] != predict_proba.shape[1]:
                raise ValueError("Not all classes are present in the test set, increase the test size to be able to "
                                 "evaluate individual channels")
            self.times["Predictions"] += time.process_time() - start

            start = time.process_time()
            score = self.evaluation_metric(y_val, predict_proba)
            self.times["Computing evaluation metric"] += time.process_time() - start
            start = time.process_time()
            if self.irrelevant_filter:
                # Test evaluation score high enough
                to_keep = not (score < self.irrelevant_threshold) if self.higher_is_better \
                    else not (score > self.irrelevant_threshold)
                if not to_keep:
                    self.removed_series_too_low_metric.add((col, score))
                    predictions_removed_signals[col] = predict_proba
                    if score > best_removed_score if self.higher_is_better else score < best_removed_score:
                        best_removed_score = score
                    continue
            self.times["Removing irrelevant channels"] += time.process_time() - start
            self.evaluation_metric_per_channel[col] = score

            start = time.process_time()
            ranks[col] = probabilities2rank(predict_proba)
            self.times["Computing ranks"] += time.process_time() - start

        del x_val
        del y_val
        self.models["Models"] = {}
        gc.collect()

        # If no series passed the threshold filtering, keep all series for now
        if len(ranks) == 0:
            warnings.warn(
                f"No series passed the absolute threshold, please make the threshold less strict. The best score"
                f" was {best_removed_score}. For this run, no series worse than the absolute threshold "
                f"(default=0.5) were removed.")
            for col, score in self.removed_series_too_low_metric:
                self.evaluation_metric_per_channel[col] = score
                start2 = time.process_time()
                ranks[col] = probabilities2rank(predictions_removed_signals[col])
                self.times["Computing ranks"] += time.process_time() - start2

        if self.print_times:
            print("         Total: Time evaluating each channel: ", time.process_time() - start)
            print("             | Time extracting features: ", self.times["Extracting features"])
            print("             | Time training model: ", self.times["Training model"])
            print("             | Time predictions: ", self.times["Predictions"])
            print("             | Time computing evaluation metric: ", self.times["Computing evaluation metric"])
            print("             | Time removing uninformative signals: ",
                  self.times["Removing irrelevant channels"])
            print("             | Time computing ranks: ", self.times["Computing ranks"])
            print("             | Time computing multiple models: ", self.times["Multiple models weighing"])
        return ranks, best_removed_score

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

        if self.print_times:
            print("         Time computing rank correlations: ", time.process_time() - start)
        start = time.process_time()
        self.clusters = cluster_correlations(self.rank_correlation, included_series,
                                             threshold=self.filtering_threshold_corr)
        if self.print_times:
            print("         Time clustering: ", time.process_time() - start)
        start = time.process_time()
        self.selected_channels = self.choose_from_clusters()
        if self.print_times:
            print("         Time choose from cluster: ", time.process_time() - start)

    def update_metadata(self, metadata):
        """
        Update the metadata with the results of the filter.
        """
        if metadata:
            metadata[Keys.series_filtering][Keys.acc_score].append(self.acc_col)
            metadata[Keys.series_filtering][Keys.score_per_channel].append(self.evaluation_metric_per_channel)
            metadata[Keys.series_filtering][Keys.rank_correlation].append(self.rank_correlation)
            metadata[Keys.series_filtering][Keys.removed_series_irrelevant].append(self.removed_series_too_low_metric)
            metadata[Keys.series_filtering][Keys.removed_series_corr].append(self.removed_series_corr)
            metadata[Keys.series_filtering][Keys.series_filter].append(self)

    def extract_features(self, X: Union[dict, np.ndarray], col, i, train_ix=None, test_ix=None, X_val=None,
                         tsfuse_format=False) -> (
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
        X_val: 3D numpy array or dictionary of Collections
            The validation set. If train_ix and test_ix are not provided, X_val should be provided.
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
        if train_ix is not None and test_ix is not None:
            if not tsfuse_format:
                X_i = Collection(X[:, i, :].reshape(X.shape[0], 1, X.shape[2]), from_numpy3d=True)
            else:
                X_i = X[col]
            stats = SinglePassStatistics().transform(X_i).values[:, :, 0]
            features_train = stats[train_ix, :]
            features_test = stats[test_ix, :]
        elif X_val is not None:
            if not tsfuse_format:
                X_i = Collection(X[:, i, :].reshape(X.shape[0], 1, X.shape[2]), from_numpy3d=True)
                X_val = Collection(X_val[:, i, :].reshape(X_val.shape[0], 1, X_val.shape[2]), from_numpy3d=True)
            else:
                X_i = X[col]
                X_val = X_val[col]
            features_train = SinglePassStatistics().transform(X_i).values[:, :, 0]
            features_test = SinglePassStatistics().transform(X_val).values[:, :, 0]
        else:
            raise ValueError("Either X_val or train_ix and test_ix should be provided")

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

    def extract_features_generator(self, gen, validation=False) -> (Dict[str | int, np.ndarray], np.ndarray):
        """
        Extract features from a generator that yields batches of data.
        The generator should yield batches of data in the format (x_batch, y_batch), where x_batch is a pandas DataFrame,
        a numpy array or a tf.Tensor, and y_batch is the target variable.

        Parameters
        ----------
        gen: generator
            A generator that yields batches of data in the format (x_batch, y_batch). The x_batch can be a pandas DataFrame,
            a numpy array or a tf.Tensor. The y_batch is the target variable.
        validation: bool, default=False
            Whether the data is used for validation or not. If True, the features are scaled and NaN columns are dropped
        Returns
        -------
        Dict[str | int, np.ndarray]
            A dictionary with the features for each channel. The keys are the channel names and the values are the features.
        np.ndarray
            The target variable as a numpy array. If no data was yielded by the generator, an empty numpy array is returned.
        """
        import tensorflow as tf
        all_features = defaultdict(list)
        y = []
        for x_batch, y_batch in gen:
            gc.collect()
            if isinstance(x_batch, pd.DataFrame):
                self.columns = x_batch.columns
                self.map_columns_np = {col: i for i, col in enumerate(x_batch.columns)}
                self.index = x_batch.index
                x_batch = self.preprocessing(x_batch)
                y_batch = y_batch.values if isinstance(y_batch, pd.Series) else y_batch
            elif isinstance(x_batch, np.ndarray) or isinstance(x_batch, tf.Tensor):
                if isinstance(x_batch, tf.Tensor):
                    x_batch = x_batch.numpy().squeeze(-1) if x_batch.ndim > 3 else x_batch.numpy()
                    y_batch = y_batch.numpy() if isinstance(y_batch, tf.Tensor) else y_batch
                x_batch = self.preprocessing(x_batch.transpose(0, 2, 1))
                self.columns = list(range(x_batch.shape[1])) if self.columns is None else self.columns
                self.map_columns_np = {col: i for i, col in enumerate(self.columns)}
                self.index = range(x_batch.shape[0])
            else:
                raise ValueError("The input data should be either a pandas DataFrame, a numpy array or a tf.Tensor.")

            y.append(y_batch[:, 1] if y_batch.ndim > 1 else y_batch)
            for i, col in enumerate(self.columns):
                x_i = Collection(x_batch[:, i, :].reshape(x_batch.shape[0], 1, x_batch.shape[2]), from_numpy3d=True)
                stats: np.ndarray = SinglePassStatistics().transform(x_i).values[:, :, 0].astype(np.float32)
                all_features[col].append(stats)

        # Concatenate all features for each channel
        for col in all_features.keys():
            # noinspection PyTypeChecker
            all_features[col] = np.concatenate(all_features[col], axis=0)

            if validation:
                nan_cols = self.models["DroppedNanCols"].get(col, None)
                all_features[col] = all_features[col][:, ~nan_cols] if nan_cols is not None else all_features[col]
                replace_nans_by_col_mean(all_features[col])
                scaler = self.models["Scaler"].get(col, None)
                if scaler is not None:
                    all_features[col] = scaler.transform(all_features[col])
            else:
                # Drop all NaN columns
                if np.isnan(all_features[col]).any():
                    nan_cols = np.isnan(all_features[col]).all(axis=0)
                    if col not in self.models["DroppedNanCols"].keys():
                        self.models["DroppedNanCols"][col] = nan_cols
                    all_features[col] = all_features[col][:, ~nan_cols]

                # Drop rows where NaN still exist
                if np.isnan(all_features[col]).any():
                    replace_nans_by_col_mean(all_features[col])

                # Scale the features
                scaler = MinMaxScaler()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    all_features[col] = scaler.fit_transform(all_features[col])
                if col not in self.models["Scaler"].keys():
                    self.models["Scaler"][col] = scaler

        return all_features, np.concatenate(y, axis=0) if len(y) > 0 else np.array([])



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

        if self.print_times:
            print("         Test size: ", test_size)
        train_ix_all, test_ix_all = train_test_split(list(range(nb_instances)),
                                                     test_size=test_size,
                                                     random_state=self.random_state)
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
        Choose the series to keep from the clusters. From each cluster, the series with the best metric score is kept.

        Returns
        -------
        list
            The series that were chosen.
        """
        chosen = []
        for cluster in self.clusters:
            cluster = list(cluster)
            all_scores = itemgetter(*cluster)(self.evaluation_metric_per_channel)
            max_ix = np.argmax(all_scores)
            chosen.append(cluster[max_ix])
            self.removed_series_corr.update(set(cluster[:max_ix] + cluster[max_ix + 1:]))
        return chosen

    @property
    def sorted_scores(self) -> list:
        """
        Return the series sorted by their metric score.

        Returns
        -------
        list
            The series sorted by their metric score.
        """
        if self._sorted_scores is None:
            self._sorted_scores = sorted(self.evaluation_metric_per_channel, key=lambda k: self.evaluation_metric_per_channel[k], reverse=True if self.higher_is_better else False)
        return self._sorted_scores

    def irrelevant_selector_percentage(self, p: float = 0.75, data_to_filter: dict = None) -> Optional[dict]:
        """
        Remove the channels with the worst score. The percentage of series to keep equals p.

        Parameters
        ----------
        p: float, default=0.75
            The percentage of series to keep. If p=0.75, the 75% series with the best score are kept.
        data_to_filter: dict, default=None
            The data to filter. If None, the data is not filtered.

        Returns
        -------
        dict
            The filtered data
        """
        assert 0 <= p <= 1

        i = len(self.evaluation_metric_per_channel.keys()) - ceil(len(self.evaluation_metric_per_channel.keys()) * p)  # how many items should be removed
        if i == 0:
            return data_to_filter
        threshold = self.evaluation_metric_per_channel[self.sorted_scores[-i]]  # the threshold below which we will delete items
        self.irrelevant_selector_threshold(threshold)
        if data_to_filter is not None:
            return {k: data_to_filter[k] for k in self.sorted_scores if k in data_to_filter.keys()}
        return None

    def irrelevant_selector_threshold(self, threshold) -> None:
        """
        Remove the channels with a score below the threshold.

        Parameters
        ----------
        threshold: float
            The threshold used for removing irrelevant series based on their score. All signals below this
            threshold are removed.
        """
        if self.evaluation_metric_per_channel[self.sorted_scores[-1]] > threshold if self.higher_is_better else self.evaluation_metric_per_channel[self.sorted_scores[-1]] < threshold:
            return
        i = len(self.sorted_scores) - 1
        for i in range(len(self.sorted_scores) - 1, -1, -1):
            if self.evaluation_metric_per_channel[self.sorted_scores[i]] > threshold if self.higher_is_better else self.evaluation_metric_per_channel[self.sorted_scores[i]] < threshold:
                break
            self.removed_series_too_low_metric.add((self.sorted_scores[i], self.evaluation_metric_per_channel[self.sorted_scores[i]]))
        self._sorted_scores = self.sorted_scores[:i + 1]
        self.evaluation_metric_per_channel = {k: self.evaluation_metric_per_channel[k] for k in self.sorted_scores}

    def irrelevant_selector_multiple_models(self, group_sizes: List[float], group_amounts: List[int],
                                            features_channel: Dict[str, Dict[str, np.ndarray]], y_train: pd.Series,
                                            y_test: pd.Series):
        # Divide channels in (1) 3 groups (2) 2 groups
        groups = self.make_groups_multiple_models(group_amounts, group_sizes)

        # Train a model on each of the groups
        ind_scores = {ch: [] for ch in self.evaluation_metric_per_channel.keys()}
        nb_features_per_channel = features_channel[list(self.evaluation_metric_per_channel.keys())[0]]["train"].shape[1]
        for big_group in groups:
            for group in big_group:
                model = LogisticRegression(random_state=self.random_state, penalty='l1', solver='saga')
                features_train = np.concatenate([features_channel[channel]["train"] for channel in group], axis=1)
                model.fit(features_train, y_train)
                self.models["Models"][frozenset(group)] = model

                features_test = np.concatenate([features_channel[channel]["test"] for channel in group], axis=1)
                predictions = model.predict_proba(features_test)
                score = self.evaluation_metric(y_test, predictions)
                importances = np.max(np.abs(model.coef_), axis=0)
                importances = importances / (np.max(importances) - np.min(importances))
                for channel_index, channel in enumerate(group):
                    channel_importance = np.max(importances[channel_index * nb_features_per_channel:
                                                            (channel_index + 1) * nb_features_per_channel])
                    ind_scores[channel].append(score * channel_importance)

        # Weigh the individual score with the group score
        for channel in self.evaluation_metric_per_channel.keys():
            if len(ind_scores[channel]) == 0:
                continue
            self.evaluation_metric_per_channel[channel] = (self.evaluation_metric_per_channel[channel] + np.mean(ind_scores[channel])) / 2

    def make_groups_multiple_models(self, group_amounts, group_sizes):
        assert len(group_sizes) == len(group_amounts)
        all_channels = list(self.evaluation_metric_per_channel.keys())
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

    def irrelevant_selector_random(self, features_train, features_test, y_train, y_test, reference_score):
        """
        For a single channel, train a model on a random permutation of the target variable and check if the score is
        below the reference score. If it is, the channel is considered irrelevant and False is returned.
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
        reference_score:
            The reference score. This should be the metric score of the unshuffled target variable.

        Returns
        -------
        bool
            Whether the channel should be kept or not. If False, the channel is considered irrelevant.

        """
        # Shuffle features train
        np.random.seed(self.random_state)
        random_indices = np.random.permutation(y_train.shape[0])
        y_train_random = y_train.iloc[random_indices]

        # Train model
        model = LogisticRegression(random_state=self.random_state)
        model.fit(features_train, y_train_random)
        predictions = model.predict_proba(features_test)
        score = self.evaluation_metric(y_test, predictions)
        if score > reference_score if self.higher_is_better else score < reference_score:
            print("Random was better with score: ", score, " compared to reference score: ", reference_score)
        return score < reference_score if self.higher_is_better else score > reference_score
