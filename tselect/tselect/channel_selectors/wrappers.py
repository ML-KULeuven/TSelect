import copy
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tsfuse.data import Collection
from tselect.utils import interpolate_nan_3d
from tselect.utils import replace_nans_by_col_mean
from tselect.utils.constants import SEED
from tsfuse.utils import encode_onehot
from tsfuse.transformers import SinglePassStatistics

from TSelect.tselect.tselect.utils.constants import Keys


class SequentialChannelSelector(BaseEstimator, TransformerMixin, ABC):
    def __init__(self, improvement_threshold=0.001, test_size=None, random_state=SEED):
        self.improvement_threshold = improvement_threshold
        self.scaler = None
        self.selected_channels = None
        self.remaining_channels = None
        self.columns = None
        self.map_columns_np = None
        self.index = None
        self.test_size = test_size
        self.random_state = random_state

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_channels is None:
            raise RuntimeError("The fit method must be called before transform.")

        return X[self.selected_channels]

    @abstractmethod
    def fit(self, X: pd.DataFrame, y, force=False):
        pass

    @staticmethod
    def extract_features(X: np.ndarray) -> Dict[int, np.ndarray]:
        features = dict()
        for i in range(X.shape[1]):
            X_i = Collection(X[:, i, :].reshape(X.shape[0], 1, X.shape[2]), from_numpy3d=True)
            features[i] = SinglePassStatistics().transform(X_i).values[:, :, 0]

        return features

    @staticmethod
    def preprocess_extracted_features(features_train: np.ndarray, features_test: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Preprocess the extracted features by dropping NaN columns and scaling the data.

        Parameters
        ----------
        features_train: np.ndarray
            The training features
        features_test: np.ndarray
            The test features

        Returns
        -------
        features_train: np.ndarray
            The preprocessed training features
        features_test: np.ndarray
            The preprocessed test features

        """
        # Drop all NaN columns
        if np.isnan(features_train).any():
            nan_cols = np.isnan(features_train).all(axis=0)
            features_train = features_train[:, ~nan_cols]
            features_test = features_test[:, ~nan_cols]
        # Impute rows where NaN still exist
        if np.isnan(features_train).any():
            replace_nans_by_col_mean(features_train)
            replace_nans_by_col_mean(features_test)
        scaler = MinMaxScaler()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features_train = scaler.fit_transform(features_train)
            features_test = scaler.transform(features_test)
        return features_train, features_test

    def evaluate_model(self, X_train, X_test, y_train, y_test):
        X_train, X_test = self.preprocess_extracted_features(X_train, X_test)
        clf = LogisticRegression(random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)
        if np.isnan(X_test).any():
            replace_nans_by_col_mean(X_test)

        predict_proba = clf.predict_proba(X_test)
        if np.unique(y_test).shape[0] != predict_proba.shape[1]:
            raise ValueError("Not all classes are present in the test set, increase the test size to be able to "
                             "compute the AUC")

        return roc_auc_score(encode_onehot(y_test), predict_proba)



    def evaluate_groups_of_channels(self, features_all: Dict[int, np.ndarray], groups_to_evaluate: List[list],
                                    train_ix: Iterable, test_ix: Iterable, y_train: pd.Series, y_test: pd.Series) -> (int, float):
        """
        Compute the initial scores for the forward selection process.

        Parameters
        ----------
        features_all: Dict[str, np.ndarray]
            The features to use for the selection process
        groups_to_evaluate: List[list]
            The groups of channels to evaluate. Each group is a list of channel indices.
        train_ix: Iterable
            The indices of the training set
        test_ix: Iterable
            The indices of the test set
        y_train: pd.Series
            The target variable for the training set
        y_test: pd.Series
            The target variable for the test set

        Returns
        -------
        (int, float)
            The index of the best group of channels and its score
        """
        scores = []
        for group in groups_to_evaluate:
            X = []
            for ch in group:
                X.append(features_all[ch])
            X = np.concatenate(X, axis=1)
            X_train = X[train_ix, :]
            X_test = X[test_ix, :]
            score = self.evaluate_model(X_train, X_test, y_train, y_test)
            scores.append(score)

        best_group = np.argmax(scores)
        best_score = scores[best_group]
        return best_group, best_score


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

    def train_test_split(self, X: np.ndarray) -> (list, list):
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
        nb_instances = X.shape[0]
        test_size = self.compute_test_size(nb_instances)
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


    def update_metadata(self, metadata):
        """
        Update the metadata with the results of the channel selector.
        """
        if metadata:
            metadata[Keys.series_filtering][Keys.series_filter].append(self)

class ForwardChannelSelector(SequentialChannelSelector):
    def __init__(self, improvement_threshold=0.001, test_size=None, random_state=SEED):
        super().__init__(improvement_threshold=improvement_threshold, test_size=test_size, random_state=random_state)

    def fit(self, X: pd.DataFrame, y, force=False, metadata=None):
        y = copy.deepcopy(y)
        X_np = self.preprocessing(X)
        self.columns = X.columns
        self.map_columns_np = {col: i for i, col in enumerate(X.columns)}
        self.index = X.index

        if self.selected_channels is not None and not force:
            return None

        n_channels = X_np.shape[1]
        all_channels = list(range(n_channels))
        self.selected_channels = []
        self.remaining_channels = set(all_channels)

        features_all = self.extract_features(X_np)
        train_ix, test_ix = self.train_test_split(X_np)
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

        current_groups = [[ch] for ch in all_channels]
        best_score = 0
        while len(self.remaining_channels) > 0:
            best_group_index, current_score = self.evaluate_groups_of_channels(features_all, current_groups, train_ix,
                                                                               test_ix, y_train, y_test)

            if current_score - best_score < self.improvement_threshold:
                break

            best_score = current_score
            best_group = current_groups[best_group_index]
            last_added_ch = best_group[-1]
            self.selected_channels.append(last_added_ch)
            self.remaining_channels.remove(last_added_ch)
            current_groups = [best_group + [ch] for ch in self.remaining_channels]

        for i, ch in enumerate(self.selected_channels):
            self.selected_channels[i] = self.columns[ch]

        self.update_metadata(metadata)
        return None

class BackwardChannelSelector(SequentialChannelSelector):
    def __init__(self, improvement_threshold=0.001, test_size=None, random_state=SEED):
        super().__init__(improvement_threshold=improvement_threshold, test_size=test_size, random_state=random_state)

    def fit(self, X: pd.DataFrame, y, force=False, metadata=None):
        y = copy.deepcopy(y)
        X_np = self.preprocessing(X)
        self.columns = X.columns
        self.map_columns_np = {col: i for i, col in enumerate(X.columns)}
        self.index = X.index

        if self.selected_channels is not None and not force:
            return None

        n_channels = X_np.shape[1]
        all_channels = list(range(n_channels))
        self.selected_channels = copy.deepcopy(all_channels)
        self.remaining_channels = set()

        features_all = self.extract_features(X_np)
        train_ix, test_ix = self.train_test_split(X_np)
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

        _, best_score = self.evaluate_groups_of_channels(features_all, [self.selected_channels], train_ix, test_ix, y_train, y_test)
        current_groups = [[c for c in self.selected_channels if c != ch] for ch in self.selected_channels]

        while len(self.selected_channels) > 1:
            best_group_index, current_score = self.evaluate_groups_of_channels(features_all, current_groups, train_ix,
                                                                               test_ix, y_train, y_test)

            if best_score > current_score:
                break

            best_score = current_score
            best_group = current_groups[best_group_index]
            last_removed_ch = self.selected_channels[best_group_index]
            self.selected_channels.remove(last_removed_ch)
            self.remaining_channels.add(last_removed_ch)
            current_groups = [[c for c in best_group if c != ch] for ch in self.selected_channels]

        for i, ch in enumerate(self.selected_channels):
            self.selected_channels[i] = self.columns[ch]

        self.update_metadata(metadata)
        return None
