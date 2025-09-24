import copy
import warnings
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tselect.utils.constants import SEED
from tsfuse.utils import encode_onehot

from tselect.utils.constants import Keys


class SequentialChannelSelector(BaseEstimator, TransformerMixin, ABC):
    def __init__(self, model=None, improvement_threshold=0.001, test_size=None, random_state=SEED):
        if model is None:
            model = LogisticRegression(random_state=random_state)
        self.model = model
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


    def evaluate_model(self, X_train, X_test, y_train, y_test):
        try:
            clf = clone(self.model)
        except TypeError:
            clf = copy.deepcopy(self.model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)
        if pd.isna(X_test).any().any():
            pd.fillna(X_test, inplace=True)

        predict_proba = clf.predict_proba(X_test)
        if np.unique(y_test).shape[0] != predict_proba.shape[1]:
            raise ValueError("Not all classes are present in the test set, increase the test size to be able to "
                             "compute the AUC")

        return roc_auc_score(encode_onehot(y_test), predict_proba)



    def evaluate_groups_of_channels(self, X: pd.DataFrame, y:pd.Series, groups_to_evaluate: List[list]) -> (int, float):
        """
        Compute the initial scores for the forward selection process.

        Parameters
        ----------
        groups_to_evaluate: List[list]
            The groups of channels to evaluate. Each group is a list of channel indices.

        Returns
        -------
        (int, float)
            The index of the best group of channels and its score
        """
        scores = []
        for group in groups_to_evaluate:
            X_i = X[group]
            X_train, X_test, y_train, y_test = self.train_test_split(X_i, y)
            # X = []
            # for ch in group:
            #     X.append(features_all[ch])
            # X = np.concatenate(X, axis=1)
            # X_train = X[train_ix, :]
            # X_test = X[test_ix, :]
            score = self.evaluate_model(X_train, X_test, y_train, y_test)
            scores.append(score)

        best_group = np.argmax(scores)
        best_score = scores[best_group]
        return best_group, best_score


    # def preprocessing(self, X: pd.DataFrame) -> np.ndarray:
    #     """
    #     Preprocess the data before fitting the filter.
    #
    #     Parameters
    #     ----------
    #     X: pd.DataFrame
    #         The data to preprocess
    #
    #     Returns
    #     -------
    #     np.ndarray
    #         The preprocessed data
    #
    #     """
    #     from tselect import MinMaxScaler3D
    #     self.scaler = MinMaxScaler3D()
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         X_np = self.scaler.fit_transform(X)
    #     if np.isnan(X_np).any():
    #         interpolate_nan_3d(X_np, inplace=True)
    #     return X_np

    def train_test_split(self, X: pd.DataFrame, y: pd.Series) -> (list, list):
        """
        Splits a MultiIndex DataFrame with (instance, timepoint) index into train and test sets.

        Parameters:
        - X (pd.DataFrame): MultiIndex DataFrame with index (instance, timepoint)
        - test_size (float): Fraction of instances to include in the test split
        - random_state (int): Seed for reproducibility

        Returns:
        - df_train (pd.DataFrame): Training split
        - df_test (pd.DataFrame): Testing split
        """
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("DataFrame must have a MultiIndex (instance, timepoint)")

        # Extract unique instances
        instances = X.index.get_level_values(0).unique()

        test_size = self.compute_test_size(len(instances))

        # Align y with instances to ensure correct indexing
        y = y.loc[instances]

        # Perform stratified split
        train_instances, test_instances, y_train, y_test = train_test_split(
            instances,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )

        # Create masks to filter the DataFrame
        train_mask = X.index.get_level_values(0).isin(train_instances)
        test_mask = X.index.get_level_values(0).isin(test_instances)

        df_train = X[train_mask]
        df_test = X[test_mask]

        return df_train, df_test, y_train, y_test

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
    def __init__(self, model=None, improvement_threshold=0.001, test_size=None, random_state=SEED):
        super().__init__(model=model, improvement_threshold=improvement_threshold, test_size=test_size,
                         random_state=random_state)

    def fit(self, X: pd.DataFrame, y, force=False, metadata=None):
        y = copy.deepcopy(y)
        # X_np = self.preprocessing(X)
        # self.columns = X.columns
        # self.map_columns_np = {col: i for i, col in enumerate(X.columns)}
        # self.index = X.index

        if self.selected_channels is not None and not force:
            return None

        # n_channels = len(X.columns)
        all_channels = X.columns
        self.selected_channels = []
        self.remaining_channels = set(all_channels)

        # features_all = self.extract_features(X_np)
        # train_ix, test_ix = self.train_test_split(X)
        # y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

        current_groups = [[ch] for ch in all_channels]
        best_score = 0
        while len(self.remaining_channels) > 0:
            best_group_index, current_score = self.evaluate_groups_of_channels(X, y, current_groups)

            if current_score - best_score < self.improvement_threshold:
                break

            best_score = current_score
            best_group = current_groups[best_group_index]
            last_added_ch = best_group[-1]
            self.selected_channels.append(last_added_ch)
            self.remaining_channels.remove(last_added_ch)
            current_groups = [best_group + [ch] for ch in self.remaining_channels]

        # for i, ch in enumerate(self.selected_channels):
        #     self.selected_channels[i] = self.columns[ch]

        self.update_metadata(metadata)
        return None

class BackwardChannelSelector(SequentialChannelSelector):
    def __init__(self, model=None, improvement_threshold=0.001, test_size=None, random_state=SEED):
        super().__init__(model=model, improvement_threshold=improvement_threshold, test_size=test_size,
                         random_state=random_state)

    def fit(self, X: pd.DataFrame, y, force=False, metadata=None):
        y = copy.deepcopy(y)
        # X_np = self.preprocessing(X)
        # self.columns = X.columns
        # self.map_columns_np = {col: i for i, col in enumerate(X.columns)}
        # self.index = X.index

        if self.selected_channels is not None and not force:
            return None


        all_channels = list(X.columns)
        self.selected_channels = copy.deepcopy(all_channels)
        self.remaining_channels = set()
        # n_channels = X_np.shape[1]
        # all_channels = list(range(n_channels))
        # self.selected_channels = copy.deepcopy(all_channels)
        # self.remaining_channels = set()

        # features_all = self.extract_features(X_np)
        # train_ix, test_ix = self.train_test_split(X_np)
        # y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

        # _, best_score = self.evaluate_groups_of_channels(features_all, [self.selected_channels], train_ix, test_ix, y_train, y_test)
        _, best_score = self.evaluate_groups_of_channels(X, y, [self.selected_channels])
        current_groups = [[c for c in self.selected_channels if c != ch] for ch in self.selected_channels]

        while len(self.selected_channels) > 1:
            best_group_index, current_score = self.evaluate_groups_of_channels(X, y, current_groups)

            if best_score > current_score:
                break

            best_score = current_score
            best_group = current_groups[best_group_index]
            last_removed_ch = self.selected_channels[best_group_index]
            self.selected_channels.remove(last_removed_ch)
            self.remaining_channels.add(last_removed_ch)
            current_groups = [[c for c in best_group if c != ch] for ch in self.selected_channels]

        # for i, ch in enumerate(self.selected_channels):
        #     self.selected_channels[i] = self.columns[ch]

        self.update_metadata(metadata)
        return None
