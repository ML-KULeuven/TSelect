from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sktime.datatypes._panel._convert import from_multi_index_to_3d_numpy


class MinMaxScaler3D(TransformerMixin):
    """
    A class to min-max normalize 3 dimensional time series data
    """
    def __init__(self):
        self.min = None
        self.max = None

    def transform(self, X) -> np.ndarray:
        """
        Transform the data by normalizing it.
        """
        if isinstance(X, pd.DataFrame):
            X = from_multi_index_to_3d_numpy(X)
        normalized_data = (X - self.min[np.newaxis, :, np.newaxis]) / (
                    self.max[np.newaxis, :, np.newaxis] - self.min[np.newaxis, :, np.newaxis])
        return normalized_data

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """
        Fit the data and transform it by normalizing it.
        """
        if isinstance(X, pd.DataFrame):
            X = from_multi_index_to_3d_numpy(X)
        self.min = np.nanmin(X, axis=(0, 2))  # Calculate the minimum values along the dimensions
        self.max = np.nanmax(X, axis=(0, 2))  # Calculate the maximum values along the dimensions

        normalized_data = (X - self.min[np.newaxis, :, np.newaxis]) / (
                    self.max[np.newaxis, :, np.newaxis] - self.min[np.newaxis, :, np.newaxis])
        return normalized_data


class MinMaxScalerCollections(TransformerMixin):
    """
    A class to min-max normalize a dictionary of Collections
    """

    def __init__(self):
        self.min = {}
        self.max = {}

    def transform(self, X: Union[dict, pd.DataFrame]) -> Union[dict, pd.DataFrame]:
        """
        Transform the data by normalizing it.
        """
        from tsfuse.data import Collection
        if isinstance(X, pd.DataFrame):
            return self.transform_pd(X)
        for key in X.keys():
            new_values = (X[key].values - self.min[str(key)]) / (self.max[str(key)] - self.min[str(key)])
            X[key] = Collection(new_values, X[key].index, X[key].dimensions, X[key].mask_value, X[key].unit,
                                X[key].tags)
        return X

    def transform_pd(self, X_pd: pd.DataFrame):
        for col in X_pd.columns:
            c_min = self.min[col]
            c_max = self.max[col]
            X_pd[col] = (X_pd[col] - c_min) / (c_max - c_min)
        return X_pd

    def fit_transform(self, X: dict, y=None, inplace=True, **fit_params) -> dict:
        """
        Fit the data and transform it by normalizing it.
        """
        from tsfuse.data import Collection
        if not inplace:
            X = X.copy()
        for key in X.keys():
            self.min[str(key)] = np.nanmin(X[key].values)
            self.max[str(key)] = np.nanmax(X[key].values)
            new_values = (X[key].values - self.min[str(key)]) / (self.max[str(key)] - self.min[str(key)])
            X[key] = Collection(new_values, X[key].index, X[key].dimensions, X[key].mask_value, X[key].unit,
                                X[key].tags)
        return X


class StandardScaler3D(TransformerMixin):
    """
    A class to standardize time series data. It is a wrapper around sklearn's StandardScaler class.
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def transform(self, X) -> np.ndarray:
        """
        Transform the data by standardizing it.
        """
        if isinstance(X, pd.DataFrame):
            X = from_multi_index_to_3d_numpy(X)
        normalized_data = (X - self.mean[np.newaxis, :, np.newaxis]) / (
                    self.std[np.newaxis, :, np.newaxis])
        return normalized_data

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """
        Fit the data and transform it by standardizing it.
        """
        if isinstance(X, pd.DataFrame):
            X = from_multi_index_to_3d_numpy(X)
        self.mean = np.mean(X, axis=(0, 2))  # Calculate the mean values along the dimensions
        self.std = np.std(X, axis=(0, 2))  # Calculate the standard deviation values along the dimensions

        normalized_data = (X - self.mean[np.newaxis, :, np.newaxis]) / (
                    self.std[np.newaxis, :, np.newaxis])
        return normalized_data

