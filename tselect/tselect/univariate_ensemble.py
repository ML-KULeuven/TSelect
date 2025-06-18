import time
import warnings
from typing import List, Callable

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sktime.datatypes._panel._convert import from_multi_index_to_3d_numpy

from tselect import SEED, Keys, replace_nans_by_col_mean, pad_until_length_tsfuse, pad_until_length
from tselect.abstract_extractor import AbstractExtractor
from tsfuse.construction.mlj20 import TSFuseExtractor
from tsfuse.data import Collection
from tsfuse.transformers import SinglePassStatistics
from sktime.transformations.panel.rocket import MiniRocket

from tselect.config import Config, get_default_config

class UnivariateEnsemble(AbstractExtractor, TransformerMixin):
    def __init__(self, series_fusion: bool = False,
                 tselect_config: Config = get_default_config(),
                 # irrelevant_filter=True,
                 # redundant_filter=True,
                 # auc_percentage: float = 0.75,
                 # auc_threshold: float = 0.5,
                 # corr_threshold: float = 0.7,
                 # feature_extractor: Callable = None,
                 # test_size: float = 0.2,
                 views: List[int] = None,
                 add_tags=lambda x: x,
                 compatible=lambda x: x,
                 random_state: int = SEED,
                 features=Keys.statistics):
        """
        The constructor for UnivariateEnsemble class.

        Parameters
        ----------
        series_fusion : bool, optional, default False
            Whether to derive new signals from the original ones ("fusion").
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
        random_state : int, optional, default SEED
            The random state used throughout the class.
        features: str, default='statistics'
            Which features to extract. Can be either 'statistics' for 8 statistical features or 'minirocket' for the
            minirocket features.
        """
        super().__init__(series_fusion, tselect_config, views, add_tags, compatible, random_state)
        if features != Keys.statistics and features != Keys.minirocket:
            raise ValueError(f"Features should be either {Keys.statistics} or {Keys.minirocket}")
        self.features = features
        self.features_minirocket_models = {}
        self._dropped_nan_cols = {}
        self._individual_models = {}
        self._scaler_cols = {}
        self.random_state = random_state

    @property
    def individual_models(self):
        if self.series_filter is None:
            return self._individual_models
        else:
            return self.series_filter.models["Models"]

    @property
    def dropped_nan_cols(self):
        if self.series_filter is None:
            return self._dropped_nan_cols
        else:
            return self.series_filter.models["DroppedNanCols"]

    @property
    def scaler_cols(self):
        if self.series_filter is None:
            return self._scaler_cols
        else:
            return self.series_filter.models["Scaler"]

    @property
    def series_filter(self):
        return self.fusion_filter.series_filter

    @property
    def tsfuse_extractor(self) -> TSFuseExtractor:
        return self.fusion_filter.tsfuse_extractor

    def predict_proba(self, X, nb_classes, metadata=None):
        start = time.process_time()
        features = self.transform(X)
        if metadata is not None:
            metadata[Keys.time_compute] = time.process_time() - start
            metadata[Keys.time_predict] = 0
            metadata[Keys.features] = features
        if not self.fusion_filter.irrelevant_filter and not self.fusion_filter.redundant_filter:
            if self.fusion_filter.series_fusion and self.series_filter is None:
                all_signals = list(map(str, self.tsfuse_extractor.series_))
            elif not self.fusion_filter.series_fusion and self.series_filter is None:
                if isinstance(X, dict):
                    all_signals = list(X.keys())
                elif isinstance(X, pd.DataFrame):
                    all_signals = X.columns
                else:
                    raise ValueError("X should be either a dictionary or a pandas DataFrame")
            else:
                all_signals = self.series_filter.columns
        else:
            all_signals = self.series_filter.filtered_series
        votes = np.empty((features.shape[0], len(all_signals), nb_classes), dtype=np.float64)

        for i, signal in enumerate(all_signals):
            start = time.process_time()
            model = self.individual_models[signal]
            features_i = features[[c for c in features.columns if c.startswith(signal + "_")]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                votes[:, i, :] = model.predict_proba(features_i)
            if metadata is not None:
                metadata[Keys.time_predict] += time.process_time() - start

        start = time.process_time()
        proba = np.mean(votes, axis=1)
        if metadata is not None:
            metadata[Keys.time_predict] += time.process_time() - start
        return proba

    def transform_model(self, X: pd.DataFrame) -> pd.DataFrame:
        features = []
        if self.features == Keys.statistics:
            columns = X.columns

            # If there is no fusion, the signals aren't scaled yet. Otherwise, they are scaled, but should be converted
            # to the 3D numpy format.
            if not self.fusion_filter.series_fusion:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X = self.series_filter.scaler.transform(X)
            else:
                X = from_multi_index_to_3d_numpy(X)

            for i, signal in enumerate(columns):
                features_i = self.extract_features(X, signal, i, tsfuse_format=False)
                features.append(pd.DataFrame(features_i, columns=[signal + "_" + str(j) for j in range(features_i.shape[1])]))

        elif self.features == Keys.minirocket:
            for i, signal in enumerate(X.columns):
                features_i = self.extract_features_minirocket(X, signal, tsfuse_format=isinstance(X, dict))
                features_i.columns = [signal + "_" + str(j) for j in range(features_i.shape[1])]
                features.append(features_i)
        else:
            raise ValueError(f"Features should be either {Keys.statistics} or {Keys.minirocket}")

        return pd.concat(features, axis=1)

    def fit_model(self, X, y):
        if self.features == Keys.statistics:
            if not self.fusion_filter.irrelevant_filter and not self.fusion_filter.redundant_filter:
                if self.series_filter is None:
                    self.fusion_filter.series_filtering = True
                    self.fusion_filter.__init_filter__()
                    self.fusion_filter.series_filtering = False
                print("Models are fitting")
                self.series_filter.fit(X, y)
        elif self.features == Keys.minirocket:
            nb_features = 10000//len(X.columns)
            if nb_features < 84:
                warnings.warn(f"Number of features is {nb_features}, which is less than the required 84. "
                              f"Setting it to 84.")
                print(f"Warning: Number of features is {nb_features}, which is less than the required 84. "
                      f"Setting it to 84.")
                nb_features = 84
            for i, signal in enumerate(X.columns):
                features = self.extract_features_minirocket(X, signal, nb_features, tsfuse_format=isinstance(X, dict))
                self.individual_models[signal] = LogisticRegression(random_state=self.random_state)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.individual_models[signal].fit(features, y)
        else:
            raise ValueError(f"Features should be either {Keys.statistics} or {Keys.minirocket}")
        return None

    def extract_features(self, X, col, i, tsfuse_format=False) -> np.ndarray:
        """
        Extract the 8 statistical features for a single dimension.

        Parameters
        ----------
        X: 3D numpy array or dictionary of Collections
            The data to fit the filter to.
        col: str
            The name of the dimension to extract the features from
        i: int
            The index of the dimension to extract the features from, needed for the raw or catch22 mode.
        tsfuse_format: bool, default=False
            Whether the data `X` is in TSFuse format or not.

        Returns
        -------
        np.ndarray
            The extracted features
        """
        if not tsfuse_format:
            X_i = Collection(X[:, i, :].reshape(X.shape[0], 1, X.shape[2]), from_numpy3d=True)
        else:
            X_i = X[col]
        features = SinglePassStatistics().transform(X_i).values[:, :, 0]
        # Drop same NaN columns as training set, otherwise the LR model won't be happy
        if col in self.dropped_nan_cols.keys():
            features = features[:, ~self.dropped_nan_cols[col]]

        if np.isnan(features).any():
            replace_nans_by_col_mean(features)

        if col in self.scaler_cols.keys():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                features = self.scaler_cols[col].transform(features)

        return features

    def extract_features_minirocket(self, X, col, nb_features=None, tsfuse_format=False) -> np.ndarray:
        """
        Extract the minirocket features for a single dimension.

        Parameters
        ----------
        X: 3D numpy array or dictionary of Collections
            The data to fit the filter to.
        col: str
            The name of the dimension to extract the features from
        nb_features: int, default=None
            The number of features to extract. If None, the models should already be trained, otherwise an error is
            raised.
        tsfuse_format: bool, default=False
            Whether the data `X` is in TSFuse format or not.

        Returns
        -------
        np.ndarray
            The extracted features
        """

        if tsfuse_format:
            X_i = X[col]
            X_i = X_i.to_pd_multiindex(col)
        else:
            X_i = X[[col]]
        if not tsfuse_format and X_i.index.levels[1].shape[0] < 9:
            X_i = pad_until_length(X_i, 9)
        elif tsfuse_format and X_i[list(X_i.keys())[0]].shape[1] < 9:
            X_i = pad_until_length_tsfuse(X_i, 9)
        if col in self.features_minirocket_models.keys():
            features = self.features_minirocket_models[col].transform(X_i)
            already_trained = True
        else:
            already_trained = False
            if nb_features is None:
                raise ValueError("nb_features should be specified when fitting the model")
            self.features_minirocket_models[col] = MiniRocket(num_kernels=nb_features, random_state=self.random_state)
            self.features_minirocket_models[col].fit(X_i)
            features = self.features_minirocket_models[col].transform(X_i)

        if already_trained and col in self.dropped_nan_cols.keys():
            features = features.drop(columns=self.dropped_nan_cols[col], axis=1)
        elif not already_trained:
            new_features = features.dropna(how='all', axis=1)
            self.dropped_nan_cols[col] = features.columns.difference(new_features.columns)
            features = new_features

        if features.isna().any().any():
            mean_cols = features.mean()
            features = features.fillna(mean_cols)

        return features
