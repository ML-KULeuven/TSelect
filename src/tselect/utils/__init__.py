import copy
from typing import Union, Dict, List, Optional

import numpy as np
import pandas as pd
import pycatch22
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sktime.datatypes._panel._convert import from_multi_index_to_3d_numpy

from tselect.utils.constants import Keys

from tsfuse.data import Collection


# \Section: Project general
def average(x: list):
    """
    Calculates the average of a list of numbers.

    Parameters
    ----------
    x : list
        The list of numbers to calculate the average of.

    Returns
    -------
    float
        The average of the list of numbers.
    """
    if len(x) == 0:
        return 0
    return sum(x) / len(x)


def remove_if_exists(x, ls: list) -> None:
    """
    Removes an element from a list if it exists in the list.

    Parameters
    ----------
    x : any
        The element to remove from the list.
    ls : list
        The list to remove the element from.

    Returns
    -------
    None, the list is adapted in place.
    """
    if x in ls:
        ls.remove(x)


def round_sign_fig(x: float, i: int = 5) -> float:
    """
    Rounds a float to a specified number of significant figures.

    Parameters
    ----------
    x : float
        The float to round.
    i : int, optional
        The number of significant figures to round to, by default 5.

    Returns
    -------
    float
        The rounded float
    """
    return float('%.{}g'.format(i) % x)


def interpolate_nan_3d(data: np.ndarray, method='pad', order=5, inplace=True):
    """
    Interpolates missing values in a 3D numpy array.

    Parameters
    ----------
    data : np.ndarray
           The 3D numpy array to interpolate.
    method : str, optional
           The interpolation method to use, by default 'time'.
    order : int, optional
           The order of the interpolation, by default 5.
    inplace : bool, optional
           Whether to perform the interpolation in place, by default True.

    Returns
    -------
    np.ndarray if
        The interpolated time series.
    """
    nan_cols = np.isnan(data).any(axis=0).any(axis=1)
    # Iterate over each time series in the reshaped data
    x = data if inplace else copy.deepcopy(data)
    for d in range(x.shape[1]):
        if not nan_cols[d]:
            continue
        for i in range(x.shape[0]):
            series = pd.Series(x[i, d, :])
            series.interpolate(inplace=True, method=method, order=order)
            nan = series.isnull().values
            if nan.any():
                if nan.all():
                    series.fillna(0, inplace=True)
                else:
                    series.fillna(method='bfill', inplace=True)

    if not inplace:
        return x


def ffill_nan(X_np, inplace=True) -> np.ndarray:
    """
    Fills NaN values in a 2D numpy array with the previous value. If the first value is NaN, it is filled with 0.

    Parameters
    ----------
    X_np : np.ndarray
        The 2D numpy array to fill NaN values in.
    inplace : bool, optional
        Whether to perform the filling in place, by default True.

    Returns
    -------
    np.ndarray
        The 2D numpy array with NaN values filled.
    """
    if not inplace:
        X_np = copy.deepcopy(X_np)
    for row_idx in range(X_np.shape[0]):
        for col_idx in range(1, X_np.shape[1]):
            if np.isnan(X_np[row_idx, col_idx]):
                if col_idx == 0:
                    X_np[row_idx, col_idx] = 0
                else:
                    X_np[row_idx, col_idx] = X_np[row_idx, col_idx - 1]
    return X_np


def pad_until_length(x: Union[pd.DataFrame, pd.Series], length: int) -> pd.DataFrame:
    """
    Pads a dataframe or series until a specified length. The last value is used for padding.

    Parameters
    ----------
    x : Union[pd.DataFrame, pd.Series]
        The dataframe or series to pad.
    length : int
        The length to pad until.

    Returns
    -------
    pd.DataFrame
        The padded dataframe or series.
    """
    nb_timepoints = x.index.levels[1].shape[0]
    all_indices = list(x.index)
    new_values = []
    for i in range(nb_timepoints - 1, len(all_indices), nb_timepoints):
        index = all_indices[i]
        value = x.loc[index]
        new_indexes = [(index[0], t) for t in range(nb_timepoints, length)]
        new_values.append(pd.DataFrame(np.tile(value, (length - nb_timepoints, 1)), index=new_indexes,
                                       columns=x.columns))

    return pd.concat([x, *new_values]).sort_index()


def pad_until_length_np(x: np.ndarray, length: int) -> np.ndarray:
    """
    Pads a 3D numpy array until a specified length. The last value is used for padding.

    Parameters
    ----------
    x : np.ndarray
        The 3D numpy array to pad.
    length : int
        The length to pad until.

    Returns
    -------
    np.ndarray
        The padded 3D numpy array.
    """
    if x.shape[2] >= length:
        return x
    N, D, t = x.shape
    to_pad = length - t
    last_value = np.repeat(x[:, :, -1], to_pad).reshape(N, D, to_pad)
    x_padded = np.concatenate((x, last_value), axis=2)
    return x_padded


def pad_until_length_tsfuse(x: Dict[Union[str, int], Collection], length: int, inplace=True) \
        -> Optional[Dict[Union[str, int], Collection]]:
    """
    Pads data in the TSFuse format until a specified length. The last value is used for padding.

    Parameters
    ----------
    x : Dict[Union[str, int], Collection]
        The data in TSFuse format to pad.
    length : int
        The length to pad until.
    inplace : bool, optional, default True
        Whether to perform the padding in place.

    Returns
    -------
    Optional[Dict[Union[str, int], Collection]]
        The padded data in TSFuse format or None if inplace is True.
    """
    if not inplace:
        x = copy.deepcopy(x)
    for key in x.keys():
        if x[key].values.shape[1] >= length:
            continue
        N, t = x[key].values.shape
        to_pad = length - t
        last_value = np.repeat(x[key].values[:, -1], to_pad).reshape(N, to_pad)
        new_values = np.concatenate((x[key].values, last_value), axis=1)
        x[key] = Collection(new_values, x[key].index, x[key].dimensions, x[key].mask_value, x[key].unit, x[key].tags)

    if not inplace:
        return x


def remove_trailing_nans_multiindex(x: pd.Series) -> pd.Series:
    """
    Removes trailing NaN values from a series in Pandas MultiIndex format.

    Parameters
    ----------
    x : pd.Series
        The series to remove trailing NaN values from.

    Returns
    -------
    pd.Series
        The series without trailing NaN values.
    """
    t = x.index.levels[1].shape[0]
    x0 = x.loc[(0, 0):(0, t)]
    max_t = np.max(np.where(~np.isnan(x0)))
    x_sliced = x[x.index.get_level_values(1) <= max_t]
    x_sliced.index = create_multiindex(x.index.levels[0].to_list(), max_t + 1)
    return x_sliced


def remove_trailing_nans_np(x: np.ndarray) -> np.ndarray:
    """
    Removes trailing NaN values from a 3D numpy array.

    Parameters
    ----------
    x : np.ndarray
        The 3D numpy array to remove trailing NaN values from.

    Returns
    -------
    np.ndarray
        The 3D numpy array without trailing NaN values.
    """
    max_t = np.max(np.where(~np.isnan(x)))
    return x[:, :max_t + 1]


def min_max_normalization(data: np.ndarray) -> np.ndarray:
    """
    Performs min-max normalization on a 3D numpy array, treating each instance separately. NaN values are ignored.

    Parameters
    ----------
    data : np.ndarray
        The 3D numpy array to normalize.

    Returns
    -------
    np.ndarray
        The normalized 3D numpy array.
    """
    min_values = np.nanmin(data, axis=(1, 2))
    max_values = np.nanmax(data, axis=(1, 2))
    normalized_data = (data - min_values[:, np.newaxis, np.newaxis]) / (
            max_values[:, np.newaxis, np.newaxis] - min_values[:, np.newaxis, np.newaxis])
    return normalized_data


# \Section: Project specific
def sort_on_mi(X: pd.DataFrame, y: pd.Series, discrete_features: bool = False) -> list:
    """
    Sorts the columns of the given dataframe on mutual information with the target variable `y`.

    Parameters
    ----------
    X: pd.DataFrame
        the dataframe whose columns will get sorted
    y: pd.Series
        the target variable
    discrete_features: [Optional] bool (default = False)
        Are all features in `X` discrete?

    Returns
    -------
    A list containing the sorted column names. The earlier in the list, the higher the mutual information with the
    target variable.

    """
    task = detect_task(y)
    f_mi = mutual_info_classif if task == Keys.task_classification else mutual_info_regression
    mi = []
    for col in X.columns:
        X_col = copy.deepcopy(X[[col]])
        if X_col.isin([np.inf, -np.inf]).any().bool():
            X_col.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_col.dropna(axis=0, inplace=True)
            y_col = y.loc[X_col.index]
        elif X_col.isin([np.nan]).any().bool():
            X_col.dropna(axis=0, inplace=True)
            if X_col.shape[0] <= X.shape[0] // 2:  # too few instances remain for this feature
                mi.append((0, col))
                continue
            y_col = y.loc[X_col.index]
        else:
            y_col = y
        mi_col = f_mi(X_col, y_col, discrete_features=discrete_features)
        mi.append((mi_col, col))

    sorted_mi = sorted(mi, reverse=True)  # sort mi in descending order
    sorted_cols = [c for (_, c) in sorted_mi]
    return sorted_cols


def average_by_matching_key(X: Union[np.array, list]) -> dict:
    """
    Calculates the average of the values in `X` per key in its dictionaries. At the first level, X is a
    list of dictionaries. At the second level are the dictionaries, whose values will be averaged. The average is the
    average over that key in all dictionaries and equals the sum of the corresponding values (over all dictionaries)
    divided by the number this key occurs in the dictionaries.

    Parameters
    ----------
    X: Union[np.array, list]
        The list of dictionaries whose values will be averaged.

    Returns
    -------
    dict
        The dictionary containing the averages. The keys are the keys of the dictionaries in `X`. The values are the
        averages of the corresponding keys in the dictionaries in `X`.
    """
    result = {}
    for fold in X:
        for key, value in fold.items():
            if key in result:
                result[key] = (result[key][0] + value, result[key][1] + 1)
            else:
                result[key] = (value, 1)

    for key, (value, i) in result.items():
        result[key] = value / i

    return result


def get_nb_instances_multiindex(X: pd.DataFrame) -> int:
    """
    Returns the number of instances in a dataframe in Pandas MultiIndex format.

    Parameters
    ----------
    X: pd.DataFrame
        the dataframe in Pandas MultiIndex format

    Returns
    -------
    int
        the number of instances in the dataframe
    """
    return X.index.get_level_values(0).value_counts().shape[0]


def detect_task(y):
    """
    Detects the task of the target variable `y`. If `y` is a float, the task is regression. Otherwise, the task is
    classification.

    Parameters
    ----------
    y: pd.Series
        the target variable

    Returns
    -------
    str
        the task of the target variable, either "regression" or "classification"
    """
    if np.issubdtype(y, np.float64):
        return Keys.task_regression
    else:
        return Keys.task_classification


def format_y(y):
    """
    Formats the target variable `y` such that it has values 0, 1, 2, ... . If `y` is not numeric, it is encoded
    first. Then, the values are shifted so that the values start with 0 . If this is already the case, y is returned
    unchanged.

    Parameters
    ----------
    y: pd.Series, np.ndarray or list
        the target variable

    Returns
    -------
    pd.Series, np.ndarray or list
        the formatted target variable

    """
    if not pd.api.types.is_numeric_dtype(y):
        y = encode_y(y)
    if isinstance(y, pd.Series):
        unique_values = y.unique()
        max_value = unique_values.max(initial=0)

    elif isinstance(y, np.ndarray):
        unique_values = np.unique(y)
        max_value = np.max(y)
    elif isinstance(y, list):
        unique_values = set(y)
        max_value = max(unique_values)
    else:
        raise ValueError("y must be a pd.Series, np.ndarray or list")

    length = len(unique_values)
    if max_value >= length:
        return y - (max_value - length + 1)
    else:
        return y


def encode_y(y):
    """
    Encodes the target variable `y` such that it has values 0, 1, 2, ... .

    Parameters
    ----------
    y: pd.Series, np.ndarray or list
        the target variable

    Returns
    -------
    pd.Series, np.ndarray or list
        the encoded target variable
    """
    distinct_values = y.unique()
    mapping = {val: i for i, val in enumerate(distinct_values)}
    return y.map(mapping)


def mapping_index(index1, index2) -> dict:
    """
    Creates a mapping from `index1` to `index2`. The mapping is a dictionary with keys the items of `index1` and values
    the items of `index2`.

    Parameters
    ----------
    index1: list
        the first index
    index2: list
        the second index
    """
    mapping = {}
    for i, v in enumerate(index1):
        mapping[v] = index2[i]
    return mapping


def init_metadata():
    """
    Initializes the metadata dictionary.
    """
    metadata = {Keys.time_series_to_series: [],
                Keys.time_series_filtering: [],
                Keys.time_series_to_attr: [],
                Keys.time_attr_to_attr: [],
                Keys.time_select: [],
                Keys.fused_series: [],
                Keys.extracted_attr: [],
                Keys.fused_attr: [],
                Keys.deleted_attr: [],
                Keys.remaining_attr: [],
                Keys.series_filtering: {Keys.acc_score: [], Keys.auc_score: [], Keys.rank_correlation: [],
                                        Keys.removed_series_auc: [], Keys.removed_series_corr: [],
                                        Keys.series_filter: []}}

    return metadata


def get_correct_formats(X: Union[pd.DataFrame, Dict[Union[str, int], Collection]], y: pd.Series = None,
                        views: List[Union[str, int]] = None, add_tags=lambda x: x, reset_index=False) \
        -> (pd.DataFrame, Dict[Union[str, int], Collection]):
    """
    Converts `X` to the TSFuse format and the pandas multiindex format.

    Parameters
    ----------
    X: Union[pd.DataFrame, Dict[Union[str, int], Collection]]
        The MultiIndex dataframe (Pandas MultiIndex format) or dictionary of Collections (TSFuse format) that will
        be converted to the TSFuse format and the Pandas MultiIndex format.
    y: [Optional] pd.Series
        The index of `y` will be reset to match the index of `X` in Pandas MultiIndex format if the index of `X` will
        be reset (reset_index is True).
    views: [Optional] list of integers or strings (default None)
        the different views of the TSFuse format
    add_tags: [Optional] a function (default the identity function)
        a function that adds the necessary tags to the dictionary of Collections (TSFuse format)
    reset_index: [Optional] bool
        Determines whether the first level of the MultiIndex Dataframe will be reset

    Return
    ------
    X_pd: `X` in Pandas MultiIndex format
    X_tsfuse: `X` in TSFuse format
    """
    from tsfuse.data import pd_multiindex_to_dict_collection, dict_collection_to_pd_multiindex
    if isinstance(X, pd.DataFrame):
        if reset_index:
            (_, t) = X.index.levshape
            n = len(X.index) // t
            index = pd.MultiIndex.from_product([range(n), range(t)], names=["Instance", "Time"])
            X_pd = pd.DataFrame(X.values, index=index, columns=X.columns)
        else:
            X_pd = X
        X_tsfuse = pd_multiindex_to_dict_collection(X, add_tags=add_tags, views=views)
    else:
        X_pd = dict_collection_to_pd_multiindex(X)
        X_tsfuse = X
    if reset_index and y is not None:
        y.reset_index(inplace=True, drop=True)
    return X_pd, X_tsfuse


def get_correct_formats_numpy(X: Union[pd.DataFrame, Dict[Union[str, int], Collection]], y: pd.Series = None,
                              views: List[Union[str, int]] = None, add_tags=lambda x: x) \
        -> (np.ndarray, Dict[Union[str, int], Collection]):
    """
    Converts `X` to the TSFuse format and the numpy 3D array format.

    Parameters
    ----------
    X: Union[pd.DataFrame, Dict[Union[str, int], Collection]]
        The MultiIndex dataframe (Pandas MultiIndex format) or dictionary of Collections (TSFuse format) that will
        be converted to the TSFuse format and the numpy 3D array format.
    y: [Optional] pd.Series
        The index of `y` will be reset.
    views: [Optional] list of integers or strings (default None)
        the different views of the TSFuse format
    add_tags: [Optional] a function (default the identity function)
        a function that adds the necessary tags to the dictionary of Collections (TSFuse format)

    Return
    ------
    X_np: `X` in numpy 3D array format
    X_tsfuse: `X` in TSFuse format
    """
    from tsfuse.data import pd_multiindex_to_dict_collection, dict_collection_to_numpy3d
    if isinstance(X, pd.DataFrame):
        X_np = from_multi_index_to_3d_numpy(X)
        X_tsfuse = pd_multiindex_to_dict_collection(X, add_tags=add_tags, views=views)
    else:
        X_np = dict_collection_to_numpy3d(X)
        X_tsfuse = X
    if y is not None:
        y.reset_index(inplace=True, drop=True)
    return X_np, X_tsfuse


def get_tsfuse_format(X: Union[pd.DataFrame, Dict[Union[str, int], Collection]], views: List[Union[str, int]] = None,
                      add_tags=lambda x: x) -> Dict[Union[str, int], Collection]:
    """
    Converts `X` to the TSFuse format.

    Parameters
    ----------
    X: Union[pd.DataFrame, Dict[Union[str, int], Collection]]
        The MultiIndex dataframe (Pandas MultiIndex format) or dictionary of Collections (TSFuse format) that will
        be converted to the TSFuse format. If it is already in the TSFuse format, it is returned unchanged.
    views: [Optional] list of integers or strings (default None)
        the different views of the TSFuse format
    add_tags: [Optional] a function (default the identity function)
        a function that adds the necessary tags to the dictionary of Collections (TSFuse format)

    Return
    ------
    X_tsfuse: Dict[Union[str, int], Collection]
        `X` in TSFuse format
    """
    from tsfuse.data import pd_multiindex_to_dict_collection
    if isinstance(X, pd.DataFrame):
        if views is None:
            views = X.columns
        X_tsfuse = pd_multiindex_to_dict_collection(X, add_tags=add_tags, views=views)
    else:
        X_tsfuse = X
    return X_tsfuse


def get_tsfuse_format_np(X: Union[np.ndarray, Dict[Union[str, int], Collection]]) -> Dict[Union[str, int], Collection]:
    """
    Converts `X` to the TSFuse format.

    Parameters
    ----------
    X: Union[pd.DataFrame, Dict[Union[str, int], Collection]]
        The MultiIndex dataframe (Pandas MultiIndex format) or dictionary of Collections (TSFuse format) that will
        be converted to the TSFuse format. If it is already in the TSFuse format, it is returned unchanged.

    Return
    ------
    X_tsfuse: Dict[Union[str, int], Collection]
        `X` in TSFuse format
    """
    from tsfuse.data import numpy3d_to_dict_collection
    if isinstance(X, np.ndarray):
        views = list(range(X.shape[1]))
        views_ext = {v: [v] for v in views}
        X_tsfuse = numpy3d_to_dict_collection(X, views_ext=views_ext)
    else:
        X_tsfuse = X
    return X_tsfuse


def get_multiindex_pd_format(X: Union[pd.DataFrame, Dict[Union[str, int], Collection]], y: pd.Series = None,
                             reset_index=False) -> pd.DataFrame:
    """
    Converts `X` to the TSFuse format and the pandas multiindex format.

    Parameters
    ----------
    X: Union[pd.DataFrame, Dict[Union[str, int], Collection]]
        The MultiIndex dataframe (Pandas MultiIndex format) or dictionary of Collections (TSFuse format) that will
        be converted to the TSFuse format and the Pandas MultiIndex format.
    y: [Optional] pd.Series
        The index of `y` will be reset to match the index of `X` in Pandas MultiIndex format if the index of `X` will
        be reset (reset_index is True).
    reset_index: [Optional] bool
        Determines whether the first level of the MultiIndex Dataframe will be reset

    Return
    ------
    X_pd: `X` in Pandas MultiIndex format
    """
    from tsfuse.data import dict_collection_to_pd_multiindex
    if isinstance(X, pd.DataFrame):
        if reset_index:
            (_, t) = X.index.levshape
            n = len(X.index) // t
            index = pd.MultiIndex.from_product([range(n), range(t)], names=["Instance", "Time"])
            X_pd = pd.DataFrame(X.values, index=index, columns=X.columns)
        else:
            X_pd = X
    else:
        X_pd = dict_collection_to_pd_multiindex(X)
    if reset_index and y is not None:
        y.reset_index(inplace=True, drop=True)
    return X_pd


def rename_columns_pd(X: pd.DataFrame, translation: dict) -> None:
    """
    Renames the columns of `X` according to the translation dictionary.

    Parameters
    ----------
    X: pd.DataFrame
        The dataframe whose columns will be renamed.
    translation: dict
        The translation dictionary. The keys are the old column names and the values are the new column names.

    Return
    ------
    None, the columns of `X` are renamed in place.
    """
    X.rename(columns={str(n1): str(n2) for n1, n2 in translation.items()}, inplace=True)


def rename_keys_dict(X: dict, translation: dict) -> dict:
    """
    Renames the keys of `X` according to the translation dictionary.

    Parameters
    ----------
    X: dict
        The dictionary whose keys will be renamed.
    translation: dict
        The translation dictionary. The keys are the old column names and the values are the new column names.

    Return
    ------
    dict
        The dictionary with the keys renamed.
    """
    result = {}
    for key, value in X.items():
        if key in translation:
            result[translation[key]] = value
        else:
            result[key] = value
    return result


def reset_first_level_index(X: pd.DataFrame, y: pd.Series = None) -> (pd.DataFrame, pd.Series):
    """
    Resets the first level of the index of `X` and `y`, if given, to 0, 1, 2, ... . The second level of the index of `X`
    is the time index and is not changed.

    Parameters
    ----------
    X: pd.DataFrame
        The dataframe whose first level of the index will be reset.
    y: [Optional] pd.Series
        The series whose index will be reset, if given.

    Return
    ------
    X: pd.DataFrame
        The dataframe with the first level of the index reset.
    y: pd.Series
        The series with the index reset, if given. Otherwise, None is returned.
    """
    indices = list(range(len(X.index.get_level_values(0).unique())))
    timepoints = list(range(len(X.index.get_level_values(1).unique())))
    X.index = pd.MultiIndex.from_product([indices, timepoints], names=['Instance', 'Time'])
    if y is not None:
        y.index = indices
    return X, y


def create_multiindex(lst: list, n: int):
    """
    Creates a multiindex with the first level being the elements of `lst` and the second level being the n timepoints.

    Parameters
    ----------
    lst: list
        The elements of the first level of the multiindex
    n: int
        The number of timepoints in the second level of the multiindex

    Return
    ------
    multiindex: pd.MultiIndex
        The multiindex with the first level being the elements of `lst` and the second level being the n timepoints.
    """
    timepoints = list(range(n))
    return pd.MultiIndex.from_product([lst, timepoints], names=['Instance', 'Time'])


def catch22_features_numpy(x: np.ndarray, catch24=False):
    """
    Calculates the catch22 features for each row in `x`. The catch22 features are calculated using the pycatch22
    library.

    Parameters
    ----------
    x: np.ndarray
        The numpy array whose rows will be used to calculate the catch22 features.
    catch24: bool, optional, default False
        Whether to calculate the catch24 features instead of the catch22 features. The catch24 features are a superset
        of the catch22 features that additionally includes the mean and standard deviation of the time series.
    """
    features_dicts = [pycatch22.catch22_all(row, catch24=catch24) for row in x]
    result_list = [d['values'] for d in features_dicts]
    return np.array(result_list)


def multiindex_to_singleindex(X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert a MultiIndex DataFrame with one column or Series to a single index DataFrame. The second dimension of the
    index becomes the column. E.g. if the DataFrame has an index with shape (4,5) and 1 column, the returned DataFrame
    will have shape (4,5), i.e. 4 rows and 5 columns.

    Parameters
    ----------
    X: pd.Series or pd.DataFrame
        a MultiIndex DataFrame with one column or Series that has to be converted

    Returns
    -------
    A single index DataFrame were the rows represent the first level of the index of the given `X` and the columns
    represent the second level of the index of `X`.
    """
    if len(X.index.levels) < 2:
        raise Exception(f"A MultiIndex DataFrame is expected, but the index of the given DataFrame only has "
                        f"{X.index.levels} levels")
    if isinstance(X, pd.DataFrame) and len(X.columns) > 1:
        raise Exception(f"A single column MultiIndex DataFrame is expected, but the given DataFrame has "
                        f"{len(X.columns)} columns.")

    x1 = X.reset_index()
    return x1.pivot(index=x1.columns[0], columns=x1.columns[1])


def replace_nans_by_col_mean(features):
    col_mean = np.nanmean(features, axis=0)
    inds = np.where(np.isnan(features))
    features[inds] = np.take(col_mean, inds[1])