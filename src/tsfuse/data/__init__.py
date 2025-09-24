from enum import Enum
from typing import Dict, Union, List

import numpy as np
import pandas as pd
from sktime.datatypes._panel._convert import from_3d_numpy_to_multi_index, from_multi_index_to_3d_numpy

from .tags import Tags, TagKey, HierarchicalTagKey
from .units import units

__all__ = [
    'Collection',
    'Type',
    'dict_collection_to_pd_multiindex',
    'dict_collection_to_numpy3d',
    'pd_multiindex_to_dict_collection',
    'numpy3d_to_dict_collection'
]


class Collection(object):
    """
    Data collection for representing time series and attributes.

    Parameters
    ----------

    values : array-like
        Three-dimensional array of shape ``(N, t, d)`` where:

        - `N` is the number of windows
        - `t` is the number of time stamps in each window
        - `d` is the number of dimensions.

        Alternatively, the array can be in Numpy3D format, in which the shape of the array is (N, d, t). If so,
        please specify this in the `from_numpy3d` argument.

    index : array-like, optional
        Two-dimensional array of shape ``(N, t)`` with the time stamps of each window.

    dimensions : array-like, optional
        One-dimensional array of length ``d`` with the dimension names.

    mask_value : optional
        Placeholder representing missing values.
        Can be used for representing variable-length time series.
        Default: ``nan``

    unit : optional
        Unit, see the `Pint documentation <https://pint.readthedocs.io>`_.

    tags : optional
        :class:`~tsfuse.data.tags.Tags` object which specifies values for one or more tag keys.

    from_numpy3d: optional, default=False
        Specifies whether `values` is in Numpy3D format. This is the case when the shape of `values` is (N, d, t).
        If True, the array of `values` will first be converted to an array of shape (N, t, d).
    """

    def __init__(self, values, index=None, dimensions=None,
                 mask_value=np.nan, unit=None, tags=None, from_numpy3d=False):
        if from_numpy3d:
            values = Collection.numpy3d_to_collection(values)
        # Unit
        self._unit = unit
        # Tags
        self._tags = tags if tags is not None else Tags()
        # Data
        # TODO: Add checks for shapes
        if _collections(values):
            collections = []
            for i in range(len(values)):
                if values[i].__class__.__name__ == 'Collection':
                    collections.append(values[i])
                else:
                    collection_index = index[i] if index is not None else None
                    collections.append(Collection(
                        values[i],
                        index=collection_index,
                        dimensions=dimensions,
                        mask_value=mask_value,
                        unit=self._unit,
                        tags=self._tags
                    ))
            self._values = np.array(collections)
            if index is None:
                self._index = np.arange(np.sum([c.shape[0] for c in collections]))
            else:
                self._index = index
            if dimensions is None:
                if len(set(tuple(c.dimensions) for c in collections)) == 1:
                    self._dimensions = collections[0].dimensions
                else:
                    self._dimensions = np.arange(collections[0].shape[2])
            else:
                self._dimensions = np.array(dimensions, copy=False)
        else:
            values = np.array(values, copy=False)
            index = np.array(index, copy=False) if index is not None else None
            dimensions = np.array(dimensions, copy=False) if dimensions is not None else None
            values, index, dimensions = _reshape(values, index, dimensions)
            # Determine data type of values
            self._dtype = values.dtype
            if np.issubdtype(self._dtype, np.number):
                self._dtype = np.float64
            elif np.issubdtype(self._dtype, np.character):
                self._dtype = np.str_
            elif np.issubdtype(self._dtype, np.bool_):
                self._dtype = np.bool_
            else:
                self._dtype = object
            # Determine data type of index
            self._itype = index.dtype
            if np.issubdtype(self._itype, np.datetime64):
                self._itype = np.dtype('datetime64[ns]')
            elif np.issubdtype(self._itype, np.integer):
                self._itype = np.int64
            else:
                self._itype = object
            # Determine type of dimensions
            self._ntype = dimensions.dtype
            if np.issubdtype(self._ntype, np.character):
                self._ntype = np.str_
            elif np.issubdtype(self._ntype, np.integer):
                self._ntype = np.int64
            else:
                self._ntype = object
            # Values
            self._values = np.empty(values.shape, dtype=self._dtype)
            self._values[:, :, :] = values
            # Index and dimensions
            self._index = np.array(index, copy=True, dtype=self._itype)
            self._dimensions = np.array(dimensions, copy=True, dtype=self._ntype)
        # Mask value
        self._mask_value = mask_value

    @property
    def values(self):
        """
        numpy.array : Three-dimensional array of shape ``(N, t, d)``
        """
        return self._values

    @property
    def index(self):
        """
        numpy.array : Two-dimensional array of shape ``(N, t)``
        """
        return self._index

    @property
    def dimensions(self):
        """
        numpy.array : One-dimensional array of length ``d``
        """
        return self._dimensions

    @property
    def tags(self):
        """
        :class:`~tsfuse.data.tags.Tags` object which specifies values for one or more tag keys.
        """
        return self._tags

    @property
    def unit(self):
        """
        pint.unit.Unit : Unit of ``self.values``
        """
        return self._unit

    @property
    def type(self):
        """
        tsfuse.data.Type : Type of data.
        """
        if self.shape[0] > 1:
            return Type.WINDOWS
        elif np.max(self.shape[1]) > 1:
            return Type.SERIES
        elif self.shape[2] > 1:
            return Type.ATTRIBUTES
        else:
            return Type.SCALAR

    @property
    def transform_axis(self):
        if (self.type == Type.WINDOWS) or (self.type == Type.SERIES):
            return 'timestamps'
        else:
            return 'dimensions'

    @property
    def mask_value(self):
        """
        Mask value.
        """
        return self._mask_value

    @property
    def shape(self):
        """
        Shape ``(N, t, d)``
        """
        if np.isscalar(self.values):
            return ()
        elif len(self.values.shape) == 1:
            return (
                np.sum([c.shape[0] for c in self.values]),
                tuple([c.shape[1] for c in self.values]),
                len(self.dimensions)
            )
        else:
            return self.values.shape

    @property
    def loc(self):
        return NotImplementedError()

    @property
    def iloc(self):
        return IndexLocationIndexer(self)

    @property
    def dtype(self):
        """
        Data type of ``self.values``
        """
        if (self.values is not None) and (len(self.values.shape) == 3):
            return np.array(self.values, copy=False).dtype
        else:
            return None

    @property
    def itype(self):
        """
        Data type of ``self.index``
        """
        if (self.values is not None) and (len(self.index.shape) == 2):
            return self.index.dtype
        else:
            return None

    def append(self, other):
        values = np.concatenate((self.values, other.values), axis=0)
        index = np.concatenate((self.index, other.index), axis=0)
        dimensions = self.dimensions
        m = self.mask_value
        return Collection(values, dimensions=dimensions, index=index, mask_value=m)

    def __len__(self):
        return self.shape[0]

    def to_numpy3d(self) -> np.ndarray:
        """
        Convert the time series data in this collection to the format used by numpy3D.

        Parameters
        ----------
        self : Collection
            The instance of the Collection that the method is being called on.

        Returns
        -------
        result : ndarray
            An ndarray of shape (N, d, t) containing the time series data in the numpy3D format.

        Notes
        -----
        - The original time series data is stored in the `values` attribute of the object, which has a shape of
        (N, t, d).
        - The resulting ndarray is obtained by transposing each time series in the original data, such that the
        resulting array has shape (N, d, t).
        """
        (n, t, d) = self.values.shape
        result = np.empty(shape=(n, d, t), dtype=self.dtype)
        for i, ts in enumerate(self.values):  # ts if of dimension (t, d)
            result[i] = np.transpose(ts)
        return result

    def to_pd_multiindex(self, key: str) -> pd.DataFrame:
        """
        Converts the time series data stored in the `values` attribute of this Collection to a Pandas DataFrame with
        a multi-index and renames the columns using the given `key` and the name of the corresponding dimension of this
        Collection. The multi-index has shape (N, t) where the first axis represents the instances and the second axis
        the timepoints.

        Parameters
        ----------
        key : str
            The key to use for renaming the columns in the resulting DataFrame.

        Returns
        -------
        df : DataFrame
            A Pandas DataFrame with a multi-index and renamed columns.

        Notes
        -----
        - The columns of the DataFrame are renamed as follows if there are multiple dimensions:
        [key + "_" + str(i) for i in self.dimensions], where `self.dimensions` is a list of the dimensions of this
        Collection. If there is only 1 dimension, the column will be named after `key`.
        """
        if len(self.dimensions) > 1:
            column_names = [str(key) + "_" + str(i) for i in self.dimensions]
        else:
            column_names = [str(key)]
        return from_3d_numpy_to_multi_index(self.to_numpy3d(),
                                            column_names=column_names)

    @staticmethod
    def numpy3d_to_collection(values: np.ndarray) -> np.ndarray:
        """
        Converts the time series data in the numpy3D format to the format used by the Collection class.

        Parameters
        ----------
        values : ndarray
            An ndarray of shape (N, d, t) containing the time series data in the numpy3D format.

        Returns
        -------
        result : ndarray
            An ndarray of shape (N, t, d) containing the time series data in the format used by the Collection class.
        """
        (n, d, t) = values.shape
        result = np.empty(shape=(n, t, d), dtype=values.dtype)
        for i, ts in enumerate(values):  # ts has dimension (d, t)
            result[i] = np.transpose(ts)
        return result

    def scale(self, scaler):
        """
        Scale the values of this collection using the given scaler.

        Parameters
        ----------
        scaler : callable
            A callable that takes a numpy array as input and returns a numpy array as output.

        Returns
        -------
        None
            The values are scaled in-place.
        """
        scaled_values = scaler.fit_transform(self.values)
        self._values = scaled_values


def plot(X, i=0):
    """Plot time series collections.

    Parameters
    ----------
    X : dict
        Time series collections.
    i : int, optional
        Select window ``i`` using ``.values[i, :, :]``. Default: 0
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 2 * len(X)))
    for v, name in enumerate(X):
        plt.subplot(len(X), 1, v + 1)
        collection = X[name]
        for j in range(collection.shape[2]):
            # TODO: Fix index of Collection
            plt.plot(X[name].index[i, :], X[name].values[i, :, j].flatten())
        plt.ylabel(name)
    plt.show()


class IndexLocationIndexer(object):
    def __init__(self, collection):
        self.collection = collection

    def __getitem__(self, item):
        values = self.collection.values[item]
        dimensions = self.collection.dimensions[item[-1]]
        index = self.collection.index[item[:-1]]
        m = self.collection.mask_value
        return Collection(values, dimensions=dimensions, index=index, mask_value=m)


class Type(Enum):
    """
    Type of data.
    """

    SCALAR = 0
    """
    One value (0-dimensional)
    """

    ATTRIBUTES = 1
    """
    Vector of attributes (1-dimensional)
    """

    SERIES = 2
    """
    Vector of one or more time series (2-dimensional)
    """

    WINDOWS = 3
    """
    Windows, each with one or more time series (3-dimensional)
    """


def _reshape(values, index, dimensions):
    if len(values.shape) == 0:
        v = np.empty((1, 1, 1), dtype=values.dtype)
        v[0, 0, 0] = values
        if dimensions is None:
            n = np.zeros((1,))
        else:
            n = np.empty((1,))
            n[0] = dimensions
        if index is None:
            i = np.zeros((1, 1))
        else:
            i = np.empty((1, 1))
            i[0, 0] = index
    elif len(values.shape) == 1:
        v = np.empty((1, 1, values.shape[0]), dtype=values.dtype)
        v[0, 0, :] = values
        if dimensions is None:
            n = np.arange(v.shape[2])
        else:
            n = dimensions
        if index is None:
            i = np.zeros((1, 1))
        else:
            i = np.empty((1, 1))
            i[0, 0] = index
    elif len(values.shape) == 2:
        v = np.empty((1, values.shape[0], values.shape[1]), dtype=values.dtype)
        v[0, :, :] = values
        if dimensions is None:
            n = np.arange(v.shape[2])
        else:
            n = dimensions
        if index is None:
            i = np.empty((1, v.shape[1]))
            i[0, :] = np.arange(v.shape[1])
        else:
            i = np.empty((1, v.shape[1]), dtype=index.dtype)
            i[0, :] = index
    else:
        v = np.empty((values.shape[0], values.shape[1], values.shape[2]), dtype=values.dtype)
        v[:, :, :] = values
        if dimensions is None:
            n = np.arange(v.shape[2])
        else:
            n = dimensions
        if index is None:
            t = np.prod(v.shape[:2])
            i = np.empty((v.shape[0], v.shape[1]))
            i[:, :] = np.arange(t).reshape(*v.shape[:2][::-1]).T
        else:
            i = np.empty((v.shape[0], v.shape[1]), dtype=index.dtype)
            i[:, :] = index
    return v, i, n


def _collections(values):
    if isinstance(values, (int, float, str)):
        return False
    if len(values) == 0:
        return False
    if isinstance(values[0], (list, tuple, np.ndarray)) \
            and (len(set(len(values[i]) for i in range(len(values)))) > 1):
        return True
    if values[0].__class__.__name__ == 'Collection':
        return True
    return False


def dict_collection_to_numpy3d(X: Dict[Union[str, int], Collection]) -> np.ndarray:
    """
    Transforms the given data `X` to the numpy3D time series format. See examples/AA_datatypes_and_datasets.ipynb
    (Section 1.2.2)

    In the numpy3D time series format, axis 0 represents the instances, axis 1 the variables, and axis 2
    the timepoints.

    Parameters
    ----------
    X : Dict[Union[str, int], Collection]
        A dictionary where the keys are either strings or integers and the values are Collections of shape (N, t, d).

    Returns
    -------
    result : ndarray
        An ndarray of shape (N, D, t), where D is the sum over the dimensions of each collection, containing the
        time series data in the numpy3D time series format.

    Notes
    -----
    - The resulting ndarray is obtained by transposing the time series data for each key and concatenating them along
      the second dimension, such that the resulting array has shape (N, D, t), where D is the sum over the dimensions of
       each collection.
    """
    # n and t should be equal for all collections in the dictionary, but d can differ
    (n, t, _) = list(X.values())[0].shape
    d = []
    for c in X.values():
        d.append(c.shape[2])

    result = np.empty(shape=(n, sum(d), t), dtype=list(X.values())[0].dtype)
    for i, (key, collection) in enumerate(X.items()):
        result[:, i * d[i]:(i + 1) * d[i], :] = collection.to_numpy3d()
    return result


def numpy3d_to_dict_collection(X: np.ndarray, views_ext: Dict[Union[str, int], List[Union[str, int]]],
                               add_tags=lambda x: x) -> Dict[Union[str, int], Collection]:
    """
    Transforms the given data `X` in the numpy3D time series format to a dictionary of Collections.

    Parameters
    ----------
    X : ndarray
        An ndarray of shape (N, D, t), where D is the number of dimensions, N is the number of instances and t is the
        number of timepoints.
    views_ext : Dict[Union[str, int], List[Union[str, int]]]
        A dictionary where the keys are either strings or integers, representing the names of the views, and the values
        are lists of strings or integers, representing the dimensions belonging to that view.
    add_tags: Callable, default=lambda x: x
        A function that adds tags to the data. This parameter is used to convert to the internal TSFuse Collection
        and adds tags to the Collection, describing what sensors were used for the different dimensions.

    Returns
    -------
    result : Dict[Union[str, int], Collection]
        A dictionary where the keys are either strings or integers, representing the names of the views, and the values
        are Collections of shape (N, t, d).
    """
    result = {}
    i = 0
    for view, dimensions in views_ext.items():
        d = len(dimensions)
        X_view = X[:, i: i + d]
        i += d
        result[view] = Collection(X_view, dimensions=dimensions, from_numpy3d=True)

    return add_tags(result)


def dict_collection_to_pd_multiindex(X: Dict[Union[str, int], Collection], index=None) -> pd.DataFrame:
    """
    Converts the time series data stored in the Collections in the given dictionary `X` to a Pandas DataFrame with a
    multi-index. The multi-index has shape (N, t) where the first axis represents the instances and the second axis the
    timepoints.

    The Collections are concatenated along the columns of the DataFrame.

    Parameters
    ----------
    X : Dict[Union[str, int], Collection]
        A dictionary where the keys are used as the keys for renaming the columns in the resulting DataFrame and the
        values are the collections to convert to the multi-index DataFrame.
    index : Union[None, pd.Index, pd.MultiIndex], optional
        The index to use for the resulting DataFrame. If None, a default index is used.

    Returns
    -------
    df : DataFrame
        A Pandas DataFrame with a multi-index and renamed columns.
    """
    from tselect.utils import create_multiindex

    data = []
    for i, (key, collection) in enumerate(X.items()):
        data.append(collection.to_pd_multiindex(key))
    result = pd.concat(data, axis=1)

    if index is not None:
        if isinstance(result.index, pd.MultiIndex) and not isinstance(index, pd.MultiIndex):
            result.index = create_multiindex(index, result.index.levels[1].shape[0])
        elif isinstance(result.index, pd.MultiIndex) and isinstance(index, pd.MultiIndex) and \
                result.index.shape != index.shape:
            result.index = create_multiindex(index.get_level_values(0).unique().to_list(),
                                             result.index.levels[1].shape[0])
        else:
            result.index = index
    return result


def pd_multiindex_to_dict_collection(X: pd.DataFrame, views: List[Union[str, int]] = None, add_tags=lambda x: x) \
        -> Dict[Union[str, int], Collection]:
    """
    Converts the time series data stored in the given Pandas DataFrame `X` with a multi-index to a dictionary of
    Collections. The multi-index has shape (N, t) where the first axis represents the instances and the second axis the
    timepoints. The columns of the DataFrame are assumed to be named after the views and represent the dimensions. For
    example, the columns "acc_x", "acc_y" and "acc_z" are assumed to represent the dimensions of the view "acc".

    Parameters
    ----------
    X : DataFrame
        A Pandas DataFrame with a multi-index
    views : List[Union[str, int]], optional
        A list of strings or integers representing the names of the views. If None, the names of the views are derived
        from the columns of the DataFrame by assuming that each column represents a different view.
    add_tags: Callable, default=lambda x: x
        A function that adds tags to the data. This parameter is used to convert to the internal TSFuse Collection
        and adds tags to the Collection, describing what sensors were used for the different dimensions.

    Returns
    -------
    result : Dict[Union[str, int], Collection]
        A dictionary where the keys are either strings or integers, representing the names of the views, and the values
        are Collections of shape (N, t, d).
    """
    arr = from_multi_index_to_3d_numpy(X)
    views = list(X.columns) if views is None else views
    views_ext = {}
    for view in views:
        if str(view).startswith('dim_'):
            views_ext[view] = [c for c in X.columns if str(c) == str(view)]
        else:
            views_ext[view] = [c for c in X.columns if str(c).startswith(str(view))]
    return numpy3d_to_dict_collection(arr, views_ext, add_tags=add_tags)


def from_dict_collection_to_nested(x: Dict[Union[str, int], Collection]) -> pd.DataFrame:
    max_t = get_max_timepoint(x)
    data = {}
    for key in x.keys():
        N, t, _ = x[key].values.shape
        values = []
        for timepoints in x[key].values.reshape(N, t):
            if t < max_t:
                timepoints = np.concatenate((timepoints, np.repeat(timepoints[-1], max_t - t)))
            values.append(pd.Series(timepoints))
        data[key] = values

    return pd.DataFrame(data)


def get_max_timepoint(x: Dict[Union[str, int], Collection]) -> int:
    max_t = -1
    for key in x.keys():
        if x[key].values.shape[1] > max_t:
            max_t = x[key].values.shape[1]
    return max_t
