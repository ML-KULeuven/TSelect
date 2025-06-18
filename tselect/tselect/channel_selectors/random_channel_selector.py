import random

from sklearn.base import TransformerMixin
from tselect import SEED, Keys
from tsfuse.data import from_dict_collection_to_nested, Collection


class RandomChannelSelector(TransformerMixin):
    def __init__(self, seed=SEED):
        self.seed = seed
        self.nb_channels = None
        self.selected_channels = None

    def transform(self, X):
        return X[self.selected_channels]

    def fit(self, X, y=None, metadata=None):
        """
        Perform the channel selection by (1) randomly selecting the number of channels to keep (from 1 to all channels),
         and (2) randomly selecting which channels to keep.
        Parameters
        ----------
        X: Union[pd.DataFrame, Dict[Union[str, int], Collection]]
            The data to fit in the MultiIndex Pandas format ((n*t), d) or the TSFuse format.
        y: pd.Series
            The target variable.
        metadata
            The metadata dictionary to update with the results of the channel selector.

        Returns
        -------
        None, but the selected channels are stored in the selected_channels attribute.

        """
        random.seed(self.seed)
        if isinstance(X, dict):
            X = from_dict_collection_to_nested(X)
        self.nb_channels = random.randint(1, X.shape[1])
        self.selected_channels = random.sample(X.columns.to_list(), self.nb_channels)
        self.update_metadata(metadata)

        return None

    def update_metadata(self, metadata):
        """
        Update the metadata with the results of the filter.
        """
        if metadata:
            metadata[Keys.series_filtering][Keys.series_filter].append(self)
