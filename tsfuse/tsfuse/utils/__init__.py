from typing import Union

import numpy as np
import pandas as pd


def encode_onehot(y: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Encodes the given labels using one-hot encoding.
    """
    classes = sorted(np.unique(y))
    y_onehot = np.zeros((len(y), len(classes)))
    for i, c in enumerate(classes):
        y_onehot[y == c, i] = 1
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y_onehot = pd.DataFrame(data=y_onehot, index=y.index, columns=classes)
    return y_onehot
