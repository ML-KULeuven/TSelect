import pandas as pd
import pycatch22
from sklearn.base import TransformerMixin

from tselect.abstract_extractor import AbstractExtractor


class Catch22Extractor(AbstractExtractor, TransformerMixin):
    def transform_model(self, X: pd.DataFrame):
        """
        Transform the data by extracting features from it using Catch22. The average and standard deviation are also
        included as features.

        Parameters
        ----------
        X : pd.DataFrame
            The data to transform.

        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        f = pd.DataFrame(index=X.index.levels[0])
        for d in X.columns:
            for i in X.index.levels[0]:
                ts = X.loc[i, d]
                result = pycatch22.catch22_all(ts, catch24=True)
                for j, name in enumerate(result['names']):
                    f.loc[i, d + '|' + name] = result['values'][j]
        return f

    def fit_model(self, X: pd.DataFrame, y):
        """
        No model should be fitted for Catch22.
        """
        return None
