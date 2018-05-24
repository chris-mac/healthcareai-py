import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

class DataFrameCreateDummyVariables(TransformerMixin):
    """Convert all categorical columns into dummy/indicator variables. Exclude given columns."""

    def __init__(self, excluded_columns=None):
        self.excluded_columns = excluded_columns

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        columns_to_dummify = X.select_dtypes(include=[object, 'category'])

        # remove excluded columns (if they are still in the list)
        for column in columns_to_dummify:
            if column in self.excluded_columns:
                columns_to_dummify.remove(column)

        # Create dummy variables
        X = pd.get_dummies(X, columns=columns_to_dummify, drop_first=True, prefix_sep='.')

        return X
