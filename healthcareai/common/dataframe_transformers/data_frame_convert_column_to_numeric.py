import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

class DataFrameConvertColumnToNumeric(TransformerMixin):
    """Convert a column into numeric variables."""

    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        X[self.column_name] = pd.to_numeric(arg=X[self.column_name], errors='raise')

        return X
