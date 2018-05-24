import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

class DataFrameFeatureScaling(TransformerMixin):
    """Scales numeric features. Columns that are numerics are scaled, or otherwise specified."""

    def __init__(self, columns_to_scale=None, reuse=None):
        self.columns_to_scale = columns_to_scale
        self.reuse = reuse

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Check if it's reuse, if so, then use the reuse's DataFrameFeatureScaling
        if self.reuse:
            return self.reuse.fit_transform(X, y)

        # Check if we know what columns to scale, if not, then get all the numeric columns' names
        if not self.columns_to_scale:
            self.columns_to_scale = list(X.select_dtypes(include=[np.number]).columns)

        X[self.columns_to_scale] = StandardScaler().fit_transform(X[self.columns_to_scale])

        return X