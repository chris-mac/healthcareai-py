import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

class DataFrameImputer(TransformerMixin):
    """
    Impute missing values in a dataframe.

    Columns of dtype object or category (assumed categorical) are imputed with the mode (most frequent value in column).

    Columns of other types (assumed continuous) are imputed with mean of column.
    """

    def __init__(self, impute=True, verbose=True):
        self.impute = impute
        self.object_columns = None
        self.fill = None
        self.verbose = verbose

    def fit(self, X, y=None):
        # Return if not imputing
        if self.impute is False:
            return self

        # Grab list of object column names before doing imputation
        self.object_columns = X.select_dtypes(include=['object']).columns.values

        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O')
                                  or pd.core.common.is_categorical_dtype(X[c])
                               else X[c].mean() for c in X], index=X.columns)

        if self.verbose:
            num_nans = sum(X.select_dtypes(include=[np.number]).isnull().sum())
            num_total = sum(X.select_dtypes(include=[np.number]).count())
            percentage_imputed = num_nans / num_total * 100
            print("Percentage Imputed: %.2f%%" % percentage_imputed)
            print("Note: Impute will always happen on prediction dataframe, otherwise rows are dropped, and will lead "
                  "to missing predictions")

        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        # Return if not imputing
        if self.impute is False:
            return X

        result = X.fillna(self.fill)

        for i in self.object_columns:
            if result[i].dtype not in ['object', 'category']:
                result[i] = result[i].astype('object')

        return result
