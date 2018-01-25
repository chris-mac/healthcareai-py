from sklearn.base import TransformerMixin
from pandas.core.frame import DataFrame

from healthcareai.common.dataframe_filters import filters_helpers as filtHelp


class DataframeColumnRemover(TransformerMixin):
    """Given a pandas dataframe, remove the given column or columns in list form."""

    def __init__(self, columns_to_remove):
        self.columns_to_remove = columns_to_remove

    def fit(self, x, y=None):
        return self

    def transform(self, X, y=None):
        filtHelp.validate_dataframe_input(X)
        if self.columns_to_remove is None:
            # if there is no grain column, for example
            return X

        # Build a list of all columns except for the grain column'
        filtered_column_names = [c for c in X.columns if c not in self.columns_to_remove]

        # return the filtered dataframe
        return X[filtered_column_names]

