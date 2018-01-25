from sklearn.base import TransformerMixin
from pandas.core.frame import DataFrame

from healthcareai.common.dataframe_filters import filters_helpers as filtHelp


class DataframeColumnSuffixFilter(TransformerMixin):
    """Given a pandas dataframe, remove columns with suffix 'DTS'."""

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        filtHelp.validate_dataframe_input(x)

        # Build a list that contains column names that do not end in 'DTS'
        filtered_column_names = [column for column in x.columns if not column.endswith('DTS')]

        # Select all data excluding datetime columns
        return x[filtered_column_names]
