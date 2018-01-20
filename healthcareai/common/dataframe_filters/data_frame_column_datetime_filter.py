from sklearn.base import TransformerMixin
from pandas.core.frame import DataFrame


class DataFrameColumnDateTimeFilter(TransformerMixin):
    """Given a pandas dataframe, remove any columns that has the type datetime."""

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        validate_dataframe_input(x)

        # Select all data excluding datetime columns
        return x.select_dtypes(exclude=["datetime"])
