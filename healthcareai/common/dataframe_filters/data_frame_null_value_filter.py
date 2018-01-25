from sklearn.base import TransformerMixin
from pandas.core.frame import DataFrame

from healthcareai.common.dataframe_filters import filters_helpers as filtHelp

from healthcareai.common.healthcareai_error import HealthcareAIError

class DataframeNullValueFilter(TransformerMixin):
    """Given a pandas dataframe, remove rows that contain null values in any column except the excluded."""

    def __init__(self, excluded_columns=None):
        # TODO validate excluded column is a list
        self.excluded_columns = excluded_columns or []

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        filtHelp.validate_dataframe_input(x)

        subset = [c for c in x.columns if c not in self.excluded_columns]

        x.dropna(axis=0, how='any', inplace=True, subset=subset)

        if x.empty:
            raise HealthcareAIError(
                "Because imputation is set to False, rows with missing or null/NaN values are being dropped. "
                "In this case, all rows contain null values and therefore were ALL dropped. "
                "Please consider using imputation or assessing the data quality and availability")

        return x
