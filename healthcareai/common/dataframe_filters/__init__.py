"""Filters

This module contains filters for preprocessing data. Most operate on DataFrames and are named appropriately.
"""

from .data_frame_column_suffix_filter import DataframeColumnSuffixFilter
from .data_frame_column_datetime_filter import DataFrameColumnDateTimeFilter
from .data_frame_column_remover import DataframeColumnRemover
from .data_frame_null_value_filter import DataframeNullValueFilter

from healthcareai.common.healthcareai_error import HealthcareAIError

def validate_dataframe_input(possible_dataframe):
    """Validate that input is a pandas dataframe and raise an error if it is not. Stays silent if it is."""
    if is_dataframe(possible_dataframe) is False:
        raise HealthcareAIError(
            'This transformer requires a pandas dataframe and you passed in a {}'.format(type(possible_dataframe)))


def is_dataframe(possible_dataframe):
    """Simple helper that returns True if an input is a pandas dataframe."""
    return issubclass(DataFrame, type(possible_dataframe))



__all__ = [
    'DataframeColumnSuffixFilter',
    'DataFrameColumnDateTimeFilter',
    'DataFrameColumnRemover',
    'DataFrameNullValueFilter'
]
