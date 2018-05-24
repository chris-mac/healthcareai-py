import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

class DataFrameConvertTargetToBinary(TransformerMixin):
    # TODO Note that this makes healthcareai only handle N/Y in pred column
    """
    Convert classification model's predicted col to 0/1 (otherwise won't work with GridSearchCV). Passes through data
    for regression models unchanged. This is to simplify the data pipeline logic. (Though that may be a more appropriate
    place for the logic...)

    Note that this makes healthcareai only handle N/Y in pred column
    """

    def __init__(self, model_type, target_column):
        self.model_type = model_type
        self.target_column = target_column

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        # TODO: put try/catch here when type = class and predictor is numeric
        # TODO this makes healthcareai only handle N/Y in pred column
        if self.model_type == 'classification':
            # Turn off warning around replace
            pd.options.mode.chained_assignment = None  # default='warn'
            # Replace 'Y'/'N' with 1/0
            X[self.target_column].replace(['Y', 'N'], [1, 0], inplace=True)

        return X
