import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

class DataFrameUnderSampling(TransformerMixin):
    """
    Performs undersampling on a dataframe.
    
    Must be done BEFORE train/test split so that when we split the under/over sampled dataset.

    Must be done AFTER imputation, since under/over sampling will not work with missing values (imblearn requires target
     column to be converted to numerical values)
    """

    def __init__(self, predicted_column, random_seed=0):
        self.random_seed = random_seed
        self.predicted_column = predicted_column

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        # TODO how do we validate this happens before train/test split? Or do we need to? Can we implement it in the
        # TODO      simple trainer in the correct order and leave this to advanced users?

        # Extract predicted column
        y = np.squeeze(X[[self.predicted_column]])

        # Copy the dataframe without the predicted column
        temp_dataframe = X.drop([self.predicted_column], axis=1)

        # Initialize and fit the under sampler
        under_sampler = RandomUnderSampler(random_state=self.random_seed)
        x_under_sampled, y_under_sampled = under_sampler.fit_sample(temp_dataframe, y)

        # Build the resulting under sampled dataframe
        result = pd.DataFrame(x_under_sampled)

        # Restore the column names
        result.columns = temp_dataframe.columns

        # Restore the y values
        y_under_sampled = pd.Series(y_under_sampled)
        result[self.predicted_column] = y_under_sampled

        return result