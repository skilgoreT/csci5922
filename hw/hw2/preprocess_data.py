# %load preprocess_data.py


from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import re
room_data.describe()

#attributes = [attr for attr in list(data) if not re.search(attr, r'date|Occupancy')]
# Create a class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

# list of attributes for the DataFrameSelector (pandas to numpy)
room_attrib = [attr for attr in list(room_data) if not re.search(attr, r'date|Occupancy')]
print(room_attrib)
pipeline = Pipeline([
    ('selector', DataFrameSelector(room_attrib)),
    ('std_scaler', StandardScaler()),
])

# axis=1 implies column
room_prepared = pipeline.fit_transform(room_data)
print(room_prepared)