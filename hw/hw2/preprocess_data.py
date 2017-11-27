# %load preprocess_data.py

from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
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
pipeline = Pipeline([
    ('selector', DataFrameSelector(room_attrib)),
    ('std_scaler', StandardScaler()),
])

room_prepared = pipeline.fit_transform(room_data)
#print(np.shape(room_prepared))
#print(room_prepared[0])
df = pd.DataFrame(room_prepared)
# shuffle the normalized data with stable seed (repoducable debug)
# df = df.sample(frac=1, random_state=0).reset_index(drop=True)
print(df.head(5))
df.plot(subplots=True, layout=(2,3), figsize=(12, 8));
