#%load read_data.py
import pandas as pd
import os
def load_data(filename):
    csv_path = os.path.join(".", filename)
    return pd.read_csv(csv_path)

room_data = load_data('train_data.txt')
room_data.head()
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
print(room_prepared)# %load train.py
import os as os
import numpy as np
import scipy.io
import scipy.optimize as optimization

N_EPOCH = 1

n_mis = []
def dwp(w,d,X):
    # using numpy einstein summation to vectorize the computation
    # calculating w.x.d -- w.x.d < 0 => incorrect classification
    C = d*np.einsum('i,ji->j', w, X) # shape is (100,)
    # use enumerate for the equivalent to each_with_index (Ruby)
    xd = [ d[i]*X[i] for i,c in enumerate(C) if c <= 0]
    # the error is the number of misclassified teachers
    E = len(xd)/float(len(d))
    # sum xd element-wise
    # dw is x.t summed over misclassified teachers
    dw =  np.einsum('ij->j',xd) if(len(xd) > 0) else 0
    return dw, E

def train():
    # data
    room_with_bias = np.full((len(room_prepared),6),1.0)
    room_with_bias[:,1:] = room_prepared
    # shape is (nrow, 6)
    X = room_with_bias
    print(np.shape(X))
    # use domain -1,1 for teachers so we can use x.w.d<0 for the classification test
    # shpae is (nrow,1)
    t = [z if z == 1 else -1 for z in room_data['Occupancy']]
    print(np.shape(t))
    # starting vector of weights
    w = np.zeros((np.shape(room_with_bias)[1]))

    # sweep through the data N_EPOCH times
    for _ in range(N_EPOCH):
         dw, E = dwp(w,t,X)
         w += dw
         print(E)
    return w

# invoke main
w_perceptron = train()
print(w_perceptron)