#%%
import os
import numpy as np
import scipy.io
import scipy.optimize as optimization
#from IPython.core.debugger import set_trace

# Data
filedir = os.path.dirname(os.path.realpath('__file__'))
datapath = os.path.join(filedir,'hw/hw1/assign1_data.mat')
data = scipy.io.loadmat(datapath)

# fitter wants shape (k,M) (k number of predictors)
# data['x'] has shape (M,k) so we fix that
print(data['x'].shape)
x1 = data['x'][:,0]
x2 = data['x'][:,1]
X = np.array([x1,x2])
print(X.shape)
# argg - shape (100,1) - gives obstuse error in the fitter
# we fix that too (flatten)
y = np.array(data['y']).flatten()
print(y.shape)

# Initial guess
w0 = np.array([1, 1, 1])

# Objective fucntion
# y = w1 * x1 + w2 * x2 + b.
def func(X, w1, w2, b):
  # unpack independent vars
  rval = w1*X[0] + w2*X[1] + b
  return rval

result = optimization.curve_fit(func, X, y, w0)
print(result[0])


