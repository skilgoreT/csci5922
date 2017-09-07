import os
import numpy as np
import scipy.io
import scipy.optimize as optimization
#from IPython.core.debugger import set_trace
#%%
EPS = 0.001
N_EPOCH = 1000
def dw(w,d,X):
    # using numpy einstein summation to vectorize the computation
    # vector of coefficients (d-w.x) for each pattern (alpha)
    C = d - np.einsum('i,ij->j', w, X) # shape is (100,)
    # C_i * X_ji or C * transpose(X)
    dw = EPS*np.einsum('i,ji->j', C, X) # shape is (3,)
    return dw

def main():
    # data
    filedir = os.path.dirname(os.path.realpath('__file__'))
    datapath = os.path.join(filedir,'hw/hw1/assign1_data.mat')
    data = scipy.io.loadmat(datapath)
    x1 = data['x'][:,0]
    x2 = data['x'][:,1]
    # inputs - with bias input tied high, shape is (3,100)
    X = np.array([np.full(len(x1),1),x1,x2])
    # output
    d = np.array(data['y']).flatten()
    # starting vector of weights
    w = [1,1,1]
    # sweep through the data N_EPOCH times
    for _ in range(N_EPOCH):
        w = w + dw(w,d,X)
    print(w)

# invoke main
main()