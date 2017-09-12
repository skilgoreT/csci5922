import os as os
import numpy as np
import scipy.io
import scipy.optimize as optimization

N_EPOCH = 20

n_mis = []
def dwp(w,d,X):
    # using numpy einstein summation to vectorize the computation
    # calculating w.x.d -- w.x.d < 0 => incorrect classification
    C = d*np.einsum('i,ij->j', w, X) # shape is (100,)
    XT = np.einsum('ij->ji', X)
    # use enumerate for the equivalent to each_with_index (Ruby)
    xd = [ d[i]*XT[i] for i,c in enumerate(C) if c < 0]
    # sum xd element-wise
    # dw is x.t summed over misclassified teachers
    dw =  np.einsum('ij->j',xd) if(len(xd) > 0) else 0 # shape is (3,)
    return dw

def train(n, X, d):
    # use n_sample for training
    X = X[:,0:n]
    d = d[0:n]
    # starting vector of weights
    w = [1,1,1]
    # sweep through the data N_EPOCH times
    for _ in range(N_EPOCH):
        w = w + dwp(w,d,X)

    return w

def  performance(w,X,d):
    X = X[:,75:]
    d = d[75:]
    C = d*np.einsum('i,ij->j', w, X) # shape is (100,)
    XT = np.einsum('ij->ji', X)
    # d.x.t for misclassified
    xd = [ d[i]*XT[i] for i,c in enumerate(C) if c < 0]
    return len(xd)/25.0

# invoke main
def main():
    # data
    filedir = os.path.dirname(os.path.realpath('__file__'))
    datapath = os.path.join(filedir,'assign1_data.mat')
    data = scipy.io.loadmat(datapath)
    x1 = data['x'][:,0]
    x2 = data['x'][:,1]
    # inputs - with bias input tied high, shape is (3,100)
    X = np.array([np.full(len(x1),1),x1,x2])
    # output
    d = np.array(data['z']).flatten()
    # use domain -1,1 for teachers so we can use x.w.d<0 for the classification test
    d = [z if z == 1 else -1 for z in d]
    samples = [5, 10, 25, 50, 75]
    perf = []
    for n in samples:
        w = train(n, X, d)
        p = performance(w, X, d)
        perf.append(1-p)
    return samples, perf

samples, perf = main()