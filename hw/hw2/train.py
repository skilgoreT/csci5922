# %load train.py
import os as os
import numpy as np
import scipy.io
import scipy.optimize as optimization

N_EPOCH = 100

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
    #print(np.shape(X))
    # use domain -1,1 for teachers so we can use x.w.d<0 for the classification test
    # shpae is (nrow,1)
    t = [z if z == 1 else -1 for z in room_data['Occupancy']]
    #print(np.shape(t))
    # starting vector of weights
    w = np.zeros((np.shape(room_with_bias)[1]))
    perf = []
    # sweep through the data N_EPOCH times
    for _ in range(N_EPOCH):
         dw, E = dwp(w,t,X)
         w += dw
         perf.append(1-E)
    return w, perf

# invoke main
w_perceptron, performance = train()