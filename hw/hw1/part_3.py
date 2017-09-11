import os
import numpy as np
import scipy.io
import scipy.optimize as optimization

N_EPOCH = 30
def dwp(w,d,X):
    # using numpy einstein summation to vectorize the computation
    # calculating w.x.d -- w.x.d < 0 => incorrect classification
    C = d*np.einsum('i,ij->j', w, X) # shape is (100,)
    XT = X.transpose()
    # use enumerate for the equivalent to each_with_index (Ruby)
    xd = [ d[i]*XT[i] for i,c in enumerate(C) if c < 0]
    print(len(xd))
    # sum xd element-wise
    # dw is x.t summed over misclassified teachers
    dw = np.einsum('ij->j',xd) # shape is (3,)
    return dw

def main():
    # data
    filedir = os.path.dirname(os.path.realpath('__file__'))
    datapath = os.path.join(filedir,'assign1_data.mat')
    data = scipy.io.loadmat(datapath)
    x1 = data['x'][:,0]
    x2 = data['x'][:,1]
    # inputs - with bias input tied high, shape is (3,100)
    X = np.array([np.full(len(x1),1),x1,x2])
    XT = X.transpose();
    # output
    d = np.array(data['z']).flatten()
    # use domain -1,1 for teachers so we can use x.w.d<0 for the classification test
    d = map(lambda x: -1 if x==0 else x, d)
    # starting vector of weights
    w = [1,1,1]
    # sweep through the data N_EPOCH times
    for _ in range(N_EPOCH):
        w = w + dwp(w,d,X)
    print(w)

# invoke main
main()