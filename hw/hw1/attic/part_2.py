#%%
import os
import numpy as np
import scipy.io
import scipy.optimize as optimization
#from IPython.core.debugger import set_trace

# global const
BATCH_SIZE = 10
EPS = 0.001
N_EPOCH = 1000

def process_batch(batch):
    print(len(batch))

def func(w, x1, x2):
    return w[0] + w[1]*x1 + w[2]*x2

def delta_w_v2(w, y, x1, x2):
    dw = np.array([0., 0., 0.])
    for i in range(len(y)):
        dw[0] = dw[0] + (y[i] - func(w, x1[i], x2[i]))
        dw[1] = dw[1] + (y[i] - func(w, x1[i], x2[i]))*x1[i]
        dw[2] = dw[2] + (y[i] - func(w, x1[i], x2[i]))*x2[i]
    return EPS*dw

def delta_w(w, y, x1, x2):
    c = y - (w[0] + w[1]*x1 + w[2]*x2)
    delta = EPS*np.array([np.sum(c), np.sum(c*x1), np.sum(c*x2)])
    return delta

def main():
    filedir = os.path.dirname(os.path.realpath('__file__'))
    datapath = os.path.join(filedir,'hw/hw1/assign1_data.mat')
    data = scipy.io.loadmat(datapath)
    x1 = np.array(data['x'][:,0])
    x2 = np.array(data['x'][:,1])
    y = np.array(data['y']).flatten()

    #batches = np.array_split(x1, BATCH_SIZE)
    #for batch in batches:
    #    process_batch(batch)

    w = np.array([1,1,1])
    for _ in range(N_EPOCH):
        dw = delta_w(w,y,x1,x2)
        print(dw)
        dw = delta_w_v2(w,y,x1,x2)
        print(dw)
        w = w + dw
        print(w)
    print('all done')

main()
