# %load mlp_train.py
import os as os
import numpy as np
import scipy.io
import scipy.optimize as optimization

def add_bias(X):
    bias = np.full((np.shape(X)[0],
                    np.shape(X)[1]+1), 1.0)
    bias[:,1:] = X
    return bias

def forward_propagation(X, W1, W2):
    print(f"W1 shape is {np.shape(W1)}")
    print(f"X shape is {np.shape(X)}")
    # X has shape (n_batch x n_input)
    # 'ij,kj' is W*XT and then ->ki transposes the output
    # so we end up with column-wise activations.
    Z1 = np.tanh(np.einsum('ij,kj->ki',W1,X))
    Z1 = add_bias(Z1)
    print(f"Z1 shape is {np.shape(Z1)}")
    Z2 = np.tanh(np.einsum('ij,kj->ki',W2,Z1))
    print(f"Z2 shape is {np.shape(Z2)}")
#    Z2 = np.tanh(np.einsum('ij,kj->ki',W2,Z1))

def mlp_train(*, n_epoch, minibatch_size, learning_rate, n_hidden):
    # data
    # shape is (nrow, 6)
    X = add_bias(room_prepared)
    # number of inputs
    n_input = np.shape(X)[1]
    # use domain -1,1 for teachers for tanh activation
    # shape is (nrow,1)
    t = np.array([z if z == 1 else -1 for z in room_data['Occupancy']])
    # number of outputs
    n_output = 1
    # starting matrix of weights for input layer, shape is (6 x n_hidden)
    # shape is n_outputs x n_inputs
    # equivalent to: ncol == n_input, n_row == n_output (or n_hidden)
    W1 = np.random.uniform(size=(n_hidden, n_input), low=-1, high=1)
    # starting matrix of weights for hidden layer
    # shape is (n_output, n_hidden+1), +1 in first dim is for hidden layer bias
    W2 = np.random.uniform(size=(n_output, n_hidden+1), low=-1, high=1)
    # minibatch processing

    XBS = np.array_split(X, np.ceil(len(X)/minibatch_size))
    for XB in XBS:
        forward_propagation(XB, W1, W2)
        break

    # w = np.zeros((np.shape(room_with_bias)[1]))
    # perf = []
    # # sweep through the data N_EPOCH times
    # for _ in range(N_EPOCH):
    #      dw, E = dwp(w,t,X)
    #      w += dw
    #      perf.append(1-E)
    # return w, perf
    return 'foo', 'bar'
# invoke main
w_trained, performance = mlp_train(n_epoch=1,
                                   minibatch_size=10,
                                   learning_rate=0.001,
                                   n_hidden=5)