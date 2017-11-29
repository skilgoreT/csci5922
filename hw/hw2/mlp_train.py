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

#debug = True
debug = False

def forward_propagation(T, X, W1, W2):
    if debug : print(f"W1 shape is {np.shape(W1)}")
    if debug : print(f"W2 shape is {np.shape(W2)}")
    if debug : print(f"X shape is {np.shape(X)}")
    if debug : print(f"T shape is {np.shape(X)}")
    # X has shape (n_batch x n_input)
    # 'ij,kj' is W*XT and then ->ki transposes the output
    # so we end up with column-wise activations.
    #Z1 = add_bias(np.einsum('ij,kj->ki',W1,X))
    Z1 = np.einsum('ij,kj->ki',W1,X)
    Y1 = np.tanh(Z1)
    if debug : print(f"Y1 shape is {np.shape(Y1)}")
    Y1B = add_bias(Y1)
    Z2 = np.einsum('ij,kj->ki', W2, Y1B)
    Y2 = np.tanh(Z2)
    if debug : print(f"Y2 shape is {np.shape(Y2)}")
    if debug : print(f"T shape is {np.shape(T)}")
    # ouptut layer
    # * for numpy (n,) is element wise (T-Y)*(1-Y)*(1+Y)
    R = (T-Y2) # residual
    # correctly classified
    cc = [1 for r in np.nditer(np.squeeze(R)) if r >=-1 and r <= 1 ]
    M = len(cc)/float(len(R))
    D2 = R * (np.ones(np.shape(T)) + Y2) * (np.ones(np.shape(T)) - Y2)
    # einsum for dot product (overkill - LOL)
    E = (1.0/float(len(R))) * np.einsum('ij,ij', R,R)
    if debug : print(f"D2 shape is {np.shape(D2)}")

    # W2 slice that excludes the last element
    # since don't need D (delta) for bias at layer 1
    #
    # In other words -- there is no matrix element that
    # connects the input (layer 0) to the bias (layer 1)
    D1 = W2[:,:-1] * (1 + Y1) * (1 - Y1)
    if debug : print(f"D1 shape is {np.shape(D1)}")
    # Z0 is just the inputs with bias (i.e. X) - but this keeps
    # the notation consistent
    Z0 = X
    return E, M, D1, Z0, D2, Z1

def back_propagation(D2, Z1, D1, Z0, eps):
    dW2 = eps*np.einsum('ij,ij->j', D2, add_bias(Z1));
    if debug : print(f"dW2 shape is {np.shape(dW2)}")
    # outer product ij,ik-jk where the first index that represents
    # the sample label in the mini-batch is summed over
    dW1 = eps*np.einsum('ij,ik->jk', D1, Z0);
    if debug : print(f"dW1 shape is {np.shape(dW1)}")
    return dW1, dW2

def mlp_train(*, n_epoch, minibatch_size, learning_rate, n_hidden):
    # fix the seed for stable debugging
    np.random.seed(0)
    # data
    # shape is (nrow, 6)
    X = add_bias(room_prepared)
    # number of inputs
    n_input = np.shape(X)[1]
    # use domain -1,1 for teachers for tanh activation
    # shape is (nrow,1)
    T = np.array([z if z == 1 else -1 for z in room_data['Occupancy']])
    T = T.reshape(len(T),1)
    # number of outputs
    n_output = 1
    # starting matrix of weights for input layer, shape is (6 x n_hidden)
    # shape is n_outputs x n_inputs
    # equivalent to: ncol == n_input, n_row == n_output (or n_hidden)
    W1 = np.random.uniform(size=(n_hidden, n_input), low=-1, high=1)
    if debug : print(f"W1 shape is {np.shape(W1)}")
    # starting matrix of weights for hidden layer
    # shape is (n_output, n_hidden+1), +1 in first dim is for hidden layer bias
    W2 = np.random.uniform(size=(n_output, n_hidden+1), low=-1, high=1)
    if debug : print(f"W2 shape is {np.shape(W2)}")
    # minibatch processing
    XBS = np.array_split(X, np.ceil(len(X)/minibatch_size))
    TBS = np.array_split(T, np.ceil(len(T)/minibatch_size))

    for j in range(n_epoch):
        for i in range(len(XBS)):
            XB = XBS[i]
            TB = TBS[i]
            EB, MB, D1, Z0, D2, Z1 = forward_propagation(TB, XB, W1, W2)
            #print(EB)
            dW1, dW2 = back_propagation(D2, Z1, D1, Z0, learning_rate)
            W1 += dW1
            W2 += dW2

        E, M, _, _, _, _ = forward_propagation(T, X, W1, W2)
        print(f"MSE #{E}, % correct ${M}")

    # weights, performance
    return 0, 0

# invoke main
w_trained, performance = mlp_train(n_epoch=20,
                                   minibatch_size=100,
                                   learning_rate=0.00001,
                                   n_hidden=6)