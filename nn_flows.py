import numpy as np


# Define activation functions and their back propagation: sigmoid, relu, and softmax
def sigmoid(z):
    return 1./(1.+np.exp(-z))


def sigmoid_backward(da, z):
    return da * sigmoid(z) * (1. - sigmoid(z))


def relu(z):
    return np.maximum(0, z)


def relu_backward(da, z):
    dz = np.array(da, copy=True)
    dz[z <= 0] = 0
    return dz


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def softmax_backward(dA, Z):
    return dA


# Initialize all the parameters of the neural network
def init_layers(nn_layers, seed=0):
    # nn_layer contains the structure setting of the neural network
    # seed is to ensure the same random initialization
    np.random.seed(seed)

    param_cache = {}

    for idx, layer in enumerate(nn_layers):
        layer_idx = idx + 1

        layer_input_size = layer["size_in"]
        layer_output_size = layer["size_out"]

        param_cache['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        param_cache['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
    return param_cache


# Functions for forward and backward propagations
def forward_one(A_prev, W_curr, b_curr, act_func="relu"):
    # A_prev is the output of last layer after activation function
    # W_curr and b_curr are the weight matrix and bias of the current layer
    # act_func is the type of activation function
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    if act_func is "relu":
        activation_func = relu
    elif act_func is "sigmoid":
        activation_func = sigmoid
    elif act_func is "softmax":
        activation_func = softmax
    else:
        raise Exception('Non-supported activation function')

    return activation_func(Z_curr), Z_curr


def forward_all(X, param_cache, nn_layers):
    # X is the input of the network: feature x sample
    # param_cache contains all the weight and bias
    layer_out_cache = {}
    A_curr = X

    for idx, layer in enumerate(nn_layers):
        layer_idx = idx + 1
        A_prev = A_curr

        activ_function_curr = layer["act_func"]
        W_curr = param_cache["W" + str(layer_idx)]
        b_curr = param_cache["b" + str(layer_idx)]
        A_curr, Z_curr = forward_one(A_prev, W_curr, b_curr, activ_function_curr)

        layer_out_cache["A" + str(idx)] = A_prev
        layer_out_cache["Z" + str(layer_idx)] = Z_curr

    return A_curr, layer_out_cache


def backward_one(dA_curr, W_curr, b_curr, Z_curr, A_prev, act_func="relu"):
    # dA_curr is the derivate w.r.t. the current activation function

    m = A_prev.shape[1]

    if act_func is "relu":
        backward_activation_func = relu_backward
    elif act_func is "sigmoid":
        backward_activation_func = sigmoid_backward
    elif act_func is "softmax":
        backward_activation_func = softmax_backward
    else:
        raise Exception('No supported activation function')

    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def backward_all(Y_hat, Y, memory, param_cache, nn_layers):
    # Y_hat is the output of the whole network
    # Y is the real label

    grads_values = {}

    # number of examples
    m = Y.shape[1]
    # a hack ensuring the same shape of the prediction vector and labels vector
    Y = Y.reshape(Y_hat.shape)

    # initiation of gradient descent algorithm
    # dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
    dA_prev = Y_hat - Y
    for layer_idx_prev, layer in reversed(list(enumerate(nn_layers))):
        # we number network layers from 1
        layer_idx_curr = layer_idx_prev + 1
        # extraction of the activation function for the current layer
        activ_function_curr = layer["act_func"]

        dA_curr = dA_prev

        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]

        W_curr = param_cache["W" + str(layer_idx_curr)]
        b_curr = param_cache["b" + str(layer_idx_curr)]

        dA_prev, dW_curr, db_curr = backward_one(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr

    return grads_values


# Update the parameters: W and b
def update_param(param_cache, grads_values, nn_layers, lr):

    # iteration over network layers
    for layer_idx, layer in enumerate(nn_layers, 1):
        param_cache["W" + str(layer_idx)] -= lr * grads_values["dW" + str(layer_idx)]
        param_cache["b" + str(layer_idx)] -= lr * grads_values["db" + str(layer_idx)]

    return param_cache


# Calculate the value of loss function
def compute_loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1/m) * L_sum
    return L


# Training
def nn_train(X_train, y_train, X_test, y_test, nn_layers, param_cache, epochs, batch_size, learning_rate):
    for i in range(epochs):
        # Random shuffle the training set
        shuffle_index = np.random.permutation(X_train.shape[1])
        X_train_shuffled = X_train[:, shuffle_index]
        y_train_shuffled = y_train[:, shuffle_index]

        for j in range(int(X_train.shape[1]/batch_size)):
            # Get mini-batch
            begin = j * batch_size
            end = min(begin + batch_size, X_train.shape[1] - 1)
            x_batch = X_train_shuffled[:, begin:end]
            y_batch = y_train_shuffled[:, begin:end]

            # Forward to calculate loss
            Y_hat, cache = forward_all(x_batch, param_cache, nn_layers)

            # Backward to calculate gradients
            grads_values = backward_all(Y_hat, y_batch, cache, param_cache, nn_layers)

            # Update W and b
            param_cache = update_param(param_cache, grads_values, nn_layers, learning_rate)

        # Evaluation in every epoch

        # Prediction of the training set
        Y_hat, _ = forward_all(X_train, param_cache, nn_layers)
        cost_train = compute_loss(y_train, Y_hat)
        acc_train = np.mean(Y_hat.argmax(axis=0) == y_train.argmax(axis=0))

        # Prediction of the test set
        Y_test_hat, _ = forward_all(X_test, param_cache, nn_layers)
        cost_test = compute_loss(y_test, Y_test_hat)
        acc_test = np.mean(Y_test_hat.argmax(axis=0) == y_test.argmax(axis=0))

        # Print performance in every epoch
        print("Epoch {} finished, train_cost: {:.4f}, test_cost: {:.4f}, train_acc = {:.3f}, test_acc: {:.3f}".format(
            i + 1, cost_train, cost_test, acc_train, acc_test))
    return param_cache, acc_test


