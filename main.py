import numpy as np
from nn_flows import *
from load_mnist_hdf5 import load_mnist

# Network structure definition
nn_layers = [
    {"size_in": 784, "size_out": 256, "act_func": "relu"},
    # {"size_in": 128, "size_out": 64, "act_func": "relu"},
    # {"size_in": 64, "size_out": 32, "act_func": "relu"},
    {"size_in": 256, "size_out": 10, "act_func": "softmax"}
            ]

# Set hyper-parameters
epochs = 30
batch_size = 128
learning_rate = 0.2

# Load MNIST file
X_train, y_train, X_test, y_test = load_mnist("MNIST_data.hdf5")

# Initialize all the weight and bias
param_cache = init_layers(nn_layers)

# Training
param_cache, test_acc = nn_train(X_train, y_train, X_test, y_test, nn_layers, param_cache, epochs, batch_size, learning_rate)
print("Final accuracy on the test set: {:.3f}".format(test_acc))
