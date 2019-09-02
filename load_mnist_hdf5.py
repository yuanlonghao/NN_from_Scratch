import numpy as np
import h5py


# Load h5py MNIST file, output training set and testing set
def load_mnist(file_name):
    # load MNIST data
    mnist_data = h5py.File(file_name, 'r')
    x_train = np.float64(mnist_data['x_train'][:])
    y_train = np.int64(np.array(mnist_data['y_train'][:, 0])).reshape(-1, 1)
    x_test = np.float64(mnist_data['x_test'][:])
    y_test = np.int64(np.array(mnist_data['y_test'][:, 0])).reshape(-1, 1)
    mnist_data.close()

    # stack together for next step
    X = np.vstack((x_train, x_test))
    y = np.vstack((y_train, y_test))

    # one-hot encoding
    digits = 10
    examples = y.shape[0]
    y = y.reshape(1, examples)
    Y_new = np.eye(digits)[y.astype('int64')]
    Y_new = Y_new.T.reshape(digits, examples)

    # number of training set
    m = 60000
    X_train, X_test = X[:m].T, X[m:].T
    Y_train, Y_test = Y_new[:, :m], Y_new[:, m:]

    # shuffle training set
    shuffle_index = np.random.permutation(m)
    X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    return X_train, Y_train, X_test, Y_test

