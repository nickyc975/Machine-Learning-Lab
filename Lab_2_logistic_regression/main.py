import numpy

DATA_SCALE = 50
DIMENSION = 5

SCALE = 0.3
POSITIVE = 0.8
NEGATIVE = 0.3

ETA = 1e-2
LAMBDA = 1e-3
DELTA = 1e-6


def get_data(data_scale):
    X, Y = [], []
    for i in range(0, data_scale):
        x = [1]
        x.extend(numpy.random.normal(POSITIVE, SCALE, DIMENSION))
        X.append(x)
        Y.append(1)
        x = [1]
        x.extend(numpy.random.normal(NEGATIVE, SCALE, DIMENSION))
        X.append(x)
        Y.append(0)
    return X, Y


def gradient_ascent(X_mat, Y_mat):
    delta = 1
    W_mat = numpy.mat([0] * (DIMENSION + 1)).T
    while delta > DELTA:
        exp_xTw = numpy.exp(X_mat.T * W_mat)
        delta_mat = ETA * X_mat * (Y_mat - exp_xTw / (1 + exp_xTw))
        delta = delta_mat.T * delta_mat
        W_mat = W_mat + delta_mat
    return W_mat


def gradient_ascent_regular(X_mat, Y_mat):
    delta = 1
    W_mat = numpy.mat([0] * (DIMENSION + 1)).T
    while delta > DELTA:
        exp_xTw = numpy.exp(X_mat.T * W_mat)
        delta_mat = ETA * X_mat * (Y_mat - exp_xTw / (1 + exp_xTw))
        W_mat = W_mat - ETA * LAMBDA * W_mat + delta_mat
        delta = delta_mat.T * delta_mat
    return W_mat


def newton_method(X_mat, Y_mat):
    pass


X, Y = get_data(DATA_SCALE)

X_mat, Y_mat = numpy.mat(X).T, numpy.mat(Y).T

W_mat = gradient_ascent_regular(X_mat, Y_mat)

test_X, test_Y = get_data(5)

test_X_mat, test_Y_mat = numpy.mat(test_X).T, numpy.mat(test_Y).T

print(test_Y)

for i in range(0, 10):
    print(W_mat.T * test_X_mat[:, i])
