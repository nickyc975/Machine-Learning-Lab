import numpy

DATA_SCALE = 50
DIMENSION = 10

ETA = 0.01
DELTA = 1e-7


def get_data(data_scale):
    X = []
    Y = []
    for i in range(0, data_scale):
        x = [1]
        x.extend(numpy.random.normal(0.8, 0.1, 10))
        X.append(x)
        Y.append(1)
        x = [1]
        x.extend(numpy.random.normal(0.3, 0.1, 10))
        X.append(x)
        Y.append(0)
    return X, Y


def gradient_ascent(X_mat, Y_mat):
    delta = 1
    W = numpy.mat([0] * (DIMENSION + 1)).T
    while delta > DELTA:
        V = []
        for i in range(0, 2 * DATA_SCALE):
            temp = numpy.exp(W.T * X_mat[:, i])[0, 0]
            V.append(temp/(1 + temp))
        V_mat = Y_mat - numpy.mat(V).T
        delta_mat = ETA * X_mat * V_mat
        W = W + delta_mat
        delta = delta_mat.T * delta_mat
    return W


X, Y = get_data(DATA_SCALE)

X_mat, Y_mat = numpy.mat(X).T, numpy.mat(Y).T

W_mat = gradient_ascent(X_mat, Y_mat)

test_X, test_Y = get_data(5)

test_X_mat, test_Y_mat = numpy.mat(test_X).T, numpy.mat(test_Y).T

print(test_Y)

for i in range(0, 10):
    print(W_mat.T * test_X_mat[:, i])
