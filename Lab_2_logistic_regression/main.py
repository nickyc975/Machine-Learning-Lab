import numpy
import numpy.matlib
import matplotlib.pyplot as plt

DATA_SCALE = 500
DIMENSION = 2

SCALE = 0.1
POSITIVE = 0.8
NEGATIVE = 0.3

ETA = 1e-2
LAMBDA = 1e-1
DELTA = 1e-9
ITER_COUNT = 1e2


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
    iter_count = 0
    W_mat = numpy.matlib.zeros((DIMENSION + 1, 1))
    while delta > DELTA and iter_count < ITER_COUNT:
        exp_xTw = numpy.exp(X_mat.T * W_mat)
        delta_mat = ETA * X_mat * (Y_mat - exp_xTw / (1 + exp_xTw))
        delta = delta_mat.T * delta_mat
        W_mat = W_mat + delta_mat
        iter_count += 1
    return W_mat


def gradient_ascent_regular(X_mat, Y_mat):
    delta = 1
    iter_count = 0
    W_mat = numpy.matlib.zeros((DIMENSION + 1, 1))
    while delta > DELTA and iter_count < ITER_COUNT:
        exp_xTw = numpy.exp(X_mat.T * W_mat)
        delta_mat = ETA * X_mat * (Y_mat - exp_xTw / (1 + exp_xTw))
        W_mat = W_mat - ETA * LAMBDA * W_mat + delta_mat
        delta = delta_mat.T * delta_mat
        iter_count += 1
    return W_mat


def newton_method(X_mat, Y_mat):
    def f(W_mat):
        exp_xTw = numpy.exp(X_mat.T * W_mat)
        return X_mat * (Y_mat - exp_xTw / (1 + exp_xTw))

    def df(W_mat):
        exp_xTw = numpy.exp(X_mat.T * W_mat)
        A = numpy.multiply((exp_xTw / (1 + exp_xTw)), (1 / (1 + exp_xTw)))
        EA = numpy.multiply(numpy.identity(2 * DATA_SCALE), A)
        return X_mat * EA * X_mat.T

    delta = 1
    iter_count = 0
    W_mat = numpy.matlib.zeros((DIMENSION + 1, 1))

    try:
        while delta > DELTA and iter_count < ITER_COUNT:
            delta_mat = df(W_mat).I * f(W_mat)
            delta = delta_mat.T * delta_mat
            W_mat = W_mat + delta_mat
            iter_count += 1
    except:
        print("Singular matrix!")
        exit()
    
    return W_mat


def newton_method_regular(X_mat, Y_mat):
    def f(W_mat):
        exp_xTw = numpy.exp(X_mat.T * W_mat)
        return  X_mat * (Y_mat - exp_xTw / (1 + exp_xTw)) - LAMBDA * W_mat

    def df(W_mat):
        exp_xTw = numpy.exp(X_mat.T * W_mat)
        A = numpy.multiply((exp_xTw / (1 + exp_xTw)), (1 / (1 + exp_xTw)))
        EA = numpy.multiply(numpy.identity(2 * DATA_SCALE), A)
        return X_mat * EA * X_mat.T

    delta = 1
    iter_count = 0
    W_mat = numpy.matlib.zeros((DIMENSION + 1, 1))

    try:
        while delta > DELTA and iter_count < ITER_COUNT:
            delta_mat = df(W_mat).I * f(W_mat)
            delta = delta_mat.T * delta_mat
            W_mat = W_mat + delta_mat
            iter_count += 1
    except:
        print("Singular matrix!")
        exit()
    
    return W_mat


X, Y = get_data(DATA_SCALE)

X_mat, Y_mat = numpy.mat(X).T, numpy.mat(Y).T

W_mat = newton_method(X_mat, Y_mat)

test_scale = 500

right_count = 0.0

test_X, test_Y = get_data(test_scale)

test_X_mat, test_Y_mat = numpy.mat(test_X).T, numpy.mat(test_Y).T

result_Y = list((W_mat.T * test_X_mat).T)

pos_x_1 = []
pos_x_2 = []
neg_x_1 = []
neg_x_2 = []
for i in range(0, 2 * test_scale):
    if (test_Y[i] == 0 and result_Y[i] < 0) or (test_Y[i] == 1 and result_Y[i] > 0):
        right_count += 1.0
    if test_Y[i] == 0:
        neg_x_1.append(test_X[i][1])
        neg_x_2.append(test_X[i][2])
    else:
        pos_x_1.append(test_X[i][1])
        pos_x_2.append(test_X[i][2])

print(right_count / (test_scale * 2))
fig, axs = plt.subplots(1, 1)
axs.scatter(pos_x_1, pos_x_2)
axs.scatter(neg_x_1, neg_x_2)
axs.plot(pos_x_1 + neg_x_1, [(-x * W_mat[1, 0] / W_mat[2, 0] - W_mat[0, 0] / W_mat[2, 0]) for x in pos_x_1 + neg_x_1])
plt.show()