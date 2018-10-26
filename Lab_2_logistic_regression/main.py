import numpy
import numpy.matlib
import matplotlib.pyplot as plt

DATA_SCALE = 500
DIMENSION = 2

SCALE = 0.5
POSITIVE = 0.8
NEGATIVE = 0.3

ETA = 1e-2
LAMBDA = 3e0
DELTA = 1e-6
GRADIENT_ITER_COUNT = 1e5
NEWTON_ITER_COUNT = 1e2


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
    while delta > DELTA and iter_count < GRADIENT_ITER_COUNT:
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
    while delta > DELTA and iter_count < GRADIENT_ITER_COUNT:
        exp_xTw = numpy.exp(X_mat.T * W_mat)
        delta_mat = ETA * X_mat * (Y_mat - exp_xTw / (1 + exp_xTw))
        W_mat = W_mat + ETA * LAMBDA * W_mat + delta_mat
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
        while delta > DELTA and iter_count < NEWTON_ITER_COUNT:
            delta_mat = df(W_mat).I * f(W_mat)
            delta = delta_mat.T * delta_mat
            W_mat = W_mat + delta_mat
            iter_count += 1
    except:
        print("Singular matrix!")
    
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
        while delta > DELTA and iter_count < NEWTON_ITER_COUNT:
            delta_mat = df(W_mat).I * f(W_mat)
            delta = delta_mat.T * delta_mat
            W_mat = W_mat + delta_mat
            iter_count += 1
    except:
        print("Singular matrix!")
    
    return W_mat


def statistics(Y, result):
    right = 0
    length = len(Y)
    for i in range(0, length):
        if (Y[i] == 0 and result[i] < 0) or (Y[i] == 1 and result[i] > 0):
            right += 1
    return float(right) / float(length)


def classifier_line(X, W_mat):
    return [(-x * W_mat[1, 0] / W_mat[2, 0] - W_mat[0, 0] / W_mat[2, 0]) for x in X]


X, Y = get_data(DATA_SCALE)
X_mat, Y_mat = numpy.mat(X).T, numpy.mat(Y).T
test_X, test_Y = get_data(DATA_SCALE)
test_X_mat, test_Y_mat = numpy.mat(test_X).T, numpy.mat(test_Y).T

W_mat_gradient = gradient_ascent(X_mat, Y_mat)
W_mat_gradient_regular = gradient_ascent_regular(X_mat, Y_mat)
W_mat_newton = newton_method(X_mat, Y_mat)
W_mat_newton_regular = newton_method_regular(X_mat, Y_mat)

gradient_result = list((W_mat_gradient.T * test_X_mat).T)
gradient_regular_result = list((W_mat_gradient_regular.T * test_X_mat).T)
newton_result = list((W_mat_newton.T * test_X_mat).T)
newton_regular_result = list((W_mat_newton_regular.T * test_X_mat).T)

gradient_rate = statistics(test_Y, gradient_result)
gradient_regular_rate = statistics(test_Y, gradient_regular_result)
newton_rate = statistics(test_Y, newton_result)
newton_regular_rate = statistics(test_Y, newton_regular_result)

X1, positive, negative = [], [], []
for i in range(0, len(test_Y)):
    if test_Y[i] == 0:
        negative.append(test_X[i])
    else:
        positive.append(test_X[i])

positive = numpy.transpose(positive)
negative = numpy.transpose(negative)
X1.extend(positive[1])
X1.extend(negative[1])

fig, axs = plt.subplots(1, 1)
axs.set_xlabel("x1")
axs.set_ylabel("x2")
axs.scatter(positive[1], positive[2], color="orange", label="Y=1")
axs.scatter(negative[1], negative[2], color="turquoise", label="Y=0")
axs.plot(X1, classifier_line(X1, W_mat_gradient), "green", label="gradient ascent, rate=%.2f%%" % (gradient_rate * 100))
axs.plot(X1, classifier_line(X1, W_mat_gradient_regular), "red", label="gradient ascent with regular term, rate=%.2f%%" % (gradient_regular_rate * 100))
axs.plot(X1, classifier_line(X1, W_mat_newton), "blue", label="newton method, rate=%.2f%%" % (newton_rate * 100))
axs.plot(X1, classifier_line(X1, W_mat_newton_regular), "yellowgreen", label="newton method with regular term, rate=%.2f%%" % (newton_regular_rate * 100))
axs.legend()
plt.show()