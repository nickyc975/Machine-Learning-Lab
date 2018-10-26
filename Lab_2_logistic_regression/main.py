import numpy
import numpy.matlib
import matplotlib.pyplot as plt

DATA_SCALE = 500
DIMENSION = 2

SCALE = 0.5
POSITIVE_MEAN = 0.8
NEGATIVE_MEAN = 0.3

ETA = 1e-3
LAMBDA = 1e0
DELTA = 1e-6
GRADIENT_ITER_COUNT = 1e6
NEWTON_ITER_COUNT = 1e2


def get_data(data_scale):
    """
    Generate data with given parameters.

    :param data_scale: data scale to generate.
    """
    X, Y = [], []
    for i in range(0, data_scale):
        x = [1]
        x.extend(numpy.random.normal(POSITIVE_MEAN, SCALE, DIMENSION))
        X.append(x)
        Y.append(1)
        x = [1]
        x.extend(numpy.random.normal(NEGATIVE_MEAN, SCALE, DIMENSION))
        X.append(x)
        Y.append(0)
    return X, Y


def gradient_ascent(X_mat, Y_mat):
    """
    Gradient ascent method without regular term.

    :param X_mat: matrix of property of samples, shape=(DIMENSION + 1, 2 * DATA_SCALE)
    :param Y_mat: vector of type of samples, shape=(2 * DATA_SCALE, 1)
    """
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
    """
    Gradient ascent method with regular term.

    :param X_mat: matrix of property of samples, shape=(DIMENSION + 1, 2 * DATA_SCALE)
    :param Y_mat: vector of type of samples, shape=(2 * DATA_SCALE, 1)
    """
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
    """
    Newton method without regular term.

    :param X_mat: matrix of property of samples, shape=(DIMENSION + 1, 2 * DATA_SCALE)
    :param Y_mat: vector of type of samples, shape=(2 * DATA_SCALE, 1)
    """
    def f(W_mat):
        """
        First derivative of l(W)

        :param W_mat: vector W, shape=(DIMENSION + 1, 1)
        """
        exp_xTw = numpy.exp(X_mat.T * W_mat)
        return X_mat * (Y_mat - exp_xTw / (1 + exp_xTw))

    def df(W_mat):
        """
        Second derivative of l(W)

        :param W_mat: vector W, shape=(DIMENSION + 1, 1)
        """
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
    """
    Newton method with regular term.

    :param X_mat: matrix of property of samples, shape=(DIMENSION + 1, 2 * DATA_SCALE)
    :param Y_mat: vector of type of samples, shape=(2 * DATA_SCALE, 1)
    """
    def f(W_mat):
        """
        First derivative of l(W)

        :param W_mat: vector W, shape=(DIMENSION + 1, 1)
        """
        exp_xTw = numpy.exp(X_mat.T * W_mat)
        return  X_mat * (Y_mat - exp_xTw / (1 + exp_xTw)) - LAMBDA * W_mat

    def df(W_mat):
        """
        Second derivative of l(W)

        :param W_mat: vector W, shape=(DIMENSION + 1, 1)
        """
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
    """
    Calculate accuracy of the result "result".

    :param Y: list Y, the exact type of each sample.
    :param result: list result, the classifying result.
    """
    right = 0
    length = len(Y)
    for i in range(0, length):
        if (Y[i] == 0 and result[i] < 0) or (Y[i] == 1 and result[i] > 0):
            right += 1
    return float(right) / float(length)


def classifier_line(X, W_mat):
    """
    Used to draw the classifying line.

    :param X: list X, the samples.
    :param W_mat: the result of optimizing methods.
    """
    return [(-x * W_mat[1, 0] / W_mat[2, 0] - W_mat[0, 0] / W_mat[2, 0]) for x in X]


# Generate training data and testing data.
X, Y = get_data(DATA_SCALE)
X_mat, Y_mat = numpy.mat(X).T, numpy.mat(Y).T
test_X, test_Y = get_data(DATA_SCALE)
test_X_mat, test_Y_mat = numpy.mat(test_X).T, numpy.mat(test_Y).T

# Call optimizing methods.
W_mat_gradient = gradient_ascent(X_mat, Y_mat)
W_mat_gradient_regular = gradient_ascent_regular(X_mat, Y_mat)
W_mat_newton = newton_method(X_mat, Y_mat)
W_mat_newton_regular = newton_method_regular(X_mat, Y_mat)

# Classify.
gradient_result = list((W_mat_gradient.T * test_X_mat).T)
gradient_regular_result = list((W_mat_gradient_regular.T * test_X_mat).T)
newton_result = list((W_mat_newton.T * test_X_mat).T)
newton_regular_result = list((W_mat_newton_regular.T * test_X_mat).T)

# Calculate accuracy.
gradient_rate = statistics(test_Y, gradient_result)
gradient_regular_rate = statistics(test_Y, gradient_regular_result)
newton_rate = statistics(test_Y, newton_result)
newton_regular_rate = statistics(test_Y, newton_regular_result)

# Process samples for drawing graphs.
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

# Draw graphs.
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