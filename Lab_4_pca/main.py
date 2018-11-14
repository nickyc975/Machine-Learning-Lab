import numpy
import matplotlib.pyplot as plt

DATA_SCALE = 500
MEANS = [1, 3, 2]
VARIANCES = [0.2, 0.3, 0.05]

def generate_data(means, variances, data_scale):
    return numpy.random.normal(means, variances, (data_scale, len(means)))

def pca(x, k):
    centralized_x = x - numpy.sum(x, axis=0) / x.shape[0]
    covariance = numpy.dot(centralized_x.T, centralized_x)
    eigenvalues, eigenvectors = numpy.linalg.eig(covariance)
    sorted_indices = numpy.argsort(eigenvalues)
    return eigenvectors[:, sorted_indices[-k-1:-1:1]]

data = generate_data(MEANS, VARIANCES, DATA_SCALE)
eigenvectors = pca(data, len(MEANS) - 1)
projected_data = numpy.transpose(numpy.dot(data, eigenvectors))

fig, axs = plt.subplots(1, 1)
axs.scatter(projected_data[0], projected_data[1])
plt.show()