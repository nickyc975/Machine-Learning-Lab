import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DATA_SCALE = 500
MEANS = [1, 2, 3]
VARIANCES = [0.5, 0.3, 0.1]

def generate_data(means, variances, data_scale):
    return numpy.random.normal(means, variances, (data_scale, len(means)))

def pca(x, k):
    centralized_x = x - numpy.sum(x, axis=0) / x.shape[0]
    covariance = numpy.dot(centralized_x.T, centralized_x)
    eigenvalues, eigenvectors = numpy.linalg.eig(covariance)
    sorted_indices = numpy.argsort(eigenvalues)
    return eigenvectors[:, sorted_indices[:-k-1:-1]]

data = generate_data(MEANS, VARIANCES, DATA_SCALE)
eigenvectors = pca(data, len(MEANS) - 1)

data = data.T
projected_data = numpy.dot(eigenvectors.T, data)

axs1 = plt.subplot(111, projection='3d')
axs1.set_xlim(MEANS[0] - 1, MEANS[0] + 1)
axs1.set_ylim(MEANS[1] - 1, MEANS[1] + 1)
axs1.set_zlim(MEANS[2] - 1, MEANS[2] + 1)
axs1.scatter(data[0], data[1], data[2])

fig, axs2 = plt.subplots(1, 1)
axs2.scatter(projected_data[0], projected_data[1])

plt.show()