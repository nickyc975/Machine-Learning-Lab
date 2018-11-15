import numpy
from PIL import Image
from mnist import MNIST
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DATA_SCALE = 500
MEANS = [1, 2, 3]
VARIANCES = [0.5, 0.3, 0.1]

def generate_data(means, variances, data_scale):
    return numpy.random.normal(means, variances, (data_scale, len(means)))


def pca(x, k):
    mean = x.mean(axis=0)
    centralized_x = x - mean
    covariance = numpy.cov(centralized_x, rowvar=0)
    eigenvalues, eigenvectors = numpy.linalg.eig(covariance)
    sorted_indices = numpy.argsort(eigenvalues)
    top_k_eigvecs = eigenvectors[:, sorted_indices[:-k-1:-1]]
    return numpy.dot(numpy.dot(centralized_x, top_k_eigvecs), top_k_eigvecs.T) + mean


def construct_img(images, col, row, width, heigth):
    image = Image.new("L", (col * width, row * heigth))
    for i in range(0, len(images)):
        each_img = Image.fromarray(images[i].reshape(width, heigth).astype(numpy.uint8))
        image.paste(each_img, ((i % col) * width, int((i / row)) * heigth))
    return image


def process_images():
    org_images, labels = MNIST("./minist", return_type="numpy").load_testing()
    images = []
    for i in range(1000):
        if labels[i] == 7 and len(images) < 100:
            images.append(org_images[i])
    
    images = numpy.asarray(images)
    image = construct_img(images, 10, 10, 28, 28)
    image.show()
    
    new_image = construct_img(pca(images, 1), 10, 10, 28, 28)
    new_image.show()
    return


process_images()


data = generate_data(MEANS, VARIANCES, DATA_SCALE).T
projected_data = pca(data.T, len(MEANS) - 1).T

axs1 = plt.subplot(111, projection='3d')
axs1.set_xlim(MEANS[0] - 1, MEANS[0] + 1)
axs1.set_ylim(MEANS[1] - 1, MEANS[1] + 1)
axs1.set_zlim(MEANS[2] - 1, MEANS[2] + 1)
axs1.scatter(data[0], data[1], data[2])
axs1.scatter(projected_data[0], projected_data[1], projected_data[2])
plt.show()