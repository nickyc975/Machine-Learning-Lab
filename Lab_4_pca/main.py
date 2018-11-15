import numpy
from PIL import Image
from mnist import MNIST
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DATA_SCALE = 100
MEANS = [1, 2, 3]
VARIANCES = [0.5, 0.3, 0.1]


def generate_data(means, variances, data_scale):
    """
    Generate data for dimensionality reduction and reconstruction.

    :param means: mean vector of the multi-dimensional data.
    :param vriances: variances vector of the multi-dimensional data.
    :param data_scale: number of data to generate.
    :return : data generated, data.shape = (data_scale, len(means)).
    """
    return numpy.random.normal(means, variances, (data_scale, len(means)))


def pca(x, k):
    """
    Calculate the k eigenvectors with top k eigenvalues of covariance matrix of x.

    :param x: dataset to calculate eigenvectors.
    :param k: number of eigenvectors to return.
    :return : k eigenvectors with top k eigenvalues.
    """
    mean = x.mean(axis=0)
    centralized_x = x - mean
    covariance = numpy.cov(centralized_x, rowvar=0)
    eigenvalues, eigenvectors = numpy.linalg.eig(covariance)
    sorted_indices = numpy.argsort(eigenvalues)
    return eigenvectors[:, sorted_indices[: -k - 1 : -1]]


def reconstruct(x, eigvectors):
    """
    Reconstruct dataset with given eigvectors.

    :param x: dataset to reconstruct.
    :param eigvectors: eigvectors used to reconstruct the dataset.
    :return : reconstructed dataset.
    """
    mean = x.mean(axis=0)
    centralized_x = x - mean
    return numpy.dot(numpy.dot(centralized_x, eigvectors), eigvectors.T) + mean


def construct_img(images, col, row, width, heigth):
    """
    Construct an image with col * row small images. The size of each small image is width * height.

    :param images: col * row small images used to construct the result image.
    :param col: number of columns of small images in the result image.
    :param row: number of rows of small images in the result image.
    :param width: width of each small image.
    :param height: height of each small image.
    :return: image constructed with col * row small images.
    """
    image = Image.new("L", (col * width, row * heigth))
    for i in range(0, len(images)):
        each_img = Image.fromarray(images[i].reshape(width, heigth).astype(numpy.uint8))
        image.paste(each_img, ((i % col) * width, int((i / row)) * heigth))
    return image


def parse_mnist(mnist_dir, digit):
    """
    Parse mnist dataset files to get 100 training images and 100 testing images.

    :param mnist_dir: path of the dir that contains the mnist dataset files.
    :param digit: ths label of the images to choose.
    :return : 100 images as training set, 100 images as testing set. 
    """
    i = 0
    images = []
    image_arrays, labels = MNIST(mnist_dir, return_type="numpy").load_testing()
    while i < len(labels) and len(images) < 200:
        if labels[i] == digit:
            images.append(image_arrays[i])
        i += 1

    images = numpy.asarray(images)
    return images[0:100], images[100:200]


# Test with mnist dataset.

NUMBER = 6
USE_EIGVEC_NUM = 1
MNIST_DIR = "./Lab_4_pca/mnist"

training_images, testing_images = parse_mnist(MNIST_DIR, NUMBER)
recon_testing_images = reconstruct(testing_images, pca(training_images, USE_EIGVEC_NUM))

construct_img(testing_images, 10, 10, 28, 28).save("testing.bmp")
construct_img(recon_testing_images, 10, 10, 28, 28).save("recon_testing.bmp")


# Test with generated dataset.

data = generate_data(MEANS, VARIANCES, DATA_SCALE)
projected_data = reconstruct(data, pca(data, len(MEANS) - 1)).T

data_T = data.T
axs1 = plt.subplot(111, projection="3d")
axs1.set_xlim(MEANS[0] - 1, MEANS[0] + 1)
axs1.set_ylim(MEANS[1] - 1, MEANS[1] + 1)
axs1.set_zlim(MEANS[2] - 1, MEANS[2] + 1)
axs1.scatter(data_T[0], data_T[1], data_T[2])
axs1.scatter(projected_data[0], projected_data[1], projected_data[2])
plt.show()
