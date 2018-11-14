import numpy
import numpy.matlib
import matplotlib.pyplot as plt

CLS_NUM = 4
SAMPLES_PER_CLS = 1000
MEANS = [[1, 1], [-1, 1], [1, -1], [-1, -1]]
SCALES = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]]

DELTA = 1e-9
ITER_COUNT = 1e9


def generate_data(samples_per_cls=int, means=list, scales=list):
    return numpy.asarray(
        [
            numpy.random.normal(means[i], scales[i], (samples_per_cls, len(means[i])))
            for i in range(0, len(means))
        ]
    )


def cal_distance(x1=list, x2=list):
    x1, x2 = numpy.mat(x1), numpy.mat(x2)
    return (x1 - x2) * (x1 - x2).T


def select_means(data_set, cls_num):
    # return data_set[0:cls_num]

    return [data_set[i, :] for i in numpy.random.randint(0, len(data_set), cls_num)]

    # step = int(len(data_set) / cls_num)
    # return [data_set[i * step] for i in range(0, cls_num)]

    # return MEANS


def k_means_cluster(data_set, cls_num):
    delta = 1
    iter_count = 0
    result = [[] for i in range(0, cls_num)]
    means = select_means(data_set, cls_num)
    while delta > DELTA and iter_count < ITER_COUNT:
        result = [[] for i in range(0, cls_num)]
        for data in data_set:
            distances = [cal_distance(data, mean)[0, 0] for mean in means]
            cls_index = distances.index(min(distances))
            result[cls_index].append(data)

        new_means = [
            list(numpy.sum(data_cls, axis=0) / (len(data_cls) + 1))
            for data_cls in result
        ]
        delta_mat = numpy.mat(new_means) - numpy.mat(means)
        delta = numpy.sum(delta_mat.T * delta_mat)
        means = new_means
        iter_count += 1
    return result, means


def gauss(x, mu, sigma):
    det = numpy.linalg.det(sigma)
    inv = numpy.linalg.inv(sigma)
    const = (2 * numpy.pi) ** (x.shape[0] / 2)
    exp = numpy.exp(-0.5 * numpy.dot(numpy.dot((x - mu).T, inv), (x - mu)))

    return exp / (const * (det ** 0.5))


def gmm_cluster(data_set, cls_num):
    iter_count = 0
    MAX_ITER_COUNT = 25
    sample_num = data_set.shape[0]
    sample_dim = data_set.shape[1]
    data_set_T = numpy.transpose(data_set)

    alphas = numpy.asarray([1.0 / cls_num] * cls_num).T
    means = numpy.asarray(select_means(data_set, cls_num))
    covariances = numpy.asarray([numpy.cov(data_set_T) for i in range(0, cls_num)])

    gammas = numpy.asarray(
        [[alphas[j] for j in range(0, cls_num)] for i in range(0, sample_num)]
    )

    while iter_count < MAX_ITER_COUNT:
        p = numpy.asarray(
            [
                [
                    gauss(
                        data_set_T[:, i].reshape(sample_dim, 1),
                        means[j].reshape(sample_dim, 1),
                        covariances[j],
                    )[0, 0]
                    for j in range(0, cls_num)
                ]
                for i in range(0, sample_num)
            ]
        )
        
        gammas = numpy.asarray(
            [
                numpy.multiply(p[i], alphas) / numpy.dot(p[i].T, alphas)
                for i in range(0, sample_num)
            ]
        )

        means = numpy.asarray(
            [
                (numpy.dot(data_set_T, gammas[:, i]) / numpy.sum(gammas[:, i])).reshape(sample_dim, 1) 
                for i in range(0, cls_num)
            ]
        )

        covariances = numpy.asarray(
            [
                numpy.dot(
                    numpy.multiply(data_set_T - means[i], gammas[:, i]),
                    (data_set_T - means[i]).T,
                )
                / numpy.sum(gammas[:, i])
                for i in range(0, cls_num)
            ]
        )
        alphas = numpy.asarray([numpy.sum(gammas[:, i]) / sample_num for i in range(0, cls_num)])
        iter_count += 1

    return means, gammas


clustered_data_set = generate_data(SAMPLES_PER_CLS, MEANS, SCALES)
data_set = numpy.concatenate(clustered_data_set)

gmm_means, gammas = gmm_cluster(data_set, CLS_NUM)
result, k_means_means = k_means_cluster(data_set, CLS_NUM)

gmm_result = [[] for i in range(0, CLS_NUM)]
for i in range(0, len(data_set)):
    cls_index = gammas[i].tolist().index(numpy.max(gammas[i]))
    gmm_result[cls_index].append(data_set[i])

fig, (axs1, axs2) = plt.subplots(1, 2)
for i in range(0, len(result)):
    gmm_data = numpy.transpose(gmm_result[i])
    k_means_data = numpy.transpose(result[i])
    axs1.scatter(k_means_data[0], k_means_data[1])
    axs2.scatter(gmm_data[0], gmm_data[1])
    axs1.scatter(k_means_means[i][0], k_means_means[i][1])
    axs2.scatter(gmm_means[i][0], gmm_means[i][1])

plt.show()
