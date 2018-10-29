import numpy
import numpy.matlib
import matplotlib.pyplot as plt

CLS_NUM = 4
SAMPLES_PER_CLS = 1000
MEANS = [
            [ 1, 1 ], 
            [-1, 1 ], 
            [ 1, -1], 
            [-1, -1]
        ]
SCALES = [
            [0.1, 0.1], 
            [0.2, 0.2], 
            [0.3, 0.3], 
            [0.4, 0.4]
        ]

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

    return [data_set[i] for i in numpy.random.randint(0, len(data_set), cls_num)]

    # step = int(len(data_set) / cls_num)
    # return [data_set[i * step] for i in range(0, cls_num)]

    # return MEANS


def cluster(data_set, cls_num):
    delta = 1
    iter_count = 0
    result = [[] for i in range(0, cls_num)]
    means = select_means(data_set, cls_num)
    while (delta > DELTA and iter_count < ITER_COUNT):
        result = [[] for i in range(0, cls_num)]
        for data in data_set:
            distances = [cal_distance(data, mean)[0, 0] for mean in means]
            cls_index = distances.index(min(distances))
            result[cls_index].append(data)

        new_means = [
            list(numpy.sum(data_cls, axis=0) / (len(data_cls) + 1)) for data_cls in result
        ]
        delta_mat = numpy.mat(new_means) - numpy.mat(means)
        delta = numpy.sum(delta_mat.T * delta_mat)
        means = new_means
        iter_count += 1
    return result, means


clustered_data_set = generate_data(SAMPLES_PER_CLS, MEANS, SCALES)
data_set = numpy.concatenate(clustered_data_set)
result, means = cluster(data_set, CLS_NUM)

print([len(item) for item in result])

fig, axs = plt.subplots(1, 1)
for i in range(0, len(result)):
    data = numpy.transpose(result[i])
    axs.scatter(data[0], data[1])
    axs.scatter(means[i][0], means[i][1])
plt.show()