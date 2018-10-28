import numpy
import numpy.matlib
import matplotlib.pyplot as plt

CLS_NUM = 3
SAMPLES_PER_CLS = 1000
MEANS = [
            [-1, -1], 
            [0, 0], 
            [1, 1]
        ]
SCALES = [
            [0.5, 0.5], 
            [0.5, 0.5], 
            [0.5, 0.5]
        ]

DELTA = 1e-8
ITER_COUNT = 1e8


def generate_data(samples_per_cls=int, means=list, scales=list):
    return numpy.asarray(
        [
            numpy.random.normal(means[i], scales[i], (samples_per_cls, len(means[i])))
            for i in range(0, len(means))
        ]
    )


def cal_distance(x1=list, x2=list):
    x1 = numpy.mat(x1)
    x2 = numpy.mat(x2)
    return (x1 - x2).T * (x1 - x2)


def select_means(data_set, cls_num):
    return data_set[0:cls_num]
    # return [
    #     data_set[0],
    #     data_set[int(len(data_set) / cls_num)],
    #     data_set[int(len(data_set) / cls_num) * 2],
    # ]


def cluster(data_set, cls_num):
    delta = 1
    iter_count = 0
    result = [[], [], []]
    means = select_means(data_set, cls_num)
    while delta > DELTA and iter_count < ITER_COUNT:
        result = [[], [], []]
        for data in data_set:
            distances = [cal_distance(data, mean)[0, 0] for mean in means]
            cls_index = distances.index(min(distances))
            result[cls_index].append(data)

        new_means = [
            list(numpy.sum(data_cls, axis=0) / len(data_cls)) for data_cls in result
        ]
        delta_mat = numpy.mat(new_means) - numpy.mat(means)
        delta = numpy.sum(delta_mat.T * delta_mat)
        means = new_means
        iter_count += 1
    return result


clustered_data_set = generate_data(SAMPLES_PER_CLS, MEANS, SCALES)
data_set = numpy.concatenate(clustered_data_set)
result = cluster(data_set, CLS_NUM)

print(len(result[0]), len(result[1]), len(result[2]))

fig, axs = plt.subplots(1, 1)
for data in result:
    data = numpy.transpose(data)
    axs.scatter(data[0], data[1])
plt.show()