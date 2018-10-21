import numpy
import matplotlib.pyplot as plt

# 全局参数

ORDER = 10  # 多项式函数的阶数
SRART = -1  # 数据区间左端点
STOP = 1    # 数据区间右端点
DATA_SCALE = 50 # 数据量
NOISE_SCALE = 0.1   # 噪声服从的高斯分布的标准差
LAMBDA = 1e-7   # 超参数lambda


def get_data(start, stop, data_scale, add_noises=True):
    """
    生成训练数据和测试数据。
    """
    y, X = [], []

    # 根据参数决定是否添加噪声
    if add_noises:
        noises = numpy.random.normal(0, NOISE_SCALE, data_scale)
    else:
        noises = [0]*data_scale

    # 生成x，x此时是一个列表
    x = numpy.arange(start, stop, (stop - start) / data_scale)

    # 生成y，y此时是一个列表
    for i in range(0, data_scale):
        y.append(numpy.sin(x[i] * numpy.pi) + noises[i])

    # 生成X，X此时是一个二维列表
    for xx in x:
        X.append([xx**i for i in range(0, ORDER + 1)])

    return x, y, X


def get_func(w):
    """
    根据得到的w向量构造拟合函数，用于验证。
    """
    def func(x):
        result = 0.0
        for i in range(0, len(w)):
            result += w[i] * (x**i)
        return result[0, 0]
    return func


def analytical_solution(mat_x, mat_y, mat_X, regular=False):
    """
    解析解方案，根据regular参数决定是否加正则项。
    """

    if regular:
        return (mat_X.T * mat_X + LAMBDA * numpy.identity(ORDER + 1)).I * mat_X.T * mat_y
    else:
        return (mat_X.T * mat_X).I * mat_X.T * mat_y


def gradient_descent(mat_x, mat_y, mat_X, regular=False):
    """
    梯度下降方案，根据regular参数决定是否添加正则项。
    """

    GAMMA = 0.01    # 步长系数
    STEP_COUNT = 0  # 迭代次数
    STEP_LENGTH = 1.0   # 迭代步长
    MAX_STEP_COUNT = 1e9    # 最大迭代次数
    MIN_STEP_LENGTH = 1e-9  # 最小迭代步长

    mat_w = numpy.asmatrix([0.0] * (ORDER + 1)).T   # w的初始值为零向量

    # 根据传入的参数决定是否加正则项
    if regular:
        derivative = lambda w: mat_X.T * mat_X * w - mat_X.T * mat_y + LAMBDA * w
    else:
        derivative = lambda w: mat_X.T * mat_X * w - mat_X.T * mat_y
    
    # 当迭代步长小于某个值或迭代次数大于某个值时停止迭代
    while (STEP_LENGTH > MIN_STEP_LENGTH) and (STEP_COUNT < MAX_STEP_COUNT):
        prev_mat_w = mat_w
        mat_w = prev_mat_w - GAMMA * derivative(prev_mat_w)
        STEP_LENGTH = (prev_mat_w - mat_w).T * (prev_mat_w - mat_w)
        STEP_COUNT += 1

    return mat_w


def conjugate_gradient(mat_x, mat_y, mat_X, regular=False):
    """
    共轭梯度方案，根据regular参数决定是否添加正则项。
    """
    STEP_COUNT = 0  # 迭代次数
    MAX_STEP_COUNT = 1e10   # 最大迭代次数
    MIN_RTR = 1e-9  # 最小误差值，为r的内积

    p = r = mat_X.T * mat_y # 起始时p0和r0都为X'y
    rTr = r.T * r
    mat_w = numpy.asmatrix([0.0] * (ORDER + 1)).T   # w的初始值为零向量

    # 根据参数决定是否加正则项
    if regular:
        A = mat_X.T * mat_X + LAMBDA * numpy.identity(ORDER + 1)
    else:
        A = (mat_X.T * mat_X)

    # 当误差值小于某个值或迭代次数大于某个值时停止迭代
    while (rTr > MIN_RTR) and (STEP_COUNT < MAX_STEP_COUNT):
        a = rTr / (p.T * A * p)
        mat_w = mat_w + p * a

        r = r - A * p * a
        b = (r.T * r) / rTr
        p = r + p * b
        rTr = r.T * r
        STEP_COUNT += 1
        
    return mat_w

# 生成训练数据
x, y, X = get_data(SRART, STOP, DATA_SCALE)
mat_x, mat_y, mat_X = numpy.mat(x).T, numpy.mat(y).T, numpy.mat(X)

# 使用多种方案计算w

# 解析解，无正则项
mat_w_analytical = analytical_solution(mat_x, mat_y, mat_X)
func_analytical = get_func(mat_w_analytical)

# 解析解，有正则项
mat_w_analytical_regular = analytical_solution(mat_x, mat_y, mat_X, True)
func_analytical_regular = get_func(mat_w_analytical_regular)

# 梯度下降法，无正则项
mat_w_gradient_decent = gradient_descent(mat_x, mat_y, mat_X)
func_gradient_decent = get_func(mat_w_gradient_decent)

# 梯度下降法，有正则项
mat_w_gradient_decent_regular = gradient_descent(mat_x, mat_y, mat_X, True)
func_gradient_decent_regular = get_func(mat_w_gradient_decent_regular)

# 共轭梯度法，无正则项
mat_w_conjugate_gradient = conjugate_gradient(mat_x, mat_y, mat_X)
func_conjugate_gradient = get_func(mat_w_conjugate_gradient)

# 共轭梯度法，有正则项
mat_w_conjugate_gradient_regular = conjugate_gradient(mat_x, mat_y, mat_X, True)
func_conjugate_gradient_regular = get_func(mat_w_conjugate_gradient_regular)

# 绘制图像
fig, ((ax11, ax12, ax13, ax14), (ax21, ax22, ax23, ax24), (ax31, ax32, ax33, ax34)) = plt.subplots(3, 4)

# 首先绘制拟合曲线与训练数据曲线的对比

# 第一行，第一列，解析解，无正则项
ax11.plot(x, y)
ax11.plot(x, [func_analytical(xx) for xx in x ])

# 第一行，第三列，解析解，有正则项
ax13.plot(x, y)
ax13.plot(x, [func_analytical_regular(xx) for xx in x ])

# 第二行，第一列，梯度下降法，无正则项
ax21.plot(x, y)
ax21.plot(x, [func_gradient_decent(xx) for xx in x ])

# 第二行，第三列，梯度下降法，有正则项
ax23.plot(x, y)
ax23.plot(x, [func_gradient_decent_regular(xx) for xx in x ])

# 第三行，第一列，共轭梯度法，无正则项
ax31.plot(x, y)
ax31.plot(x, [func_conjugate_gradient(xx) for xx in x ])

# 第三行，第三列，共轭梯度法，有正则项
ax33.plot(x, y)
ax33.plot(x, [func_conjugate_gradient_regular(xx) for xx in x ])

# 生成真实曲线的数据
test_x, test_y, test_X = get_data(SRART, STOP, DATA_SCALE, False)

# 接下来绘制拟合曲线与真实曲线的对比

# 第一行，第二列，解析解，无正则项
ax12.plot(test_x, test_y)
ax12.plot(test_x, [func_analytical(xx) for xx in test_x ])

# 第一行，第四列，解析解，有正则项
ax14.plot(test_x, test_y)
ax14.plot(test_x, [func_analytical_regular(xx) for xx in test_x ])

# 第二行，第二列，梯度下降法，无正则项
ax22.plot(test_x, test_y)
ax22.plot(test_x, [func_gradient_decent(xx) for xx in test_x ])

# 第二行，第四列，梯度下降法，有正则项
ax24.plot(test_x, test_y)
ax24.plot(test_x, [func_gradient_decent_regular(xx) for xx in test_x ])

# 第三行，第二列，共轭梯度法，无正则项
ax32.plot(test_x, test_y)
ax32.plot(test_x, [func_conjugate_gradient(xx) for xx in test_x ])

# 第三行，第四列，共轭梯度法，有正则项
ax34.plot(test_x, test_y)
ax34.plot(test_x, [func_conjugate_gradient_regular(xx) for xx in test_x ])

plt.show()
