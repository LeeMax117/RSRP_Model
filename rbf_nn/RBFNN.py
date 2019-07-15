from numpy import *
import pandas as pd
from math import sqrt


# define a drop mapper.
class DropMapper(object):
    curdf = None

    def __init__(self, df):
        self.curdf = df

    def mapper(self, label):
        if self.curdf is not None:
            cnt = 0
            row = self.curdf.iloc[label, 2:]
            for item in row:
                if item > -126.0:
                    cnt += 1
                if item > -110.0 or cnt > 1:
                    return True
            return False
        else:
            return True


def load_data(file_name):
    """
    load data
    :param file_name: data file name.
    :return: features and ground truth.
    """
    # 1 read data and data pre-processing.
    data = pd.read_csv(file_name)
    dropper = DropMapper(data)
    data = data.select(dropper.mapper)
    # 2 split into feature and label data.
    features = data.iloc[:, 2:]
    label = data.iloc[:, :2]

    return array(features), array(label)


def linear(x):
    return x


def hidden_out(features, center, delta):
    """
    rbf activation function.
    :param features: features.
    :param center: center of rbf function.
    :param delta: std square error.
    :return: hidden_output
    """
    m, n = shape(features)
    m1, n1 = shape(center)
    hidden_output = zeros((m, m1), dtype=float)
    for i in range(m):
        for j in range(m1):
            hidden_output[i, j] = exp(-1.0 * (features[i, :]-center[j, :]) * (features[i,:]-center[j,:]).T/(2*delta[0,j]*delta[0,j]))

    return hidden_output


def predict_in(hidden_output, w):
    predicts_in = hidden_output * w
    return predicts_in


def predict_out(predicts_in):
    result = linear(predicts_in)
    return result


def bp_train(features, label, n_hidden, maxcycle, alpha, n_output, tol=1):
    """计算隐含层的输入c
        input:  features(mat):特征
                label(mat):标签
                n_hidden(int):隐含层的节点个数
                maxcycle(int):最大的迭代次数
                alpha(float):学习率
                n_output(int):输出层的节点个数
        output: center(mat):rbf函数中心
                delta(mat):rbf函数扩展常数
                w(mat):隐含层到输出层之间的权重
    """
    m, n = shape(features)
    # 1. 初始化
    center = mat(random.rand(n_hidden, n))
    center = center * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - mat(ones((n_hidden, n))) * (4.0 * sqrt(6) / sqrt(n + n_hidden))
    delta = mat(random.rand(1, n_hidden))
    delta = delta * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - mat(ones((1, n_hidden))) * (4.0 * sqrt(6) / sqrt(n + n_hidden))
    w = mat(random.rand(n_hidden, n_output))
    w = w * (8.0 * sqrt(6) / sqrt(n_hidden + n_output)) - mat(ones((n_hidden, n_output))) * (4.0 * sqrt(6) / sqrt(n_hidden + n_output))

    # 2. 训练
    iters = 0
    while iters <= maxcycle:
        # 2.1 计算正向传播
        hidden_output = hidden_out(features, center, delta)
        output_in = predict_in(hidden_output, w)
        output_out = predict_out(output_in)

        # 2.2 计算误差
        error = mat(label - output_out)
        # 2.3 反向传播
        for j in range(n_hidden):
            sum_1 = 0
            sum_2 = 0
            sum_3 = 0
            for i in range(m):
                sum_1 += error[i, :].T * exp(
                    -1.0 * (features[i] - center[j]) * (features[i] - center[j]).T / (2 * delta[0, j] * delta[0, j])) * (
                                    features[i] - center[j])
                sum_2 += error[i, :].T * exp(
                    -1.0 * (features[i] - center[j]) * (features[i] - center[j]).T / (2 * delta[0, j] * delta[0, j])) * (
                                    features[i] - center[j]) * (features[i] - center[j]).T
                sum_3 += error[i, :].T * exp(
                    -1.0 * (features[i] - center[j]) * (features[i] - center[j]).T / (2 * delta[0, j] * delta[0, j]))
            delta_center = (w[j, :] / (delta[0, j]*delta[0, j])) * sum_1
            delta_delta = (w[j, :] / (delta[0, j] * delta[0, j] * delta[0, j])) * sum_2
            delta_w = sum_3.T

            # 2.4 修正权重和rbf函数中心和扩展常数
            center[j, :] = center[j, :] + alpha * delta_center
            delta[0, j] = delta[0, j] + alpha * delta_delta
            w[j, :] = w[j, :] + alpha * delta_w

        if iters % 2 == 0:
            cost = (1.0 / 2) * getcost(predict_out(predict_in(hidden_out(features, center, delta), w)) - label)
            print("\t-------- iter: ", iters, " ,cost: ", cost)
            if cost < tol:
                break
        iters += 1
    return center, delta, w


def getcost(cost):
    m, _ = shape(cost)
    costs = square(cost)
    return sum(costs) / float(m)


def save_model(center, delta, w):
    savetxt('center.txt', center)
    savetxt('delta.txt', delta)
    savetxt('w', w)


def get_predict(features, center, delta, w):
    return predict_out(predict_in(hidden_out(features, center, delta), w))


if __name__ == '__main__':
    print("--------- 1.load data ------------")
    features, label = load_data('train.csv')
    print("--------- 2.training -------------")
    center, delta, w = bp_train(features, label, 6, 5000, 0.8, 2)
    print("--------- 3.get prediction -------")
    result = get_predict(features, center, delta, w)
    print("--------- 4.save model and result ")
    save_model(center, delta, w)

