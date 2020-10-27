import pandas as pd
import numpy as np


def data_process(data):
    x_list, y_list = [], []
    # 将NR替换成 0
    data = data.replace(['NR'], [0.0])
    # astype() 转换为float
    data = np.array(data).astype(float)
    # 将每个月20天数据连成一大行
    month_data = []
    for month in range(12):
        # 每个月的数据
        sub_data = np.empty([18, 20 * 24])
        for day in range(20):
            # 每一天的数据
            sub_data[:, day * 24:(day + 1) * 24] = data[(month * 18 * 20 + day * 18):(month * 18 * 20 + (day + 1) * 18),
                                                   :]
        month_data.append(sub_data)

    # 将每个月中20天，相邻9个小时生成一笔数据，第10个小时的pm2.5值，生成一个label
    for i in range(12):
        sub_data = month_data[i]
        for j in range(20 * 24 - 9):
            # 相邻9小时的数据
            x_list.append(sub_data[:, j:j + 9])
            # 第10小时的 pm2.5
            y_list.append(sub_data[9, j + 9])

    x = np.array(x_list)
    y = np.array(y_list)

    return x, y, month_data


def train(x_train, y_train, times):
    # 定义参数 b，w   b作为w0
    W = np.ones(1 + 9 * 6)
    # 多少笔数据
    n = y_train.size
    # 学习率
    learning_rate = 100
    # 正则项大小
    reg_rate = 0.011

    # 将训练数据转化成 每一笔数据一行，并且前面添加 1，作为b的权值 [[1, ...], [1, ...],...,[1, ...]]
    X = np.empty([n, W.size - 1])
    for i in range(n):
        X[i] = x_train[i][4:10].reshape(1, -1)
    # 添加 1
    X = np.concatenate((np.ones([n, 1]), X), axis=1)
    # data_X = pd.DataFrame(X)
    # data_X.to_csv('data.csv')
    adagrad = 0
    # 正则项的选择矩阵， 去掉bias部分
    reg_mat = np.concatenate((np.array([0]), np.ones([9 * 6, ])), axis=0)

    for t in range(times):
        # 计算梯度 W = X转置.(XW-Y)
        w_1 = np.dot(X.transpose(), X.dot(W) - y_train)
        # 加正则项
        w_1 += reg_rate * W * reg_mat
        # 正则项参数更新
        adagrad += sum(w_1 ** 2) ** 0.5
        # 梯度下降
        W -= learning_rate / adagrad * w_1
        # 每200次迭代输出一次
        if t % 200 == 0:
            loss = 0
            for j in range(n):
                loss += (y_train[j] - X[j].dot(W)) ** 2
            print('After ', t, ' times loss=', loss / n)

    return W


def validate(x_val, y_val, w):
    n = y_val.size
    # 转化成一行，并加一列 1
    X = np.empty([n, w.size - 1])
    for i in range(n):
        X[i] = x_val[i][4:10].reshape(1, -1)
    X = np.concatenate((np.ones([n, 1]), X), axis=1)

    loss = 0
    # 计算loss
    for j in range(n):
        loss += (y_val[j] - X[j].dot(W)) ** 2
    return loss / n


if __name__ == '__main__':

    data = pd.read_csv('data/train.csv', encoding='big5')
    # 去掉前三列
    data = data.iloc[:, 3:]
    [x, y, month_data] = data_process(data)

    # 8:2 cross validation
    x_train = x[:(int)(x.shape[0] * 0.8)]
    y_train = y[:(int)(x.shape[0] * 0.8)]
    x_val = x[(int)(x.shape[0] * 0.8 + 0.5):]
    y_val = y[(int)(y.shape[0] * 0.8 + 0.5):]

    try:
        W = np.load('weight_2.npy')
    except IOError:
        # 迭代次数
        times = 10000
        W = train(x_train, y_train, times)
        np.save('weight_2.npy', W)

    ## 计算在val上的loss  ##
    loss = validate(x_val, y_val, W)
    print('validate loss=', loss)

    ## 在test上进行验证  ##
    # header=None 无表头读入
    data_test = pd.read_csv('./test.csv', header=None, encoding='big5')
    # 去掉前两列
    test = data_test.iloc[:, 2:]
    test = test.replace(['NR'], [0.0])
    # 处理数据
    test = np.array(test).astype(float)
    [n, m] = test.shape
    # 读出参数值
    X_test = np.empty([int(n / 18), 9 * 6])
    for i in range(0, n, 18):
        X_test[int(i / 18), :] = test[i + 4:i + 10, :].reshape(1, -1)

    [n_test, m_test] = X_test.shape
    # 加一列 1
    X_test = np.concatenate((np.ones([n_test, 1]), X_test), axis=1)

    ## 计算预测值  ##
    Y = X_test.dot(W)
    # 预测值写入
    data_test = np.array(data_test)
    data_test = np.concatenate((data_test, np.zeros([n, 1])), axis=1)
    for j in range(0, n, 18):
        data_test[j + 9, 11] = int(Y[int(j / 18)] + 0.5)

    # 保存结果
    data_test = pd.DataFrame(data_test)
    data_test.to_csv('test_res.csv')
