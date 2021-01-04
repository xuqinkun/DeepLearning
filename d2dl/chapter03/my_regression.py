import torch
import random
import numpy
import time


def synthetic_numbers(w, b, example_size):
    x = torch.normal(0, 1, (example_size, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    example_size = len(features)
    indices = list(range(example_size))
    random.shuffle(indices)
    for i in range(0, example_size, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, example_size)])
        yield features[batch_indices], labels[batch_indices]


def linear(X, w, b):
    return torch.matmul(X, w) + b


def loss(y_hat, y):
    return (y_hat - y) ** 2 / 2


def train(params, lr, batch_size):
    epoch_num = 4
    features, labels = synthetic_numbers(true_w, true_b, 1000)
    w, b = params
    for epoch in range(epoch_num):
        for x, y in data_iter(batch_size, features, labels):
            y_hat = linear(x, w, b)
            l = loss(y_hat, y)
            l.sum().backward()
            w.data.sub_(lr * w.grad / batch_size)
            w.grad.data.zero_()
            b.data.sub_(lr * b.grad / batch_size)
            b.grad.data.zero_()
        with torch.no_grad():
            train_l = loss((linear(features, w, b)), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


def compare_speed(data_size):
    mat1 = torch.randn(data_size, data_size)
    mat2 = torch.randn(data_size, data_size)
    mat3 = numpy.random.randn(data_size, data_size)
    mat4 = numpy.random.randn(data_size, data_size)
    t1 = time.time()
    out1 = torch.matmul(mat1, mat2)
    t2 = time.time()
    out2 = numpy.matmul(mat3, mat4)
    t3 = time.time()
    print(f'torch mul[{data_size}*{data_size}] takes {t2 - t1:f} s')
    print(f'numpy mul[{data_size}*{data_size}] takes {t3 - t2:f} s')


if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2

    # w = torch.normal(0, 0.03, size=(2, 1), requires_grad=True)
    # b = torch.zeros(1, requires_grad=True)
    # train((w, b), 0.01, 10)
    # print(f'error in estimating w: {true_w - w.reshape(true_w.shape)}')
    # print(f'error in estimating b: {true_b - b}')
    compare_speed(1000)