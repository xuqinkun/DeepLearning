import numpy as np
import math
import time

data_size = 2000
x = np.linspace(-math.pi, math.pi, data_size)
y = np.sin(x)

a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()


def func(x):
    return a + b * x + c * x ** 2 + d * x ** 3


lr = 1e-8

start = time.time()
loss = None
for i in range(data_size):
    y_pred = func(x)
    loss = np.square(y_pred - y).sum()
    # if (i + 1) % 1000 == 0:
    #     print(i + 1, loss)

    grad_y_pred = 2.0 * (y_pred - y)

    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    a -= lr * grad_a
    b -= lr * grad_b
    c -= lr * grad_c
    d -= lr * grad_d

print(f'loss={loss}')
print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
print(f'Elapse: {time.time() - start}')
