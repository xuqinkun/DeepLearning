import torch
import math
import time

dtype = torch.float
device = torch.device("cpu")

data_size = 2000

x = torch.linspace(-math.pi, math.pi, data_size, dtype=dtype, device=device)
y = torch.sin(x)

lr = 1e-6

a = torch.randn((), dtype=dtype, device=device, requires_grad=True)
b = torch.randn((), dtype=dtype, device=device, requires_grad=True)
c = torch.randn((), dtype=dtype, device=device, requires_grad=True)
d = torch.randn((), dtype=dtype, device=device, requires_grad=True)

start = time.time()
loss = None
for i in range(data_size):
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    loss = (y_pred - y).pow(2).sum()
    # if (i + 1) % 100 == 0:
    #     print(i + 1, loss)
    # grad_y_pred = (y_pred - y) * 2
    #
    # grad_a = grad_y_pred.sum()
    # grad_b = (grad_y_pred * x).sum()
    # grad_c = (grad_y_pred * x ** 2).sum()
    # grad_d = (grad_y_pred * x ** 3).sum()
    #
    # a -= grad_a.sum() * lr
    # b -= grad_b.sum() * lr
    # c -= grad_c.sum() * lr
    # d -= grad_d.sum() * lr
    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    loss.backward()
    with torch.no_grad():
        a -= a.grad * lr
        b -= b.grad * lr
        c -= c.grad * lr
        d -= d.grad * lr

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Loss = {loss}')
print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
print(f'Elapse: {time.time() - start}')
