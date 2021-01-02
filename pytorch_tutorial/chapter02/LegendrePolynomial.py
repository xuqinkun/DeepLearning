# -*- coding: utf-8 -*-
import torch
import math


class LegendrePolynomial3(torch.autograd.Function):
    """
    通过继承torch.autograd.Function来实现我们的自定义autograd Functions
    并实现前向和后向传播
    """

    @staticmethod
    def forward(ctx, input):
        """
        :param ctx: 上下文对象，用来存储反向计算的信息
        :param input:输入张量
        :return:输出张量
        """
        ctx.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: 包含损失对输出的梯度，即grad_output=-d(loss)/d(y)=-2*(y_pred - y)
        :param grad_output:损失对输入的梯度
        :return:
        """
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input ** 2 - 1)


dtype = torch.float
device = torch.device('cpu')
x = torch.linspace(-math.pi, math.pi, 2000, dtype=dtype, device=device)
y = torch.sin(x)

a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

lr = 5e-6

loss = None
for t in range(2000):
    P3 = LegendrePolynomial3.apply
    y_pred = a + b * P3(c + d * x)
    loss = (y_pred - y).pow(2).sum()
    if (t + 1) % 100 == 0:
        print(t, loss.item())

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

print(loss.sum().item())
print(f'Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)')
