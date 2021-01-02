# -*- coding: utf-8 -*-
import torch
from torch import nn
import math


class Polynomial(nn.Module):

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(()))
        self.b = nn.Parameter(torch.randn(()))
        self.c = nn.Parameter(torch.randn(()))
        self.d = nn.Parameter(torch.randn(()))

    def forward(self, x):
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    def string(self):
        return f'y = {self.a.item()} + {self.b.item()} * x + {self.c.item()} * x^2 + {self.d.item()} * x^3'


x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

model = Polynomial()
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

for i in range(2000):
    y_hat = model(x)
    loss = loss_fn(y_hat, y)

    optimizer.zero_grad()
    loss.backward()

    with torch.no_grad():
        optimizer.step()

    if (i + 1) % 100 == 0:
        print(i, loss.item())

print(model.string())