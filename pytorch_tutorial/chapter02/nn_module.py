import torch
import math

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# y是(x, x^2, x^3)的线性函数
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# 在上面的代码中，x.unsqueeze(-1)的形状为(2000, 1)，p的形状为(3,)
# 将应用广播机制来获得形状为(2000,3)的张量

model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6

for t in range(2000):
    y_pred = model(xx)

    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

linear_layer = model[0]
# For linear layer, its parameters are stored as `weight` and `bias`.
print(
    f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + '
    f'{linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
