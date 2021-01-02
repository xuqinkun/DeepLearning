import torch
from torch import nn
from torch.functional import F


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        hidden_out = self.hidden(X)
        hidden_active = F.relu(hidden_out)
        return self.out(hidden_active)


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # Add one
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


if __name__ == '__main__':
    X = torch.ones(2, 20)
    # net = MLP()
    # print(net(X))
    # net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    # net = FixedHiddenMLP()
    net = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
    print(net(X))
