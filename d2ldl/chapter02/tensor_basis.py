import torch


# use arange to create a row vector x containing the
# first 12 integers starting with 0
x = torch.arange(12)

# the length along each axis
print(x.shape)

# Total number of elements
x.numel()

# transform our tensor, x,
# from a row vector with shape (12,) to a matrix with shape (3, 4)
#  tensors can automatically work out one dimension given the rest.
#  We invoke this capability by placing -1 for the dimension that we would like tensors to automatically infer.
#  In our case, instead of calling x.reshape(3, 4),
#  we could have equivalently called x.reshape(-1, 4) or x.reshape(3, -1)
X = x.reshape(3, 4)

# all elements set to 0 and a shape of (2, 3, 4)
torch.zeros((2, 3, 4))

torch.ones((2, 3, 4))

# Each of its elements is randomly sampled
# from a standard Gaussian (normal) distribution
# with a mean of 0 and a standard deviation of 1.
torch.randn(3, 4)

# Create tensor from Python list
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation

torch.exp(x)

# concatenate multiple tensors together.
# we concatenate two matrices along rows
# (axis 0, the first element of the shape)
# vs. columns (axis 1, the second element of the shape).
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)

X == Y

# Sum elements in X
X.sum()

# Broadcasting Mechanism
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a + b)

# Indexing and Slicing
print(X[-1], X[1:3])

X[0:2, :] = 12
print(X)

before = id(Y)
Y = Y + X
id(Y) == before

Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

before = id(X)
X += Y
id(X) == before

A = X.numpy()
B = torch.tensor(A)
type(A), type(B)

a = torch.tensor([3.5])
a, a.item(), float(a), int(a)