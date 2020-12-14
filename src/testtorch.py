import torch
import torch.nn as nn

a = nn.Linear(4, 1)
weight = torch.Tensor([0, 1, 0, 0]).reshape((1, 4))
bias = torch.Tensor([0]).reshape(1)
a.weight = nn.Parameter(weight)
a.bias = nn.Parameter(bias)

b = torch.Tensor([2, 3, 4, 5])
c = a(b)
print(c)
print(c.detach().numpy())