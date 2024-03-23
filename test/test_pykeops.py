import pykeops
import torch
from pykeops.torch import Vi, Vj

pykeops.test_numpy_bindings()
pykeops.test_torch_bindings()

pbe_fn = lambda x, y, k: ((Vi(x) - Vj(y)) ** 2).sum().Kmin(k, 1)  # noqa: E731

N = 1000
x = torch.rand(1, N).cuda()
y = torch.rand(1, N).cuda()
for i in range(1, N + 1):
    r = pbe_fn(x[:, :i], y[:, :i], 10 + 1)[:, 1:].sqrt()

print("pykeops precompiling successes!")
