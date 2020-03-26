from select import epoll

import torch
from utils import jacobian
import numpy as np


if __name__ == "__main__":

    x = torch.nn.Parameter(torch.tensor([[1, 1, 1, 1, 4]], dtype=torch.float32))

    def func(x):
        return (x * x)[:, :3]

    y = func(x)

    J = jacobian.compute_jacobian_autograd(x, y)

    J2 = jacobian.compute_jacobian_using_finite_differences_v3(x, func, epsilon=1e-3)

    print(J)
    print(J2)

    J = J2

    for i in range(x.shape[0]):
        u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
        logD = np.sum(np.log(np.abs(s)))  # | \mathrm{det} S^{-1} |
        print(np.exp(logD))

        J_1 = np.linalg.pinv(J[i, :, :])
        u, s, vh = np.linalg.svd(J_1, full_matrices=False)
        print(np.prod(1.0 / s))
