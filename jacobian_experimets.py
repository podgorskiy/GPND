import torch
from utils import jacobian
import numpy as np


if __name__ == "__main__":

    x = torch.nn.Parameter(torch.tensor([[1, 1, 1, 1, 4]], dtype=torch.float32))

    y = (x * x)[:, :3]

    J = jacobian.compute_jacobian(x, y)

    print(J)

    for i in range(x.shape[0]):
        u, s, vh = np.linalg.svd(J[i, :, :].T, full_matrices=True)
        logD = np.sum(np.log(np.abs(s)))  # | \mathrm{det} S^{-1} |
        print(np.exp(logD))

        J_1 = np.linalg.pinv(J[i, :, :].T)
        u, s, vh = np.linalg.svd(J_1, full_matrices=True)
        print(np.prod(1.0 / s))
