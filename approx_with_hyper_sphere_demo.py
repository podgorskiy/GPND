# Copyright 2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from normed_mv_gaussiun import *

sigma_gt = 0.5
n_gt = 20

mean = np.zeros(n_gt)
cov = np.eye(n_gt) * sigma_gt * sigma_gt

samples = np.random.multivariate_normal(mean, cov, 10000)

plt.figure(num=None, figsize=(14, 12), dpi=80, facecolor='w', edgecolor='k')

print(samples.shape)

r_norm = np.linalg.norm(samples, axis=1)
counts, bins = np.histogram(r_norm, bins=100, normed=True)


def _r_pdf(x):
    if bins[0] < x < bins[-1]:
        i = np.digitize(x, bins) - 1
        return counts[i]
    if x < bins[0]:
        return counts[0]
    return 1e-308


r_pdf = np.vectorize(_r_pdf)

x = np.linspace(0.0, 5.0, 10000)
plt.plot(x, r_pdf(x))

x = np.asarray(np.linspace(0.0, 5.0, 10000))
x = np.stack([x] + [np.zeros(10000)] * (n_gt - 1), axis=1)
var = multivariate_normal(mean, cov)
plt.plot(x, var.pdf(x))

func = np.vectorize(log_integral_mvgn, excluded=('n', 'sigma'))

# plt.ylim(0, 0.9)

x = np.asarray(np.linspace(0.01, 5.0, 10000))

n, loc, scale = mvgn.fit(r_norm, floc=0)
print(n, scale)

plt.plot(x, np.exp(func(x, n, scale)))
y = mvgn.pdf(x, n=n, loc=0, scale=scale)
plt.plot(x, y)

plt.show()
