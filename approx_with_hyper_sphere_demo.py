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
import math
from scipy.stats import multivariate_normal
import scipy.stats as st
import scipy.optimize
import scipy.special

sigma_gt = 1.0
n_gt = 3

mean = np.zeros(n_gt)
cov = np.eye(n_gt) * sigma_gt

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


def func(x, n):
    return math.gamma(n / 2.0) / (2.0 * np.pi ** (n / 2.0) * x ** (n - 1)) * _r_pdf(x)


class MVGN(st.rv_continuous):
    def _pdf(self, x, n):
        sigma = 1.0
        return x ** (n - 1) / (2.0 ** (n / 2.0 - 1.0) * sigma * math.gamma(n / 2.0)) * np.exp(
            - x ** 2 / sigma ** 2 / 2.0)

    def _cdf(self, x, n):
        s = 1.0
        return 1.0 - s ** (n - 1) * scipy.special.gammaincc(n / 2.0, x ** 2 / s ** 2 / 2.0) / math.gamma(n / 2.0)


mvgn = MVGN(a=0.0, b=np.inf, name='MVGN')


func = np.vectorize(func, excluded='n')

plt.ylim(0, 0.9)

x = np.asarray(np.linspace(0.01, 5.0, 10000))
plt.plot(x, func(x, n_gt))
x = np.asarray(np.linspace(0.01, 5.0, 10000))
# plt.plot(x, pdf(x, 2.0, 1.0))

n, loc, scale = mvgn.fit(r_norm, floc=0)

print(n, loc, scale)

x = np.asarray(np.linspace(0.01, 5.0, 10000))

y = mvgn.pdf(x, n=n, loc=0, scale=scale)
plt.plot(x, y)

plt.show()
