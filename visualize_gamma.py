import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
from scipy.stats import multivariate_normal


import pickle



N = 100

mean = np.zeros(N)
cov = np.eye(N)

X = np.random.multivariate_normal(mean, cov, 10000).T

r = np.stack(X, axis=1)
print(r.shape)

plt.figure(num=None, figsize=(14, 12), dpi=80, facecolor='w', edgecolor='k')


r_norm = np.linalg.norm(r, axis=1)

with open("rlist.pkl", "rb") as file:
    rlist = pickle.load(file)

r_norm = np.asarray(rlist)


counts, bins = np.histogram(r_norm, bins=30, normed=True)


def _r_pdf(x):
    if bins[0] < x < bins[-1]:
        i = np.digitize(x, bins) - 1
        return counts[i]
    if x < bins[0]:
        return counts[0] * x / bins[0]
    return 1e-308


r_pdf = np.vectorize(_r_pdf)

x = np.linspace(0.0, 5.0, 10000)
plt.plot(x, np.log(r_pdf(x)))

g = stats.gamma.fit(r_norm)
plt.plot(x, np.log(stats.gamma.pdf(x, *g)))


x = np.asarray(np.linspace(0.0, 5.0, 10000))
x = np.stack([x] + (N - 1) * [np.zeros(10000)], axis=1)
var = multivariate_normal(mean, cov)
plt.plot(x, np.log(var.pdf(x)))


def func(x):
    n = N
    return math.gamma(n / 2.0) / (2.0 * np.pi ** (n / 2.0) * x ** (n - 1)) * _r_pdf(x)


def func2(x):
    n = N
    return math.gamma(n / 2.0) / (2.0 * np.pi ** (n / 2.0) * x ** (n - 1)) * stats.gamma.pdf(x, *g)


func = np.vectorize(func)
func2 = np.vectorize(func2)

plt.ylim(-100.0, 100.0)

x = np.asarray(np.linspace(0.01, 5.0, 10000))
plt.plot(x, np.log(func(x)))
plt.plot(x, np.log(func2(x)))

plt.show()