import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
from scipy.stats import multivariate_normal
from scipy.stats import rv_continuous
import pickle

gamma = np.vectorize(math.gamma)


class normal_norm_gen(rv_continuous):
    def _pdf(self, x, N, sigma, beta):
        # norm.pdf(x) = exp(-x**2/2)/sqrt(2*pi)
        # a = (2.0 * np.pi) ** (-N / 2.0) / sigma ** N * np.exp(-1.0 / 2.0 * ((x / sigma) ** 2))

        #alpha = math.sqrt(2)
        #a = (beta / (2.0 * alpha * gamma(1.0 / beta))) ** N * np.exp(- ((x / alpha) ** beta))

        a = (beta / sigma ** (N / 2) * gamma(N / 2.0)) / (np.pi ** (N / 2.0) * gamma(N / 2.0 / beta) * 2.0 ** (N / 2.0 / beta)) * np.exp(- (x ** 2.0) ** beta / 2.0 / sigma ** beta)

        b = gamma(N / 2.0) / (2.0 * np.pi ** (N / 2.0) * x ** (N - 1))
        return a / b

    def _stats(self, a, b, c):
        return [np.inf]*2 + [np.nan]*2

normal_norm = normal_norm_gen()

N = 17

mean = np.zeros(N)
cov = np.eye(N) * 0.1

X = np.random.multivariate_normal(mean, cov, 10000).T

r = np.stack(X, axis=1)
print(r.shape)

plt.figure(num=None, figsize=(14, 12), dpi=80, facecolor='w', edgecolor='k')

r_norm = np.linalg.norm(r, axis=1)

with open("rlist.pkl", "rb") as file:
    rlist = pickle.load(file)

r_norm = np.asarray(rlist)

x_max = max(r_norm)

counts, bins = np.histogram(r_norm, bins=30, normed=True)


def _r_pdf(x):
    if bins[0] < x < bins[-1]:
        i = np.digitize(x, bins) - 1
        return counts[i]
    if x < bins[0]:
        return counts[0] * x / bins[0]
    return 1e-308


r_pdf = np.vectorize(_r_pdf)

x = np.linspace(0.01, x_max, 10000)
plt.plot(x, np.log(r_pdf(x)))

g = normal_norm.fit(r_norm)
print(g)

plt.plot(x, np.log(normal_norm.pdf(x, *g)))


x = np.asarray(np.linspace(0.01, x_max, 10000))
x = np.stack([x] + (N - 1) * [np.zeros(10000)], axis=1)
var = multivariate_normal(mean, cov)
plt.plot(x, np.log(var.pdf(x)))

N = g[0]


def func(x):
    n = N
    return math.gamma(n / 2.0) / (2.0 * np.pi ** (n / 2.0) * x ** (n - 1)) * _r_pdf(x)


def func2(x):
    n = N
    return math.gamma(n / 2.0) / (2.0 * np.pi ** (n / 2.0) * x ** (n - 1)) * normal_norm.pdf(x, *g)


func = np.vectorize(func)
func2 = np.vectorize(func2)

plt.ylim(-40.0, 5.0)

x = np.asarray(np.linspace(0.01, x_max, 10000))
plt.plot(x, np.log(func(x)))
plt.plot(x, np.log(func2(x)))

plt.show()
