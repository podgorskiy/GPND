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

import numpy as np
import scipy.stats as st
import scipy.special

__all__ = ['mvgn', 'gmvgn', 'get_integral', 'get_log_integral', 'integral_mvgn', 'log_integral_mvgn',
           'integral_gmvgn', 'log_integral_gmvgn']


def get_integral(x, y, n):
    return scipy.special.gamma(n / 2.0) / (2.0 * np.pi ** (n / 2.0) * x ** (n - 1)) * y


def get_log_integral(x, y, n):
    return scipy.special.loggamma(n / 2.0) - np.log(2.0) - (n / 2.0) * np.log(np.pi) - (n - 1) * np.log(x) + y


class NMVG(st.rv_continuous):
    def _pdf(self, x, n):
        return x ** (n - 1) / (2.0 ** (n / 2.0 - 1.0) * scipy.special.gamma(n / 2.0)) * np.exp(- x ** 2 / 2.0)

    def _logpdf(self, x, n):
        return (n - 1) * np.log(x) - (n / 2.0 - 1.0) * np.log(2.0) - scipy.special.loggamma(n / 2.0) - x ** 2 / 2.0

    def _cdf(self, x, n):
        return 1.0 - scipy.special.gammaincc(n / 2.0, x ** 2 / 2.0) / scipy.special.gamma(n / 2.0)


mvgn = NMVG(a=0.0, b=np.inf, name='NMVG')


class GNMVG(st.rv_continuous):
    def _pdf(self, x, n, beta):
        return beta * x ** (n - 1) / scipy.special.gamma(n / beta) * np.exp(- x ** beta)

    def _logpdf(self, x, n, beta):
        return np.log(beta) + (n - 1) * np.log(x) - scipy.special.loggamma(n / beta) - x ** beta

    def _cdf(self, x, n, beta):
        return 1.0 - scipy.special.gammaincc(n / beta, x ** beta) / scipy.special.gamma(n / beta)


gmvgn = GNMVG(a=0.0, b=np.inf, name='GNMVG')


def integral_mvgn(x, n, sigma):
    y = mvgn.pdf(x, n=n, loc=0, scale=sigma)
    return get_integral(x, y, n)


def log_integral_mvgn(x, n, sigma):
    y = mvgn.logpdf(x, n=n, loc=0, scale=sigma)
    return get_log_integral(x, y, n)


def integral_gmvgn(x, n, sigma, beta):
    y = gmvgn.pdf(x, beta=beta, n=n, loc=0, scale=sigma)
    return get_integral(x, y, n)


def log_integral_gmvgn(x, n, sigma, beta):
    y = gmvgn.logpdf(x, beta=beta, n=n, loc=0, scale=sigma)
    return get_log_integral(x, y, n)


if __name__ == "__main__":
    x = np.linspace(0.001, 3.0, 200)
    n = np.random.rand() * 10
    b = np.random.rand() * 3

    assert np.all(np.abs(mvgn._pdf(x, n) - np.exp(mvgn._logpdf(x, n))) < 1e-15)
    assert np.all(np.abs(gmvgn._pdf(x, n, b) - np.exp(gmvgn._logpdf(x, n, b))) < 1e-15)
    assert np.all(np.abs(integral_mvgn(x, n, 2.0) - np.exp(log_integral_mvgn(x, n, 2.0))) < 1e-18)
