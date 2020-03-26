import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
import pickle
import normed_mv_gaussiun
from normed_mv_gaussiun import *

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')


with open("rlist_%s.pkl" % ("_".join([str(x) for x in [0]])), "rb") as f:
    data = pickle.load(f)

# Plot for comparison
plt.figure(figsize=(12, 8))

counts, bins = np.histogram(data, bins=100, normed=True)


def r_pdf(x, bins, counts):
    i = np.digitize(x, bins) - 1
    count = counts[i].clip(1e-308, 1e308)
    mask = x < bins[0]
    return np.where(mask, (counts[0] * x / bins[0]).clip(1e-308, 1e308), count)


# dist = st.genlogistic  # mvgn
dist = gmvgn

# Find best fit distribution
params = dist.fit(data, floc=0)

print(params)


# Display
x = np.linspace(0.1, 5.0, 10000)

plt.figure(figsize=(12, 8))
plt.plot(x, r_pdf(x, bins, counts))

plt.plot(x, dist.pdf(x, *params))

param_names = (dist.shapes + ', loc, scale').split(', ') if dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k, v in zip(param_names, params)])
dist_str = '{}({})'.format(dist.name, param_str)

plt.gca().set_title(u'with best fit distribution \n' + dist_str)

plt.show()

# Integral

n = 29.93

plt.figure(figsize=(12, 8))

plt.plot(x, get_log_integral(x, np.log(r_pdf(x, bins, counts)), n))

plt.plot(x, get_log_integral(x, dist.logpdf(x, *params), n))

param_names = (dist.shapes + ', loc, scale').split(', ') if dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k, v in zip(param_names, params)])
dist_str = '{}({})'.format(dist.name, param_str)

plt.gca().set_title(u'with best fit distribution \n' + dist_str)

plt.show()
