import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
import pickle
import lambda_friendly_map as lfm
import multiprocessing as mp

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

_lock = mp.Lock()


# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, _x = np.histogram(data, range=(-0.2, 0.2), bins=bins, density=True)
    _x = (_x + np.roll(_x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [
        st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.cauchy, st.chi, st.chi2, st.cosine,
        st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f, st.fatiguelife,
        st.fisk, st.foldcauchy, st.foldnorm, st.frechet_r, st.frechet_l, st.genlogistic, st.genpareto, st.gennorm,
        st.genexpon, st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat, st.gompertz,
        st.gumbel_r, st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant,
        st.invgamma, st.invgauss, st.invweibull, st.johnsonsb, st.johnsonsu, st.kstwobign, st.laplace,
        st.levy, st.levy_l, st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke,
        st.nakagami,  st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm,
        st.powernorm, st.rdist, st.reciprocal, st.rayleigh, st.rice, st.recipinvgauss, st.semicircular, st.t, st.triang,
        st.truncexpon, st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min, st.weibull_max,
        st.wrapcauchy,

        # st.nct, st.ksone,
        # st.ncx2, st.ncf,

        # st.truncnorm, st.tukeylambda, st.burr,

        # gumbel_l, cauchy, halfnorm, arcsine, gumbel_r, halfcauchy, anglit, cosine, halflogistic, gilbrat, invgamma,
        # bradford, alpha, hypsecant, gompertz, chi, levy, frechet_r, halfgennorm, laplace, norm, chi2, foldcauchy,
        # exponpow, fatiguelife, maxwell, gengamma, foldnorm, genpareto, invgauss, genhalflogistic, invweibull,
        # kstwobign, dgamma, betaprime, logistic, johnsonsb, levy_l, lomax, rayleigh, gennorm, wrapcauchy, f, exponweib,
        # semicircular, dweibull, powerlaw, frechet_l, gausshyper, wald, loggamma, recipinvgauss, beta, genlogistic,
        # loglaplace, exponnorm, pareto, t, reciprocal, nakagami, truncexpon, weibull_min, fisk, powernorm, mielke,
        # johnsonsu, genexpon, vonmises_line, weibull_max, rice, triang, vonmises, powerlognorm, rdist, pearson3,
        # genextreme, nct, ksone, ncx2, ncf
        # Interrupted lognorm,
        # Interrupted uniform,
        # Interrupted expon
        # Interrupted erlang
        # Interrupted gamma
        #
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    def try_dist(distribution):
        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                print(distribution.name)
                # fit dist to data
                params = distribution.fit(data, floc=0)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(_x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                print("Done with %s, %f" % (distribution.name, float(sse)))

                with _lock:
                    # Plot for comparison
                    plt.figure(figsize=(12, 8))

                    plt.xlim(-0.05, 0.05)
                    plt.ylim(0.0, 100)

                    x = np.linspace(-0.1, 0.1, 10000)
                    plt.plot(x, r_pdf(x))

                    plt.plot(x, pdf)

                    plt.title(u'' + distribution.name)

                    plt.show()
                    plt.close()

                return distribution, params, sse, pdf

        except (Exception, KeyboardInterrupt):
            print('Interrupted %s' % distribution.name)
            return None

    res = lfm.map(try_dist, DISTRIBUTIONS)
    # Estimate distribution parameters from data
    for r in res:
        if r is not None:
            distribution, params, sse, pdf = r

            # identify if this distribution is better
            if best_sse > sse > 0:
                best_distribution = distribution
                best_params = params
                best_sse = sse

    return best_distribution.name, best_params


def make_pdf(dist, params, size=100000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = -0.05
    end = 0.05

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


with open("difflist_%s.pkl" % ("_".join([str(x) for x in [0]])), "rb") as f:
    data = pickle.load(f)
    data = data[:20000]

counts, bins = np.histogram(data, bins=10000, normed=True)


def _r_pdf(x):
    if bins[0] < x < bins[-1]:
        i = np.digitize(x, bins) - 1
        return counts[i]
    if x < bins[0]:
        return counts[0]
    return 1e-308


x = np.linspace(-0.1, 0.1, 10000)
r_pdf = np.vectorize(_r_pdf)

# Find best fit distribution
best_fit_name, best_fit_params = best_fit_distribution(data, 10000, plt.gca())
best_dist = getattr(st, best_fit_name)

# Make PDF with best params
pdf = make_pdf(best_dist, best_fit_params)

# Display
plt.figure(figsize=(12, 8))

plt.xlim(-0.05, 0.05)
plt.ylim(0.0, 100)

ax = pdf.plot(lw=2, label='PDF', legend=True)
plt.plot(x, r_pdf(x))

param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k, v in zip(param_names, best_fit_params)])
dist_str = '{}({})'.format(best_fit_name, param_str)

ax.set_title(u'with best fit distribution \n' + dist_str)

plt.show()
