# Copyright 2018-2020 Stanislav Pidhorskyi
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

from __future__ import print_function
import numpy as np
import logging
import scipy.optimize
import pickle
from dataloading import make_datasets, make_dataloader, create_set_with_outlier_percentage
from defaults import get_cfg_defaults
from evaluation import get_f1, evaluate
from utils.threshold_search import find_maximum
from utils.save_plot import save_plot
import matplotlib.pyplot as plt
import scipy.stats
from scipy.special import loggamma
from timeit import default_timer as timer
from scipy.optimize import minimize, dual_annealing
from utils.threshold_search import find_maximum_mv, find_maximum_mv_it
import os
import multiprocessing as mp
import sys


_func = None


def worker_init(func):
    global _func
    _func = func


def worker(x):
    return _func(x)


def main(folding_id, inliner_classes, ic, total_classes, mul, folds=5, cfg=None):
    logger = logging.getLogger("logger")

    def compute_threshold_coeffs(y_scores_components, y_true):
        y_scores_components = np.asarray(y_scores_components, dtype=np.float32)

        def evaluate(threshold, beta, alpha):
            coeff = np.asarray([[1.0, beta, alpha, 1]], dtype=np.float32)
            # coeff_a = np.asarray([[1, 1, alpha, 1]], dtype=np.float32)
            # coeff_b = np.asarray([[1, -1, alpha, 1]], dtype=np.float32)
            # mask = y_scores_components[:, 2:3] > beta
            #
            # coeff = np.where(mask, coeff_a, coeff_b)

            y_scores = (y_scores_components * coeff).mean(axis=1)

            y_false = np.logical_not(y_true)

            y = np.greater(y_scores, threshold)
            true_positive = np.sum(np.logical_and(y, y_true))
            false_positive = np.sum(np.logical_and(y, y_false))
            false_negative = np.sum(np.logical_and(np.logical_not(y), y_true))
            return get_f1(true_positive, false_positive, false_negative)

        def func(x):
            threshold, beta, alpha = x
            return evaluate(threshold, beta, alpha)

        # Find initial threshold guess
        def eval(th):
            return evaluate(th, 1.0, 0.2)
        best_th, best_f1 = find_maximum(eval, -1000, 1000, 1e-2)
        logger.info("Initial e: %f best f1: %f" % (best_th, best_f1))

        x = []
        y = []

        plt.figure(num=None, figsize=(8, 6), dpi=180, facecolor='w', edgecolor='k')

        def calc_for_beta(beta):
            beta /= 30.0

            def func_beta(x):
                threshold, alpha = x
                return evaluate(threshold, beta, alpha)

            cmax, vmax = find_maximum_mv(func_beta, [-20.0, 0.0], [200.0, 10.0], xtoll=0.001, ftoll=0.001, verbose=True, n=100, max_iter=3)

            threshold, alpha = cmax

            best_f1 = evaluate(threshold, beta, alpha)
            return beta, best_f1

        current_module = sys.modules[__name__]
        current_module.fff = calc_for_beta

        with mp.Pool(None, initializer=worker_init, initargs=(calc_for_beta, )) as p:
            results = p.map(worker, range(-100, 100))

        x, y = list(zip(*results))

        plt.xlabel('beta')
        plt.ylabel('F1')
        plt.title('F1 vs Beta')
        plt.plot(x, y)
        plt.grid(True)
        plt.savefig("betta.png")
        plt.clf()
        plt.cla()
        plt.close()
        beta = -1

        logger.info("Best e: %f Best beta: %f Best a: %f best f1: %f" % (threshold, beta, alpha, best_f1))
        return threshold, beta, alpha

    def test(y_scores_components, y_true, percentage, threshold, beta, alpha):
        y_scores_components = np.asarray(y_scores_components, dtype=np.float32)

        coeff = np.asarray([[1.0, beta, alpha, 1]], dtype=np.float32)
        # coeff_a = np.asarray([[1, 1, alpha, 1]], dtype=np.float32)
        # coeff_b = np.asarray([[0, 0, alpha, 1]], dtype=np.float32)
        # mask = y_scores_components[:, 2:3] > beta
        #
        # coeff = np.where(mask, coeff_a, coeff_b)

        y_scores = (y_scores_components * coeff).mean(axis=1)

        return evaluate(logger, percentage, inliner_classes, y_scores, threshold, y_true)

    results = {}

    with open(os.path.join(cfg.OUTPUT_FOLDER, "dump/precomputed_%d_%d.pkl" %(folding_id, ic)), mode='rb') as f:
        data = pickle.load(f)
    y_scores_components_v = data['y_scores_components_v']
    y_true_v = data['y_true_v']
    y_scores_components_t = data['y_scores_components_t']
    y_true_t = data['y_true_t']

    p = 50

    threshold, beta, alpha = compute_threshold_coeffs(y_scores_components_v, y_true_v)
    results[p] = test(y_scores_components_t, y_true_t, p, threshold, beta, alpha)

    return results
