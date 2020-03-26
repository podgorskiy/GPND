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

        def func_beta(x):
            threshold, alpha = x
            return evaluate(threshold, 0.0, alpha)

        cmax, vmax = find_maximum_mv(func_beta, [-20.0, 0.0], [200.0, 10.0], xtoll=0.001, ftoll=0.001, verbose=True, n=100, max_iter=3)

        threshold, alpha = cmax

        coeff = np.asarray([[1.0, 0.0, alpha, 1]], dtype=np.float32)
        y_scores = (y_scores_components * coeff).mean(axis=1)

        plt.xlim(-15.0, 1.0)
        plt.figure(num=None, figsize=(8, 8), dpi=180, facecolor='w', edgecolor='k')
        plt.scatter(y_scores, y_scores_components[:, 1], marker='o', c=y_true, alpha=.5, s=1)
        plt.xlabel('P(z)+P(error)')
        plt.ylabel('|J^-1|')
        plt.title('Jacobian vs score')
        plt.grid(True)
        plt.savefig("scatter_perrorpz_pjacobbian.png")
        plt.clf()
        plt.cla()
        plt.close()

        plt.xlim(-15.0, 1.0)
        plt.figure(num=None, figsize=(8, 8), dpi=180, facecolor='w', edgecolor='k')
        plt.scatter(y_scores_components[:, 2] * alpha + y_scores_components[:, 3], y_scores_components[:, 0], marker='o', c=y_true, alpha=.5, s=1)
        plt.xlabel('P(error)')
        plt.ylabel('|J^-1|')
        plt.title('Jacobian vs score')
        plt.grid(True)
        plt.savefig("scatter_error_pz.png")
        plt.clf()
        plt.cla()
        plt.close()

        plt.xlim(-15.0, 1.0)
        plt.figure(num=None, figsize=(8, 8), dpi=180, facecolor='w', edgecolor='k')
        plt.scatter(y_scores_components[:, 0], y_scores_components[:, 1], marker='o', c=y_true, alpha=.5, s=1)
        plt.xlabel('P(z)')
        plt.ylabel('|J^-1|')
        plt.title('Jacobian vs score')
        plt.grid(True)
        plt.savefig("scatter_pz_pjacob.png")
        plt.clf()
        plt.cla()
        plt.close()

    with open(os.path.join(cfg.OUTPUT_FOLDER, "dump/precomputed_%d_%d.pkl" %(folding_id, ic)), mode='rb') as f:
        data = pickle.load(f)
    y_scores_components_v = data['y_scores_components_v']
    y_true_v = data['y_true_v']
    y_scores_components_t = data['y_scores_components_t']
    y_true_t = data['y_true_t']

    compute_threshold_coeffs(y_scores_components_v, y_true_v)


logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

classes_count = 10

fold_id = 0
inliner_classes = 3

cfg = get_cfg_defaults()
cfg.merge_from_file('configs/mnist_alpha_tuning.yaml')
cfg.freeze()

main(fold_id, [inliner_classes], inliner_classes, classes_count, None, cfg=cfg)
