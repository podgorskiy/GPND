# Copyright 2018 Stanislav Pidhorskyi
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
import torch.utils.data
import logging
from torchvision.utils import save_image
from net import *
from torch.autograd import Variable
from utils.jacobian import compute_jacobian
import numpy as np
import pickle
import sys
import time
from dataloading import make_datasets, make_dataloader, make_model_name, create_set_with_outlier_percentage
from defaults import get_cfg_defaults
from evaluation import get_f1, evaluate
from utils.threshold_search import find_maximum
from utils.save_plot import save_plot
import matplotlib.pyplot as plt
import scipy.stats
import os
from sklearn.metrics import roc_auc_score
from timeit import default_timer as timer

title_size = 16
axis_title_size = 14
ticks_size = 18


def main(folding_id, inliner_classes, ic, total_classes, mul, folds=5):
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/mnist.yaml')
    cfg.freeze()

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.cuda.current_device()
    print("Running on ", torch.cuda.get_device_name(device))

    train_set, valid_set, test_set = make_datasets(cfg, folding_id, inliner_classes)

    print('Validation set size: %d' % len(valid_set))
    print('Test set size: %d' % len(test_set))

    train_set.shuffle()

    G = Generator(cfg.MODEL.LATENT_SIZE)
    E = Encoder(cfg.MODEL.LATENT_SIZE)
    G.eval()
    E.eval()

    G.load_state_dict(torch.load("Gmodel_%d_%d.pkl" %(folding_id, ic)))
    E.load_state_dict(torch.load("Emodel_%d_%d.pkl" %(folding_id, ic)))

    sample = torch.randn(64, cfg.MODEL.LATENT_SIZE).to(device)
    sample = G(sample.view(-1, cfg.MODEL.LATENT_SIZE, 1, 1)).cpu()
    save_image(sample.view(64, 1, 32, 32), 'sample.png')

    zlist = []
    rlist = []

    data_loader = make_dataloader(cfg, train_set, cfg.TEST.BATCH_SIZE, 0)

    for label, x in data_loader:
        x = x.view(-1, 32 * 32)
        z = E(x.view(-1, 1, 32, 32))
        recon_batch = G(z)
        z = z.squeeze()

        recon_batch = recon_batch.squeeze().cpu().detach().numpy()
        x = x.squeeze().cpu().detach().numpy()

        z = z.cpu().detach().numpy()

        for i in range(x.shape[0]):
            distance = np.linalg.norm(x[i].flatten() - recon_batch[i].flatten())
            rlist.append(distance)

        zlist.append(z)

    zlist = np.concatenate(zlist)

    counts, bin_edges = np.histogram(rlist, bins=30, normed=True)

    if cfg.MAKE_PLOTS:
        plt.plot(bin_edges[1:], counts, linewidth=2)
        save_plot(r"Distance, $\left \|\| I - \hat{I} \right \|\|$",
                  'Probability density',
                  r"PDF of distance for reconstruction error, $p\left(\left \|\| I - \hat{I} \right \|\| \right)$",
                  'mnist_%s_reconstruction_error.pdf' % ("_".join([str(x) for x in inliner_classes])))

    def r_pdf(x, bins, count):
        if x < bins[0]:
            return max(count[0], 1e-308)
        if x >= bins[-1]:
            return max(count[-1], 1e-308)
        id = np.digitize(x, bins) - 1
        return max(count[id], 1e-308)

    for i in range(cfg.MODEL.LATENT_SIZE):
        plt.hist(zlist[:, i], bins='auto', histtype='step')

    if cfg.MAKE_PLOTS:
        save_plot(r"$z$",
                  'Probability density',
                  r"PDF of embeding $p\left(z \right)$",
                  'mnist_%s_embedding.pdf' % ("_".join([str(x) for x in inliner_classes])))

    gennorm_param = np.zeros([3, cfg.MODEL.LATENT_SIZE])
    for i in range(cfg.MODEL.LATENT_SIZE):
        betta, loc, scale = scipy.stats.gennorm.fit(zlist[:, i])
        gennorm_param[0, i] = betta
        gennorm_param[1, i] = loc
        gennorm_param[2, i] = scale

    def compute_threshold(valid_set, percentage):
        valid_set.shuffle()
        dataset = create_set_with_outlier_percentage(valid_set, inliner_classes, percentage)

        result = []
        novel = []

        data_loader = make_dataloader(cfg, dataset, cfg.TEST.BATCH_SIZE, 0)

        for label, x in data_loader:
            x = x.view(-1, 32 * 32)
            x = Variable(x.data, requires_grad=True)

            z = E(x.view(-1, 1, 32, 32))
            recon_batch = G(z)
            z = z.squeeze()

            J = compute_jacobian(x, z)
            J = J.cpu().numpy()
            z = z.cpu().detach().numpy()

            recon_batch = recon_batch.squeeze().cpu().detach().numpy()
            x = x.squeeze().cpu().detach().numpy()

            for i in range(x.shape[0]):
                u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
                logD = np.sum(np.log(np.abs(s))) # | \mathrm{det} S^{-1} |

                p = scipy.stats.gennorm.pdf(z[i], gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
                logPz = np.sum(np.log(p))

                # Sometimes, due to rounding some element in p may be zero resulting in Inf in logPz
                # In this case, just assign some large negative value to make sure that the sample 
                # is classified as unknown. 
                if not np.isfinite(logPz):
                    logPz = -1000

                distance = np.linalg.norm(x[i].flatten() - recon_batch[i].flatten())

                logPe = np.log(r_pdf(distance, bin_edges, counts)) # p_{\|W^{\perp}\|} (\|w^{\perp}\|)
                logPe -= np.log(distance) * (32 * 32 - cfg.MODEL.LATENT_SIZE) * mul # \| w^{\perp} \|}^{m-n}

                P = logD + logPz + logPe

                result.append(P)
                novel.append(label[i].item() in inliner_classes)

        result = np.asarray(result, dtype=np.float32)
        gt_inlier = np.asarray(novel, dtype=np.float32)

        minP = min(result) - 1
        maxP = max(result) + 1
        gt_outlier = np.logical_not(gt_inlier)

        # f_values = []
        # for e in np.arange(minP, maxP, 0.1):
        #     y = np.greater(result, e)
        #     true_positive = np.sum(np.logical_and(y, gt_inlier))
        #     false_positive = np.sum(np.logical_and(y, gt_outlier))
        #     false_negative = np.sum(np.logical_and(np.logical_not(y), gt_inlier))
        #     f = get_f1(true_positive, false_positive, false_negative)
        #     f_values.append(f)
        #
        # plt.plot(np.arange(minP, maxP, 0.1), f_values, '-', lw=2)
        # plt.xticks(fontsize=ticks_size)
        # plt.yticks(fontsize=ticks_size)
        # plt.savefig("f_plot")
        # plt.clf()
        # plt.cla()
        # plt.close()
        # exit()

        def evaluate(e):
            y = np.greater(result, e)
            true_positive = np.sum(np.logical_and(y, gt_inlier))
            false_positive = np.sum(np.logical_and(y, gt_outlier))
            false_negative = np.sum(np.logical_and(np.logical_not(y), gt_inlier))
            return get_f1(true_positive, false_positive, false_negative)

        best_th, best_f1 = find_maximum(evaluate, minP, maxP, 1e-3)

        logger.info("Best e: %f best f1: %f" % (best_th, best_f1))
        return best_th

    def test(test_set, percentage, e):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        test_set.shuffle()
        dataset = create_set_with_outlier_percentage(test_set, inliner_classes, percentage)

        count = 0

        result = []

        test_data_loader = make_dataloader(cfg, dataset, cfg.TEST.BATCH_SIZE, 0)

        for label, x in test_data_loader:
            x = x.view(-1, 32 * 32)
            x = Variable(x.data, requires_grad=True)

            z = E(x.view(-1, 1, 32, 32))
            recon_batch = G(z)
            z = z.squeeze()

            #print("Inference: ", timer() - start)
            #start = timer()
            J = compute_jacobian(x, z)

            J = J.cpu().numpy()

            z = z.cpu().detach().numpy()
            #print("Compute Jacobian: ", timer() - start)

            recon_batch = recon_batch.squeeze().cpu().detach().numpy()
            x = x.squeeze().cpu().detach().numpy()

            for i in range(x.shape[0]):
                #start = timer()
                u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
                #print("Compute SVD: ", timer() - start)
                #start = timer()
                logD = np.sum(np.log(np.abs(s)))

                p = scipy.stats.gennorm.pdf(z[i], gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
                logPz = np.sum(np.log(p))

                # Sometimes, due to rounding some element in p may be zero resulting in Inf in logPz
                # In this case, just assign some large negative value to make sure that the sample 
                # is classified as unknown. 
                if not np.isfinite(logPz):
                    logPz = -1000

                distance = np.linalg.norm(x[i].flatten() - recon_batch[i].flatten())

                logPe = np.log(r_pdf(distance, bin_edges, counts))
                logPe -= np.log(distance) * (32 * 32 - cfg.MODEL.LATENT_SIZE) * mul

                count += 1

                P = logD + logPz + logPe
                #print("Probability density estimation: ", timer() - start)

                if (label[i].item() in inliner_classes) != (P > e):
                    if not label[i].item() in inliner_classes:
                        false_positive += 1
                    if label[i].item() in inliner_classes:
                        false_negative += 1
                else:
                    if label[i].item() in inliner_classes:
                        true_positive += 1
                    else:
                        true_negative += 1

                result.append(((label[i].item() in inliner_classes), P))

        y_true = [x[0] for x in result]
        y_scores = [x[1] for x in result]

        return evaluate(logger, percentage, inliner_classes, y_scores, e, y_true)

    #percentages = [10, 20, 30, 40, 50]
    percentages = [50]

    results = {}

    for p in percentages:
        e = compute_threshold(valid_set, p)
        results[p] = test(test_set, p, e)

    return results


if __name__ == '__main__':
    main(0, [0], 10)
