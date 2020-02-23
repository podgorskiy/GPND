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
import torch.utils.data
from torchvision.utils import save_image
from net import *
from torch.autograd import Variable
from utils.jacobian import compute_jacobian
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
import matplotlib.cm as cm


def r_pdf(x, bins, counts):
    if bins[0] < x < bins[-1]:
        i = np.digitize(x, bins) - 1
        return max(counts[i], 1e-308)
    return 1e-300


def extract_statistics(cfg, train_set, E, G):
    zlist = []
    rlist = []

    data_loader = make_dataloader(train_set, cfg.TEST.BATCH_SIZE, torch.cuda.current_device())

    for label, x in data_loader:
        x = x.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS)
        z = E(x.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS))
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
    zcounts1, zbin_edges1 = np.histogram(zlist[:, 0], bins=30, normed=True)
    zcounts2, zbin_edges2 = np.histogram(zlist[:, 1], bins=30, normed=True)

    if cfg.MAKE_PLOTS:
        plt.plot(bin_edges[1:], counts, linewidth=2)
        save_plot(r"Distance, $\left \|\| I - \hat{I} \right \|\|$",
                  'Probability density',
                  r"PDF of distance for reconstruction error, $p\left(\left \|\| I - \hat{I} \right \|\| \right)$",
                  'reconstruction_error.pdf')

    for i in range(cfg.MODEL.LATENT_SIZE):
        plt.hist(zlist[:, i], bins='auto', histtype='step')

    if cfg.MAKE_PLOTS:
        save_plot(r"$z$",
                  'Probability density',
                  r"PDF of embeding $p\left(z \right)$",
                  'embedding.pdf')

    def fmin(func, x0, args, disp):
        x0 = [2.0, 0.0, 1.0]
        return scipy.optimize.fmin(func, x0, args, xtol=1e-12, ftol=1e-12, disp=0)

    # gennorm_param = np.zeros([3, cfg.MODEL.LATENT_SIZE])
    # for i in range(cfg.MODEL.LATENT_SIZE):
    #     betta, loc, scale = scipy.stats.gennorm.fit(zlist[:, i], optimizer=fmin)
    #     gennorm_param[0, i] = betta
    #     gennorm_param[1, i] = loc
    #     gennorm_param[2, i] = scale

    return counts, bin_edges, zcounts1, zbin_edges1, zcounts2, zbin_edges2


def main():
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/toy.yaml')
    cfg.freeze()

    logger = logging.getLogger("logger")

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.cuda.current_device()
    print("Running on ", torch.cuda.get_device_name(device))

    train_set, valid_set, test_set = make_datasets(cfg)

    print('Validation set size: %d' % len(valid_set))
    print('Test set size: %d' % len(test_set))

    train_set.shuffle()

    G = Generator(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)
    E = Encoder(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)

    state = torch.load("models/model.pkl")

    G.load_state_dict(state[0])
    E.load_state_dict(state[1])

    G.eval()
    E.eval()

    counts, bin_edges, zcounts1, zbin_edges1, zcounts2, zbin_edges2 = extract_statistics(cfg, train_set, E, G)

    def run_novely_prediction_on_dataset():
        result = []
        gt_novel = []

        include_jacobian = True

        N = 2
        logC = loggamma(N / 2.0) - (N / 2.0) * np.log(2.0 * np.pi)

        def logPe_func(x):
            # p_{\|W^{\perp}\|} (\|w^{\perp}\|)
            # \| w^{\perp} \|}^{m-n}
            return logC - (N - 1) * np.log(x) + np.log(r_pdf(x, bin_edges, counts))

        x = y = np.arange(-2.0, 2.0, 0.02)
        X, Y = np.meshgrid(x, y)

        for _x, _y in zip(X.flatten(), Y.flatten()):
            x = torch.tensor([[_x, _y]], dtype=torch.float32)
            x = Variable(x.data, requires_grad=True)

            z = E(x.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS))
            recon_batch = G(z)

            if include_jacobian:
                r = recon_batch.detach()
                r = Variable(r.data, requires_grad=True)
                z = E(r.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS))
                J = compute_jacobian(r, z)
                J = J.cpu().numpy()

                # with torch.no_grad():
                #     J = torch.zeros(cfg.MODEL.LATENT_SIZE, x.shape[0], 2, requires_grad=False)
                #
                #     J += recon_batch.view(1, -1, 2)
                #
                #     epsilon = 1e-3
                #     for i in range(cfg.MODEL.LATENT_SIZE):
                #         z_onehot = np.zeros([1, 2], dtype=np.float32)
                #         z_onehot[0, i] = epsilon
                #         z_onehot = torch.tensor(z_onehot, dtype=torch.float32)
                #         _z = z + z_onehot
                #         d_recon_batch = G(_z.view(-1, 2))
                #         J[i] -= d_recon_batch.view(-1, 2)
                #
                #     J /= epsilon
                #
                #     J = torch.transpose(J, dim0=0, dim1=1)
                #     J = torch.transpose(J, dim0=1, dim1=2)
                #
                #     J = J.cpu().numpy()

            z = z.cpu().detach().numpy()

            recon_batch = recon_batch.cpu().detach().numpy()
            x = x.cpu().detach().numpy()

            for i in range(x.shape[0]):
                if include_jacobian:
                    u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
                    # logD = np.sum(np.log(np.abs(s)))  # | \mathrm{det} S^{-1} |
                    # logD = np.sum(np.log(np.prod(1.0 / s)))  # | \mathrm{det} S^{-1} |
                    logD = np.log(np.abs((np.prod(s))))
                else:
                    logD = 0

                #p = scipy.stats.gennorm.pdf(z[i], gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
                p1 = r_pdf(z[i][0], zbin_edges1, zcounts1)
                p2 = r_pdf(z[i][1], zbin_edges2, zcounts2)

                logPz = np.sum(np.log([p1, p2]))

                # Sometimes, due to rounding some element in p may be zero resulting in Inf in logPz
                # In this case, just assign some large negative value to make sure that the sample
                # is classified as unknown.
                if not np.isfinite(logPz):
                    logPz = -1000

                distance = np.linalg.norm(x[i].flatten() - recon_batch[i].flatten())

                logPe = logPe_func(distance)

                P = logD

                result.append(P)

        Z = np.asarray(result)
        Z = Z.reshape(X.shape)

        print(dict(vmax=Z.max(), vmin=Z.min()))

        fig, ax = plt.subplots()
        im = ax.imshow(Z, interpolation='bilinear', cmap=cm.gray,
                       origin='lower', extent=[-2.0, 2.0, -2.0, 2.0],
                       vmax=Z.max(), vmin=-10)

        n = 10000
        r_0 = 2.0 * np.pi
        n_turns = 1.5
        r = np.sqrt(np.random.rand(n) * (np.square(n_turns * 2.0 * np.pi + r_0) - r_0 * r_0) + r_0 * r_0)

        t = r - r_0

        x = np.cos(t) * r * 0.1
        y = np.sin(t) * r * 0.1
        ax.scatter(x, y, c='tab:blue', s=1, label='inliers',
                   alpha=0.3, edgecolors='none')

        plt.show()

    def plot_vector_field():
        result_x = []
        result_y = []
        gt_novel = []

        include_jacobian = True

        N = 2
        logC = loggamma(N / 2.0) - (N / 2.0) * np.log(2.0 * np.pi)

        def logPe_func(x):
            # p_{\|W^{\perp}\|} (\|w^{\perp}\|)
            # \| w^{\perp} \|}^{m-n}
            return logC - (N - 1) * np.log(x) + np.log(r_pdf(x, bin_edges, counts))

        x = y = np.arange(-2.0, 2.0, 0.1)
        X, Y = np.meshgrid(x, y)

        for _x, _y in zip(X.flatten(), Y.flatten()):
            x = torch.tensor([[_x, _y]], dtype=torch.float32)
            x = Variable(x.data, requires_grad=True)

            z = E(x.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS))
            recon_batch = G(z)

            result_x.append(recon_batch[0, 0].item())
            result_y.append(recon_batch[0, 1].item())

        result_x = np.asarray(result_x)
        result_y = np.asarray(result_y)

        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_aspect('equal')

        ax.quiver(X.flatten(), Y.flatten(), (result_x - X.flatten()) * 0.1, (result_y - Y.flatten()) * 0.1, units='xy', scale=1)

        #
        # im = ax.imshow(Z, interpolation='bilinear', cmap=cm.gray,
        #                origin='lower', extent=[-2.0, 2.0, -2.0, 2.0],
        #                vmax=Z.max(), vmin=-10)

        n = 10000
        r_0 = 2.0 * np.pi
        n_turns = 1.5
        r = np.sqrt(np.random.rand(n) * (np.square(n_turns * 2.0 * np.pi + r_0) - r_0 * r_0) + r_0 * r_0)

        t = r - r_0

        x = np.cos(t) * r * 0.1
        y = np.sin(t) * r * 0.1
        ax.scatter(x, y, c='tab:blue', s=1, label='inliers',
                   alpha=0.3, edgecolors='none')

        plt.show()

    # run_novely_prediction_on_dataset()
    plot_vector_field()


if __name__ == "__main__":
    main()
