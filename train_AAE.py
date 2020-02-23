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

import torch.utils.data
from defaults import get_cfg_defaults
from torch import optim
from torch.autograd import Variable
import time
import logging
import os
from dataloading import make_datasets, make_dataloader
from net import Generator, Discriminator, Encoder, ZDiscriminator_mergebatch, ZDiscriminator
from utils.tracker import LossTracker
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from custom_adam import LREQAdam
from torch.optim.rmsprop import RMSprop
from torch.optim.adam import Adam


def save_image(x, path):
    fig, ax = plt.subplots()
    x.cpu().numpy()

    ax.scatter(x[:, 0], x[:, 1], c='tab:blue', s=10, label='generations',
               alpha=0.3, edgecolors='none')

    ax.legend()
    ax.grid(True)

    plt.savefig(path)
    plt.close()


def discriminator_logistic_simple_gp(d_result_fake, d_result_real, reals, r1_gamma=2.0):
    # return F.binary_cross_entropy_with_logits(d_result_fake, torch.zeros_like(d_result_fake)) + F.binary_cross_entropy_with_logits(d_result_real, torch.ones_like(d_result_real))

    loss = (F.softplus(d_result_fake) + F.softplus(-d_result_real))

    if r1_gamma != 0.0:
        real_loss = d_result_real.sum()
        real_grads = torch.autograd.grad(real_loss, reals, create_graph=True, retain_graph=True)[0]
        r1_penalty = torch.sum(real_grads.pow(2.0), dim=[1])
        loss = loss + r1_penalty * (r1_gamma * 0.5)
    return loss.mean()


def generator_logistic_non_saturating(d_result_fake):
    #return F.binary_cross_entropy_with_logits(d_result_real, torch.zeros_like(d_result_real))

    return F.softplus(-d_result_fake).mean()


def train():
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/toy.yaml')
    cfg.freeze()
    logger = logging.getLogger("logger")

    zsize = cfg.MODEL.LATENT_SIZE
    output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    train_set, _, _ = make_datasets(cfg)

    logger.info("Train set size: %d" % len(train_set))

    G = Generator(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)

    D = Discriminator(channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)

    E = Encoder(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)

    if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH:
        ZD = ZDiscriminator_mergebatch(zsize, cfg.TRAIN.BATCH_SIZE)
    else:
        ZD = ZDiscriminator(zsize, cfg.TRAIN.BATCH_SIZE)

    lr = 0.001  # cfg.TRAIN.BASE_LEARNING_RATE

    G_optimizer = Adam(G.parameters(), lr=lr)
    D_optimizer = Adam(D.parameters(), lr=lr)
    GE_optimizer = Adam(list(E.parameters()) + list(G.parameters()), lr=lr)
    ZD_optimizer = Adam(ZD.parameters())

    sample = torch.randn(1024, zsize).view(-1, zsize)

    tracker = LossTracker(output_folder=output_folder)

    BCE_loss = nn.BCELoss()

    for epoch in range(cfg.TRAIN.EPOCH_COUNT):
        G.train()
        D.train()
        E.train()
        ZD.train()

        epoch_start_time = time.time()

        data_loader = make_dataloader(train_set, cfg.TRAIN.BATCH_SIZE, torch.cuda.current_device())
        train_set.shuffle()

        if (epoch + 1) % 30 == 0:
            G_optimizer.param_groups[0]['lr'] /= 4
            D_optimizer.param_groups[0]['lr'] /= 4
            GE_optimizer.param_groups[0]['lr'] /= 4
            ZD_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        for y, x in data_loader:
            x = x.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS)

            #############################################

            D.zero_grad()

            D_result_real = D(x).squeeze()

            z = torch.randn((x.shape[0], zsize)).view(-1, zsize)
            z = Variable(z)

            x_fake = G(z).detach()
            D_result_fake = D(x_fake).squeeze()

            D_train_loss = discriminator_logistic_simple_gp(D_result_fake, D_result_real, x)
            D_train_loss.backward()

            D_optimizer.step()

            tracker.update(dict(D=D_train_loss))


            #############################################

            G.zero_grad()

            z = torch.randn((x.shape[0], zsize)).view(-1, zsize)
            z = Variable(z)

            x_fake = G(z)
            D_result_fake = D(x_fake).squeeze()

            G_train_loss = generator_logistic_non_saturating(D_result_fake)

            G_train_loss.backward()
            G_optimizer.step()

            tracker.update(dict(G=G_train_loss))

            #############################################

            ZD.zero_grad()

            z = torch.randn((x.shape[0], zsize)).view(-1, zsize)
            z = z.requires_grad_(True)

            D_result_real = ZD(z).squeeze()

            D_result_fake = ZD(E(x).squeeze().detach()).squeeze()

            ZD_train_loss = discriminator_logistic_simple_gp(D_result_fake, D_result_real, z, 0.0)

            ZD_train_loss.backward()

            ZD_optimizer.step()

            tracker.update(dict(ZD=ZD_train_loss))

            # #############################################

            E.zero_grad()
            G.zero_grad()

            z = E(x)
            x_d = G(z)

            ZD_result_fake = ZD(z.squeeze()).squeeze()
            E_train_loss = generator_logistic_non_saturating(ZD_result_fake)

            Recon_loss = F.mse_loss(x_d, x.detach()) * 2.0

            (Recon_loss + E_train_loss).backward()

            GE_optimizer.step()

            tracker.update(dict(GE=Recon_loss, E=E_train_loss))

            # #############################################

        # comparison = torch.cat([x, x_d])
        # save_image(comparison.cpu(), os.path.join(output_folder, 'reconstruction_' + str(epoch) + '.png'), nrow=x.shape[0])

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        logger.info('[%d/%d] - ptime: %.2f, %s' % ((epoch + 1), cfg.TRAIN.EPOCH_COUNT, per_epoch_ptime, tracker))

        tracker.register_means(epoch)
        tracker.plot()

        with torch.no_grad():
            resultsample = G(sample).cpu()
            save_image(resultsample,
                       os.path.join(output_folder, 'sample_' + str(epoch) + '.png'))

    logger.info("Training finish!... save training results")

    os.makedirs("models", exist_ok=True)

    print("Training finish!... save training results")
    torch.save([G.state_dict(), E.state_dict(), D.state_dict(), ZD.state_dict()], "models/model.pkl")
