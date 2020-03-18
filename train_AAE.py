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
from torch import optim
from torchvision.utils import save_image
from torch.autograd import Variable
import time
import logging
import os
from dataloading import make_datasets, make_dataloader
from losses import discriminator_logistic_simple_gp, make_losses
from net import Generator, Discriminator, Encoder, ZDiscriminator_mergebatch, ZDiscriminator
from utils.tracker import LossTracker


def train(folding_id, inliner_classes, ic, cfg):
    logger = logging.getLogger("logger")

    zsize = cfg.MODEL.LATENT_SIZE
    output_folder = os.path.join('results_' + str(folding_id) + "_" + "_".join([str(x) for x in inliner_classes]))
    output_folder = os.path.join(cfg.OUTPUT_FOLDER, output_folder)

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    train_set, _, _ = make_datasets(cfg, folding_id, inliner_classes)

    logger.info("Train set size: %d" % len(train_set))

    G = Generator(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS, no_tanh=cfg.LOSSES == 'logistic_gp')
    G.weight_init(mean=0, std=0.02)

    D = Discriminator(channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)
    D.weight_init(mean=0, std=0.02)

    E = Encoder(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)
    E.weight_init(mean=0, std=0.02)

    if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH:
        ZD = ZDiscriminator_mergebatch(zsize, cfg.TRAIN.BATCH_SIZE)
    else:
        ZD = ZDiscriminator(zsize, cfg.TRAIN.BATCH_SIZE)
    ZD.weight_init(mean=0, std=0.02)

    lr = cfg.TRAIN.BASE_LEARNING_RATE

    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    GE_optimizer = optim.Adam(list(E.parameters()) + list(G.parameters()), lr=lr, betas=(0.5, 0.999))
    ZD_optimizer = optim.Adam(ZD.parameters(), lr=lr, betas=(0.5, 0.999))

    discriminator_loss, generator_loss, reconstruction_loss = make_losses(cfg)

    sample = torch.randn(64, zsize).view(-1, zsize, 1, 1)

    tracker = LossTracker(output_folder=output_folder)

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
            x = x.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE)

            #############################################

            D.zero_grad()

            D_result_real = D(x).squeeze()

            z = torch.randn((x.shape[0], zsize)).view(-1, zsize, 1, 1)
            z = Variable(z)

            x_fake = G(z).detach()
            D_result_fake = D(x_fake).squeeze()

            D_train_loss = discriminator_loss(D_result_fake, D_result_real, x)
            D_train_loss.backward()

            D_optimizer.step()

            tracker.update(dict(D=D_train_loss))


            #############################################

            G.zero_grad()

            z = torch.randn((x.shape[0], zsize)).view(-1, zsize, 1, 1)
            z = Variable(z)

            x_fake = G(z)
            D_result_fake = D(x_fake).squeeze()

            G_train_loss = generator_loss(D_result_fake)

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
            E_train_loss = generator_loss(ZD_result_fake)

            Recon_loss = reconstruction_loss(x_d, x) * 2.0

            (Recon_loss + E_train_loss).backward()

            GE_optimizer.step()

            tracker.update(dict(GE=Recon_loss, E=E_train_loss))

            # #############################################

        comparison = torch.cat([x, x_d])
        save_image(comparison.cpu(), os.path.join(output_folder, 'reconstruction_' + str(epoch) + '.png'), nrow=x.shape[0])

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        logger.info('[%d/%d] - ptime: %.2f, %s' % ((epoch + 1), cfg.TRAIN.EPOCH_COUNT, per_epoch_ptime, tracker))

        tracker.register_means(epoch)
        tracker.plot()

        with torch.no_grad():
            resultsample = G(sample).cpu()
            save_image(resultsample.view(64,
                                         cfg.MODEL.INPUT_IMAGE_CHANNELS,
                                         cfg.MODEL.INPUT_IMAGE_SIZE,
                                         cfg.MODEL.INPUT_IMAGE_SIZE),
                       os.path.join(output_folder, 'sample_' + str(epoch) + '.png'))

    logger.info("Training finish!... save training results")

    os.makedirs("models", exist_ok=True)

    print("Training finish!... save training results")
    torch.save(G.state_dict(), os.path.join(output_folder, "models/Gmodel_%d_%d.pkl" %(folding_id, ic)))
    torch.save(E.state_dict(), os.path.join(output_folder, "models/Emodel_%d_%d.pkl" %(folding_id, ic)))
    #torch.save(D.state_dict(), "Dmodel_%d_%d.pkl" %(folding_id, ic))
    #torch.save(ZD.state_dict(), "ZDmodel_%d_%d.pkl" %(folding_id, ic))


