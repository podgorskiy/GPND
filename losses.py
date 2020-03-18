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

import torch.autograd
from torch.nn import functional as F


def discriminator_logistic_simple_gp(d_result_fake, d_result_real, reals, r1_gamma=2.0):
    loss = (F.softplus(d_result_fake) + F.softplus(-d_result_real))

    if r1_gamma != 0.0:
        real_loss = d_result_real.sum()
        real_grads = torch.autograd.grad(real_loss, reals, create_graph=True, retain_graph=True)[0]
        r1_penalty = torch.sum(real_grads.pow(2.0), dim=[1, 2, 3])
        loss = loss + r1_penalty * (r1_gamma * 0.5)
    return loss.mean()


def generator_logistic_non_saturating(d_result_fake):
    return F.softplus(-d_result_fake).mean()


def reconstruction_mse(x, target):
    return F.mse_loss(x, target.detach())


def discriminator_classic(d_result_fake, d_result_real, reals, r1_gamma=2.0):
    return F.binary_cross_entropy_with_logits(d_result_fake, torch.zeros_like(d_result_fake)) +\
           F.binary_cross_entropy_with_logits(d_result_real, torch.ones_like(d_result_real))


def generator_classic(d_result_fake):
    return F.binary_cross_entropy_with_logits(d_result_fake, torch.ones_like(d_result_fake))


def reconstruction_bce(x, target):
    return F.binary_cross_entropy(x, target.detach())


def make_losses(cfg):
    if cfg.LOSSES == 'logistic_gp':
        return discriminator_logistic_simple_gp, generator_logistic_non_saturating, reconstruction_mse

    elif cfg.LOSSES == 'classic':
        return discriminator_classic, generator_classic, reconstruction_bce

    assert False
