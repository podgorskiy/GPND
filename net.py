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
import torch
from torch import nn
from torch.nn import functional as F
#import lreq
import torch.nn as lreq


class Generator(nn.Module):
    def __init__(self, z_size, d=128, channels=2):
        super(Generator, self).__init__()
        self.layer1 = lreq.Linear(z_size, d)
        self.layer2 = lreq.Linear(d, d * 2)
        self.layer3 = lreq.Linear(d * 2, d * 4)
        self.layer4 = lreq.Linear(d * 4, channels)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x), 0.2)
        x = F.leaky_relu(self.layer2(x), 0.2)
        x = F.leaky_relu(self.layer3(x), 0.2)
        x = self.layer4(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, d=128, channels=2):
        super(Discriminator, self).__init__()
        self.layer1 = lreq.Linear(channels, d)
        self.layer2 = lreq.Linear(d, d * 2)
        self.layer3 = lreq.Linear(d * 2, d * 4)
        self.layer4 = lreq.Linear(d * 4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x), 0.2)
        x = F.leaky_relu(self.layer2(x), 0.2)
        x = F.leaky_relu(self.layer3(x), 0.2)
        x = self.layer4(x)
        return x


class Encoder(nn.Module):
    def __init__(self, z_size, d=256, channels=2):
        super(Encoder, self).__init__()
        self.conv1 = lreq.Linear(channels, d)
        self.conv2 = lreq.Linear(d, d*2)
        self.conv3 = lreq.Linear(d*2, d*4)
        self.conv4 = lreq.Linear(d * 4, z_size)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.conv4(x)
        return x


class ZDiscriminator(nn.Module):
    def __init__(self, z_size, batchSize, d=128):
        super(ZDiscriminator, self).__init__()
        self.linear1 = lreq.Linear(z_size, d)
        self.linear2 = lreq.Linear(d, d)
        self.linear3 = lreq.Linear(d, 1)

    def forward(self, x):
        x = F.leaky_relu((self.linear1(x)), 0.2)
        x = F.leaky_relu((self.linear2(x)), 0.2)
        x = torch.sigmoid(self.linear3(x))
        return x


class ZDiscriminator_mergebatch(nn.Module):
    def __init__(self, z_size, batchSize, d=128):
        super(ZDiscriminator_mergebatch, self).__init__()
        self.linear1 = lreq.Linear(z_size, d)
        self.linear2 = lreq.Linear(d * batchSize, d)
        self.linear3 = lreq.Linear(d, 1)

    def forward(self, x):
        x = F.leaky_relu((self.linear1(x)), 0.2).view(1, -1) # after the second layer all samples are concatenated
        x = F.leaky_relu((self.linear2(x)), 0.2)
        x = self.linear3(x)
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
