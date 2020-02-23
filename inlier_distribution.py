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

import numpy as np


def make_sampler(name):
    def spiral_sampler(n):
        r_0 = 2.0 * np.pi
        n_turns = 1.5
        r = np.sqrt(np.random.rand(n) * (np.square(n_turns * 2.0 * np.pi + r_0) - r_0 * r_0) + r_0 * r_0)

        t = r - r_0
        eta = np.random.randn(2, n) * 0.3

        x = np.cos(t) * r + eta[0]
        y = np.sin(t) * r + eta[1]
        return np.stack([x, y], axis=1) *  0.1

    def par_sampler(n):
        scale = 3
        x = scale*(np.random.random_sample((n,))-0.5)
        y = -1 + x*x
        return np.stack([x, y], axis=1)

    def uniform_sampler(n):
        x = np.random.rand(n) * 36 -18
        y = np.random.rand(n) * 36 -18
        return np.stack([x, y], axis=1) *  0.1

    def four_dots(n):
        eta = np.random.randn(2, n) * 0.3

        x = np.random.rand(n) * 2 - 1
        y = np.random.rand(n) * 2 - 1
        x = np.sign(x) + eta[0]
        y = np.sign(y) + eta[1]

        return np.stack([x, y], axis=1)

    if name == 'spiral':
        return spiral_sampler

    if name == 'par':
        return par_sampler

    if name == 'four_dots':
        return four_dots

    if name == 'uniform':
        return uniform_sampler
