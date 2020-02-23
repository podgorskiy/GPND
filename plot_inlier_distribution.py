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
import matplotlib.pyplot as plt
import inlier_distribution

fig, ax = plt.subplots()

n = 10000

inlier_sampler = inlier_distribution.make_sampler('spiral')

outlier_sampler = inlier_distribution.make_sampler('uniform')

points = inlier_sampler(n)
outliers = outlier_sampler(n)

x = outliers[:, 0]
y = outliers[:, 1]

ax.scatter(x, y, c='tab:red', s=10, label='outliers',
           alpha=0.3, edgecolors='none')

x = points[:, 0]
y = points[:, 1]

ax.scatter(x, y, c='tab:blue', s=10, label='inliers',
           alpha=0.3, edgecolors='none')

ax.legend()
ax.grid(True)

plt.show()
