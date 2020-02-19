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

import matplotlib
import matplotlib.pyplot as plt


title_size = 16
axis_title_size = 14
ticks_size = 18


def save_plot(xlabel, ylabel, title, filename, layout=(0.0, 0.0, 1, 0.95)):
    plt.xlabel(xlabel, fontsize=axis_title_size)
    plt.ylabel(ylabel, fontsize=axis_title_size)
    plt.title(title, fontsize=title_size)
    plt.grid(True)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.tight_layout(rect=layout)
    plt.savefig(filename)
    plt.clf()
    plt.cla()
    plt.close()
