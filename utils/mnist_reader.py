# Copyright 2017 Stanislav Pidhorskyi
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
"""Util for reading MNIST dataset"""

import mmap
import os
import numpy as np
from contextlib import closing
from scipy import misc

class Reader:
    """Read MNIST out of binary batches"""
    def __init__(self, path, items=None, train=True, test=False, make3channel=False):
        self.items = []

        self.__make3channel = make3channel
        self.__path = path
        self.__label_bytes = 1
        height = 28
        width = 28
        self.__image_bytes = height * width
        self.__record_bytes = self.__image_bytes # stride of items in bin file

        if items is not None:
            self.items = items
        else:
            if train:
                self.__read_batch('train-labels-idx1-ubyte', 'train-images-idx3-ubyte', 60000)

            if test:
                self.__read_batch('t10k-labels-idx1-ubyte', 't10k-images-idx3-ubyte', 10000)

    def __read_batch(self, batch_label, batch_images, n):
        """Read MNIST binary batch using mmap"""
        if self.__make3channel:
            ones = np.ones((1, 1, 3), dtype=np.uint8)
        else:
            ones = np.ones((1), dtype=np.uint8)
		
        with open(os.path.join(self.__path, batch_label), 'rb') as f_l:
            with open(os.path.join(self.__path, batch_images), 'rb') as f_i:
                with closing(mmap.mmap(f_l.fileno(), length=0, access=mmap.ACCESS_READ)) as m_l:
                    with closing(mmap.mmap(f_i.fileno(), length=0, access=mmap.ACCESS_READ)) as m_i:
                        for i in range(n):
                            l = m_l[i + 8]
                            try:
                                # Python 2
                                label = ord(l)
                            except TypeError:
                                # Python 3
                                label = l
                            img = np.fromstring(
                                m_i[16 + i * self.__record_bytes
                                  :16 + i * self.__record_bytes + self.__record_bytes], dtype=np.uint8)
                            img = np.reshape(img, (28, 28))
                            img = misc.imresize(img, (32, 32), interp='bilinear')
                            self.items.append((label, img))

    def get_labels(self):
        return [item[0] for item in self.items]

    def get_images(self):
        return [item[1] for item in self.items]
