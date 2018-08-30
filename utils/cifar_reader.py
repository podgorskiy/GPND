# Copyright 2017-2018 Stanislav Pidhorskyi
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
    """Read CIFAR out of binary batches"""
    def __init__(self, path, train=True, test=False):
        self.items = []

        self.__path = path
        self.__label_bytes = 1
        height = 32
        width = 32
        depth = 3
        self.__image_bytes = height * width * depth
        self.__record_bytes = self.__label_bytes + self.__image_bytes # stride of items in bin file

        if train:
            self.__read_batch('data_batch_1.bin')
            self.__read_batch('data_batch_2.bin')
            self.__read_batch('data_batch_3.bin')
            self.__read_batch('data_batch_4.bin')
            self.__read_batch('data_batch_5.bin')

        if test:
            self.__read_batch('test_batch.bin')

    def __read_batch(self, batch):
        """Read CIFAR binary batch using mmap"""
        with open(os.path.join(self.__path, batch), 'rb') as f:
            with closing(mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)) as m:
                for i in range(10000):
                    l = m[i * self.__record_bytes]
                    try:
                        # Python 2
                        label = ord(l)
                    except TypeError:
                        # Python 3
                        label = l
                    img = np.fromstring(
                        m[i * self.__record_bytes + self.__label_bytes
                          :i * self.__record_bytes + self.__record_bytes], dtype=np.uint8)
                    img = np.reshape(img, (3, 32, 32))
                    self.items.append((label, img))

    def get_labels(self):
        return [item[0] for item in self.items]

    def get_images(self):
        return [item[1] for item in self.items]
