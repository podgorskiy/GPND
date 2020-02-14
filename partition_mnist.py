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

import dlutils
import random
import pickle
from defaults import get_cfg_defaults
import numpy as np
from os import path
from scipy import misc
import logging


def get_mnist():
    dlutils.download.mnist()
    mnist = dlutils.reader.Mnist('mnist', train=True, test=True).items

    images = [x[1] for x in mnist]
    labels = [x[0] for x in mnist]

    images = np.asarray(images)

    assert(images.shape == (70000, 28, 28))

    _images = []
    for im in images:
        im = misc.imresize(im, (32, 32), interp='bilinear')
        _images.append(im)
    images = np.asarray(_images)

    assert(images.shape == (70000, 32, 32))

    #save_image(images[:1024], "data_samples.png", pad_value=0.5, nrow=32)
    #save_image(images.astype(dtype=np.float32).mean(0), "data_mean.png", pad_value=0.5, nrow=1)
    #save_image(images.astype(dtype=np.float32).max(0), "data_max.png", pad_value=0.5, nrow=1)

    return [(l, im) for l, im in zip(labels, images)]


def partition(cfg, logger):
    # to reproduce the same shuffle
    random.seed(0)
    mnist = get_mnist()

    random.shuffle(mnist)

    folds = cfg.DATASET.FOLDS_COUNT

    class_bins = {}

    for x in mnist:
        if x[0] not in class_bins:
            class_bins[x[0]] = []
        class_bins[x[0]].append(x)

    mnist_folds = [[] for _ in range(folds)]

    for _class, data in class_bins.items():
        count = len(data)
        logger.info("Class %d count: %d" % (_class, count))

        count_per_fold = count // folds

        for i in range(folds):
            mnist_folds[i] += data[i * count_per_fold: (i + 1) * count_per_fold]

    logger.info("Folds sizes:")
    for i in range(len(mnist_folds)):
        print(len(mnist_folds[i]))

        output = open(path.join(cfg.DATASET.PATH, 'data_fold_%d.pkl' % i), 'wb')
        pickle.dump(mnist_folds[i], output)
        output.close()


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/mnist.yaml')
    cfg.freeze()
    logger = logging.getLogger("logger")
    partition(cfg, logger)
