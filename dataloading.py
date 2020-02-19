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
from net import *
import pickle
import numpy as np
from os import path
import dlutils
import warnings


class Dataset:
    @staticmethod
    def list_of_pairs_to_numpy(l):
        return np.asarray([x[1] for x in l], np.float32), np.asarray([x[0] for x in l], np.int)

    def __init__(self, data):
        self.x, self.y = Dataset.list_of_pairs_to_numpy(data)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.y[index.start:index.stop], self.x[index.start:index.stop]
        return self.y[index], self.x[index]

    def __len__(self):
        return len(self.y)

    def shuffle(self):
        permutation = np.random.permutation(self.y.shape[0])
        for x in [self.y, self.x]:
            np.take(x, permutation, axis=0, out=x)


def make_datasets(cfg, folding_id, inliner_classes):
    data_train = []
    data_valid = []

    for i in range(cfg.DATASET.FOLDS_COUNT):
        if i != folding_id:
            with open(path.join(cfg.DATASET.PATH, 'data_fold_%d.pkl' % i), 'rb') as pkl:
                fold = pickle.load(pkl)
            if len(data_valid) == 0:
                data_valid = fold
            else:
                data_train += fold

    outlier_classes = []
    for i in range(cfg.DATASET.TOTAL_CLASS_COUNT):
        if i not in inliner_classes:
            outlier_classes.append(i)

    data_train = [x for x in data_train if x[0] in inliner_classes]

    with open(path.join(cfg.DATASET.PATH, 'data_fold_%d.pkl') % folding_id, 'rb') as pkl:
        data_test = pickle.load(pkl)

    train_set = Dataset(data_train)
    valid_set = Dataset(data_valid)
    test_set = Dataset(data_test)

    return train_set, valid_set, test_set


def make_dataloader(dataset, batch_size, device):
    class BatchCollator(object):
        def __init__(self, device):
            self.device = device

        def __call__(self, batch):
            with torch.no_grad():
                y, x = batch
                x = torch.tensor(x / 255.0, requires_grad=True, dtype=torch.float32, device=self.device)
                y = torch.tensor(y, dtype=torch.int32, device=self.device)
                return y, x

    data_loader = dlutils.batch_provider(dataset, batch_size, BatchCollator(device))
    return data_loader


def create_set_with_outlier_percentage(dataset, inliner_classes, target_percentage, concervative=True):
    np.random.seed(0)
    dataset.shuffle()
    dataset_outlier = [x for x in dataset if x[0] not in inliner_classes]
    dataset_inliner = [x for x in dataset if x[0] in inliner_classes]

    def increase_length(data_list, target_length):
        repeat = (target_length + len(data_list) - 1) // len(data_list)
        data_list = data_list * repeat
        data_list = data_list[:target_length]
        return data_list

    if not concervative:
        inliner_count = len(dataset_inliner)
        outlier_count = inliner_count * target_percentage // (100 - target_percentage)

        if len(dataset_outlier) > outlier_count:
            dataset_outlier = dataset_outlier[:outlier_count]
        else:
            outlier_count = len(dataset_outlier)
            inliner_count = outlier_count * (100 - target_percentage) // target_percentage
            dataset_inliner = dataset_inliner[:inliner_count]
    else:
        inliner_count = len(dataset_inliner)
        outlier_count = len(dataset_outlier)

        current_percentage = outlier_count * 100 / (outlier_count + inliner_count)

        if current_percentage < target_percentage:  # we don't have enought outliers
            outlier_count = int(inliner_count * target_percentage / (100.0 - target_percentage))
            dataset_outlier = increase_length(dataset_outlier, outlier_count)
        else:  # we don't have enought inliers
            inlier_count = int(outlier_count * (100.0 - target_percentage) / target_percentage)
            dataset_inliner = increase_length(dataset_inliner, inlier_count)

    dataset = Dataset(dataset_outlier + dataset_inliner)

    dataset.shuffle()

    # Post checks
    outlier_count = len([1 for x in dataset if x[0] not in inliner_classes])
    inliner_count = len([1 for x in dataset if x[0] in inliner_classes])
    real_percetage = outlier_count * 100.0 / (outlier_count + inliner_count)
    assert abs(real_percetage - target_percentage) < 0.01, "Didn't create dataset with requested percentage of outliers"

    return dataset
