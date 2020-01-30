# Copyright 2018 Stanislav Pidhorskyi
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

from __future__ import print_function
import torch.utils.data
from torch import optim
from torchvision.utils import save_image
from net import *
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import json
import pickle
import time
import random
from torch.autograd.gradcheck import zero_gradients
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.stats
import os
from sklearn.metrics import roc_auc_score
from timeit import default_timer as timer

title_size = 16
axis_title_size = 14
ticks_size = 18

power = 2.0

device = torch.device("cuda")
use_cuda = torch.cuda.is_available()

FloatTensor = torch.FloatTensor
IntTensor = torch.IntTensor
LongTensor = torch.LongTensor
torch.set_default_tensor_type('torch.FloatTensor')

if use_cuda:
    device = torch.cuda.current_device()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FloatTensor = torch.cuda.FloatTensor
    IntTensor = torch.cuda.IntTensor
    LongTensor = torch.cuda.LongTensor
    print("Running on ", torch.cuda.get_device_name(device))


def setup(x):
    if use_cuda:
        return x.cuda()
    else:
        return x.cpu()


def numpy2torch(x):
    return setup(torch.from_numpy(x))


def extract_batch(data, it, batch_size):
    x = numpy2torch(data[it * batch_size:(it + 1) * batch_size]) / 255.0
    #x.sub_(0.5).div_(0.5)
    return Variable(x)


def extract_batch_(data, it, batch_size):
    x = data[it * batch_size:(it + 1) * batch_size]
    return x


def compute_jacobian(inputs, output):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad

    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)


def GetF1(true_positive, false_positive, false_negative):
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return 2.0 * precision * recall / (precision + recall)


def main(folding_id, inliner_classes, ic, total_classes, mul, folds=5):
    batch_size = 64
    mnist_train = []
    mnist_valid = []
    z_size = 16

    def shuffle_in_unison(a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b

    outlier_classes = []
    for i in range(total_classes):
        if i not in inliner_classes:
            outlier_classes.append(i)

    for i in range(folds):
        if i != folding_id:
            with open('data_fold_%d.pkl' % i, 'rb') as pkl:
                fold = pickle.load(pkl)
            if len(mnist_valid) == 0:
                mnist_valid = fold
            else:
                mnist_train += fold

    with open('data_fold_%d.pkl' % folding_id, 'rb') as pkl:
        mnist_test = pickle.load(pkl)

    #keep only train classes
    mnist_train = [x for x in mnist_train if x[0] in inliner_classes]

    random.seed(0)
    random.shuffle(mnist_train)

    def list_of_pairs_to_numpy(l):
        return np.asarray([x[1] for x in l], np.float32), np.asarray([x[0] for x in l], np.int)

    print("Train set size:", len(mnist_train))

    mnist_train_x, mnist_train_y = list_of_pairs_to_numpy(mnist_train)

    G = Generator(z_size).to(device)
    E = Encoder(z_size).to(device)
    setup(E)
    setup(G)
    G.eval()
    E.eval()

    G.load_state_dict(torch.load("Gmodel_%d_%d.pkl" %(folding_id, ic)))
    E.load_state_dict(torch.load("Emodel_%d_%d.pkl" %(folding_id, ic)))

    sample = torch.randn(64, z_size).to(device)
    sample = G(sample.view(-1, z_size, 1, 1)).cpu()
    save_image(sample.view(64, 1, 32, 32), 'sample.png')

    if True:
        zlist = []
        rlist = []

        for it in range(len(mnist_train_x) // batch_size):
            x = Variable(extract_batch(mnist_train_x, it, batch_size).view(-1, 32 * 32).data, requires_grad=True)
            z = E(x.view(-1, 1, 32, 32))
            recon_batch = G(z)
            z = z.squeeze()

            recon_batch = recon_batch.squeeze().cpu().detach().numpy()
            x = x.squeeze().cpu().detach().numpy()

            z = z.cpu().detach().numpy()

            for i in range(batch_size):
                distance = np.linalg.norm(x[i].flatten() - recon_batch[i].flatten())
                rlist.append(distance)

            zlist.append(z)

        data = {}
        data['rlist'] = rlist
        data['zlist'] = zlist

        with open('data.pkl', 'wb') as pkl:
            pickle.dump(data, pkl)

    with open('data.pkl', 'rb') as pkl:
        data = pickle.load(pkl)

    rlist = data['rlist']
    zlist = data['zlist']

    counts, bin_edges = np.histogram(rlist, bins=30, normed=True)

    plt.plot(bin_edges[1:], counts, linewidth=2)
    plt.xlabel(r"Distance, $\left \|\| I - \hat{I} \right \|\|$", fontsize=axis_title_size)
    plt.ylabel('Probability density', fontsize=axis_title_size)
    plt.title(r"PDF of distance for reconstruction error, $p\left(\left \|\| I - \hat{I} \right \|\| \right)$", fontsize=title_size)
    plt.grid(True)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.tight_layout(rect=(0.0, 0.0, 1, 0.95))
    plt.savefig('mnist_d%d_randomsearch.pdf' % inliner_classes[0])
    plt.savefig('mnist_d%d_randomsearch.eps' % inliner_classes[0])
    plt.clf()
    plt.cla()
    plt.close()

    def r_pdf(x, bins, count):
        if x < bins[0]:
            return max(count[0], 1e-308)
        if x >= bins[-1]:
            return max(count[-1], 1e-308)
        id = np.digitize(x, bins) - 1
        return max(count[id], 1e-308)

    zlist = np.concatenate(zlist)
    for i in range(z_size):
        plt.hist(zlist[:, i], bins='auto', histtype='step')

    plt.xlabel(r"$z$", fontsize=axis_title_size)
    plt.ylabel('Probability density', fontsize=axis_title_size)
    plt.title(r"PDF of embeding $p\left(z \right)$", fontsize=title_size)
    plt.grid(True)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.tight_layout(rect=(0.0, 0.0, 1, 0.95))
    plt.savefig('mnist_d%d_embeding.pdf' % inliner_classes[0])
    plt.savefig('mnist_d%d_embeding.eps' % inliner_classes[0])
    plt.clf()
    plt.cla()
    plt.close()

    gennorm_param = np.zeros([3, z_size])
    for i in range(z_size):
        betta, loc, scale = scipy.stats.gennorm.fit(zlist[:, i])
        gennorm_param[0, i] = betta
        gennorm_param[1, i] = loc
        gennorm_param[2, i] = scale

    def compute_threshold(mnist_valid, percentage):
        #############################################################################################
        # Searching for threshold on validation set
        random.shuffle(mnist_valid)
        mnist_valid_outlier = [x for x in mnist_valid if x[0] in outlier_classes]
        mnist_valid_inliner = [x for x in mnist_valid if x[0] in inliner_classes]

        inliner_count = len(mnist_valid_inliner)
        outlier_count = inliner_count * percentage // (100 - percentage)

        if len(mnist_valid_outlier) > outlier_count:
            mnist_valid_outlier = mnist_valid_outlier[:outlier_count]
        else:
            outlier_count = len(mnist_valid_outlier)
            inliner_count = outlier_count * (100 - percentage) // percentage
            mnist_valid_inliner = mnist_valid_inliner[:inliner_count]

        _mnist_valid = mnist_valid_outlier + mnist_valid_inliner
        random.shuffle(_mnist_valid)

        mnist_valid_x, mnist_valid_y = list_of_pairs_to_numpy(_mnist_valid)

        result = []
        novel = []

        for it in range(len(mnist_valid_x) // batch_size):
            x = Variable(extract_batch(mnist_valid_x, it, batch_size).view(-1, 32 * 32).data, requires_grad=True)
            label = extract_batch_(mnist_valid_y, it, batch_size)

            z = E(x.view(-1, 1, 32, 32))
            recon_batch = G(z)
            z = z.squeeze()

            #J = compute_jacobian(x, z)
            #J = J.cpu().numpy()
            z = z.cpu().detach().numpy()

            recon_batch = recon_batch.squeeze().cpu().detach().numpy()
            x = x.squeeze().cpu().detach().numpy()

            for i in range(batch_size):
                #u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
                #logD = np.sum(np.log(np.abs(s))) # | \mathrm{det} S^{-1} |

                p = scipy.stats.gennorm.pdf(z[i], gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
                logPz = np.sum(np.log(p))

                # Sometimes, due to rounding some element in p may be zero resulting in Inf in logPz
                # In this case, just assign some large negative value to make sure that the sample 
                # is classified as unknown. 
                if not np.isfinite(logPz):
                    logPz = -1000

                distance = np.linalg.norm(x[i].flatten() - recon_batch[i].flatten())

                logPe = np.log(r_pdf(distance, bin_edges, counts)) # p_{\|W^{\perp}\|} (\|w^{\perp}\|)
                logPe -= np.log(distance) * (32 * 32 - z_size) * mul # \| w^{\perp} \|}^{m-n}

                P = logPz + logPe

                result.append(P)
                novel.append(label[i].item() in inliner_classes)

        result = np.asarray(result, dtype=np.float32)
        novel = np.asarray(novel, dtype=np.float32)

        minP = min(result) - 1
        maxP = max(result) + 1

        best_e = 0
        best_f = 0
        best_e_ = 0
        best_f_ = 0

        not_novel = np.logical_not(novel)

        for e in np.arange(minP, maxP, 0.1):
            y = np.greater(result, e)

            true_positive = np.sum(np.logical_and(y, novel))
            false_positive = np.sum(np.logical_and(y, not_novel))
            false_negative = np.sum(np.logical_and(np.logical_not(y), novel))

            if true_positive > 0:
                f = GetF1(true_positive, false_positive, false_negative)
                if f > best_f:
                    best_f = f
                    best_e = e
                if f >= best_f_:
                    best_f_ = f
                    best_e_ = e

        best_e = (best_e + best_e_) / 2.0

        print("Best e: ", best_e)
        return best_e

    def test(mnist_test, percentage, e):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        random.shuffle(mnist_test)
        mnist_test_outlier = [x for x in mnist_test if x[0] in outlier_classes]
        mnist_test_inliner = [x for x in mnist_test if x[0] in inliner_classes]

        inliner_count = len(mnist_test_inliner)
        outlier_count = inliner_count * percentage // (100 - percentage)

        if len(mnist_test_outlier) > outlier_count:
            mnist_test_outlier = mnist_test_outlier[:outlier_count]
        else:
            outlier_count = len(mnist_test_outlier)
            inliner_count = outlier_count * (100 - percentage) // percentage
            mnist_test_inliner = mnist_test_inliner[:inliner_count]

        mnist_test = mnist_test_outlier + mnist_test_inliner
        random.shuffle(mnist_test)

        mnist_test_x, mnist_test_y = list_of_pairs_to_numpy(mnist_test)

        count = 0

        result = []
        
        #batch_size = 1

        for it in range(len(mnist_test_x) // batch_size):
        
            #start = timer()
            x = Variable(extract_batch(mnist_test_x, it, batch_size).view(-1, 32 * 32).data, requires_grad=True)
            label = extract_batch_(mnist_test_y, it, batch_size)

            z = E(x.view(-1, 1, 32, 32))
            recon_batch = G(z)
            z = z.squeeze()

            #print("Inference: ", timer() - start)
            #start = timer()
            #J = compute_jacobian(x, z)

            #J = J.cpu().numpy()

            z = z.cpu().detach().numpy()
            #print("Compute Jacobian: ", timer() - start)

            recon_batch = recon_batch.squeeze().cpu().detach().numpy()
            x = x.squeeze().cpu().detach().numpy()

            for i in range(batch_size):
                #start = timer()
                #u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
                #print("Compute SVD: ", timer() - start)
                #start = timer()
                #logD = np.sum(np.log(np.abs(s)))

                p = scipy.stats.gennorm.pdf(z[i], gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
                logPz = np.sum(np.log(p))

                # Sometimes, due to rounding some element in p may be zero resulting in Inf in logPz
                # In this case, just assign some large negative value to make sure that the sample 
                # is classified as unknown. 
                if not np.isfinite(logPz):
                    logPz = -1000

                distance = np.linalg.norm(x[i].flatten() - recon_batch[i].flatten())

                logPe = np.log(r_pdf(distance, bin_edges, counts))
                logPe -= np.log(distance) * (32 * 32 - z_size) * mul

                count += 1

                P = logPz + logPe
                #print("Probability density estimation: ", timer() - start)

                if (label[i].item() in inliner_classes) != (P > e):
                    if not label[i].item() in inliner_classes:
                        false_positive += 1
                    if label[i].item() in inliner_classes:
                        false_negative += 1
                else:
                    if label[i].item() in inliner_classes:
                        true_positive += 1
                    else:
                        true_negative += 1

                result.append(((label[i].item() in inliner_classes), P))

        error = 100 * (true_positive + true_negative) / count

        y_true = [x[0] for x in result]
        y_scores = [x[1] for x in result]

        try:
            auc = roc_auc_score(y_true, y_scores)
        except:
            auc = 0

        with open('result_d%d_p%d.pkl' % (inliner_classes[0], percentage), 'wb') as output:
            pickle.dump(result, output)

        print("Percentage ", percentage)
        print("Error ", error)
        f1 = GetF1(true_positive, false_positive, false_negative)
        print("F1 ", GetF1(true_positive, false_positive, false_negative))
        print("AUC ", auc)

        #inliers
        X1 = [x[1] for x in result if x[0]]

        #outliers
        Y1 = [x[1] for x in result if not x[0]]

        minP = min([x[1] for x in result]) - 1
        maxP = max([x[1] for x in result]) + 1

        ##################################################################
        # FPR at TPR 95
        ##################################################################
        fpr95 = 0.0
        clothest_tpr = 1.0
        dist_tpr = 1.0
        for e in np.arange(minP, maxP, 0.2):
            tpr = np.sum(np.greater_equal(X1, e)) / np.float(len(X1))
            fpr = np.sum(np.greater_equal(Y1, e)) / np.float(len(Y1))
            if abs(tpr - 0.95) < dist_tpr:
                dist_tpr = abs(tpr - 0.95)
                clothest_tpr = tpr
                fpr95 = fpr

        print("tpr: ", clothest_tpr)
        print("fpr95: ", fpr95)

        ##################################################################
        # Detection error
        ##################################################################
        error = 1.0
        for e in np.arange(minP, maxP, 0.2):
            tpr = np.sum(np.less(X1, e)) / np.float(len(X1))
            fpr = np.sum(np.greater_equal(Y1, e)) / np.float(len(Y1))
            error = np.minimum(error, (tpr + fpr) / 2.0)

        print("Detection error: ", error)

        ##################################################################
        # AUPR IN
        ##################################################################
        auprin = 0.0
        recallTemp = 1.0
        for e in np.arange(minP, maxP, 0.2):
            tp = np.sum(np.greater_equal(X1, e))
            fp = np.sum(np.greater_equal(Y1, e))
            if tp + fp == 0:
                continue
            precision = tp / (tp + fp)
            recall = tp / np.float(len(X1))
            auprin += (recallTemp-recall)*precision
            recallTemp = recall
        auprin += recall * precision

        print("auprin: ", auprin)


        ##################################################################
        # AUPR OUT
        ##################################################################
        minp, maxP = -maxP, -minP
        X1 = [-x for x in X1]
        Y1 = [-x for x in Y1]
        auprout = 0.0
        recallTemp = 1.0
        for e in np.arange(minP, maxP, 0.2):
            tp = np.sum(np.greater_equal(Y1, e))
            fp = np.sum(np.greater_equal(X1, e))
            if tp + fp == 0:
                continue
            precision = tp / (tp + fp)
            recall = tp / np.float(len(Y1))
            auprout += (recallTemp-recall)*precision
            recallTemp = recall
        auprout += recall * precision

        print("auprout: ", auprout)

        with open(os.path.join("results.txt"), "a") as file:
            file.write(
                "Class: %d\n Percentage: %d\n"
                "Error: %f\n F1: %f\n AUC: %f\nfpr95: %f"
                "\nDetection: %f\nauprin: %f\nauprout: %f\n\n" %
                (inliner_classes[0], percentage, error, f1, auc, fpr95, error, auprin, auprout))

        return auc, f1, fpr95, error, auprin, auprout

    percentages = [10, 20, 30, 40, 50]

    results = {}

    for p in percentages:
        e = compute_threshold(mnist_valid, p)
        results[p] = test(mnist_test, p, e)

    return results

if __name__ == '__main__':
    main(0, [0], 10)
