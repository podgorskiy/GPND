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
import matplotlib.pyplot as plt
import scipy.stats

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

#
# def train(model, optimizer, train_data, valid_data, batch_size, epoch, train_epoch):
#     model.train()
#     train_loss = 0
#     epoch_start_time = time.time()
#
#     def shuffle(X):
#         np.take(X, np.random.permutation(X.shape[0]), axis=0, out=X)
#
#     shuffle(train_data)
#     #shuffle(valid_data)
#
#     for it in range(len(train_data) // batch_size):
#         x = extract_batch(train_data, it, batch_size).view(-1, 1, 32, 32)
#
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(x)
#         loss = loss_function(recon_batch, x, mu, logvar)
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()
#
#     train_loss /= len(train_data)
#
#     epoch_end_time = time.time()
#     per_epoch_ptime = epoch_end_time - epoch_start_time
#
#     print('[%d/%d] - ptime: %.2f, loss: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime,
#                                                  train_loss))
#
#
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for it in range(len(valid_data) // batch_size):
#             x = extract_batch(valid_data, it, batch_size).view(-1, 1, 32, 32)
#             recon_batch, mu, logvar = model(x)
#             test_loss += loss_function(recon_batch, x, mu, logvar).item()
#             if it == 0:
#                 x = x[:32]
#                 recon_batch = recon_batch[:32]
#                 n = x.size(0)
#                 comparison = torch.cat([x, recon_batch])
#                 save_image(comparison.cpu(),
#                          'results/reconstruction_' + str(epoch) + '.png', nrow=n)
#
#     test_loss /= len(valid_data)
#     print('====> Validation set loss: {:.4f}'.format(test_loss))


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


def gaussian(x, sigma=1.0, mu=0.0):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


def log_gaussian(x, sigma=1.0, mu=0.0):
    return np.log(gaussian(x, sigma, mu))


def main(folding_id, opennessid, folds=5):
    batch_size = 64
    mnist_train = []
    mnist_valid = []
    z_size = 32

    def shuffle_in_unison(a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b

    class_data = json.load(open('class_table.txt'))

    train_classes = class_data[0]["train"]
    test_classes = class_data[opennessid]["test_target"]

    for i in range(folds):
        if i != folding_id:
            with open('data_fold_%d.pkl' % i, 'rb') as pkl:
                fold = pickle.load(pkl)
            if len(mnist_valid) == 0:
                mnist_valid = fold
            mnist_train += fold

    #keep only train classes
    mnist_train = [x for x in mnist_train if x[0] in train_classes]
    mnist_test = [x for x in mnist_valid if x[0] in test_classes]
    mnist_valid = [x for x in mnist_valid if x[0] in test_classes]

    random.seed(0)
    random.shuffle(mnist_train)
    random.shuffle(mnist_valid)
    random.shuffle(mnist_test)

    def list_of_pairs_to_numpy(l):
        return np.asarray([x[1] for x in l], np.float32), np.asarray([x[0] for x in l], np.int)

    print("Train set size:", len(mnist_train))

    mnist_train_x, mnist_train_y = list_of_pairs_to_numpy(mnist_train)
    mnist_valid_x, mnist_valid_y = list_of_pairs_to_numpy(mnist_valid)
    mnist_test_x, mnist_test_y = list_of_pairs_to_numpy(mnist_test)

    mnist_test_x, mnist_test_y = shuffle_in_unison(mnist_test_x, mnist_test_y)

    model = VAE(z_size, False).to(device)
    model.cuda()

    pretrained_dict = torch.load("model_s.pkl")
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)

    #model.load_state_dict(torch.load("model_s.pkl"))

    sample = torch.randn(64, z_size).to(device)
    sample = model.decode(sample).cpu()
    save_image(sample.view(64, 1, 32, 32), 'sample.png')

    #model.eval()
    test_loss = 0

    if False:
        zlist = []
        rlist = []

        for it in range(len(mnist_train_x) // batch_size):
            x = Variable(extract_batch(mnist_train_x, it, batch_size).view(-1, 32 * 32).data, requires_grad=True)
            recon_batch, z, logvar = model(x.view(-1, 1, 32, 32))
            J = compute_jacobian(x, z)
            J = J.cpu().numpy()

            recon_batch = recon_batch.squeeze().cpu().detach().numpy()
            x = x.squeeze().cpu().detach().numpy()

            for i in range(batch_size):
                u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=True)

                null = vh[z_size:]

                reconstructed_in_null = null * recon_batch[i].flatten()
                input_in_null = null * x[i].flatten()

                distance = np.linalg.norm(input_in_null - reconstructed_in_null)
                rlist.append(distance)

            z = z.cpu().detach().numpy()
            zlist.append(z)
            #reconstruction_error = torch.norm(recon_batch.view(-1, 32 * 32) - x.view(-1, 32 * 32), 2.0, 1).cpu().detach().numpy()

        data = {}
        data['rlist'] = rlist
        data['zlist'] = zlist

        with open('data.pkl', 'wb') as pkl:
            pickle.dump(data, pkl)

    with open('data.pkl', 'rb') as pkl:
        data = pickle.load(pkl)

    rlist = data['rlist']
    zlist = data['zlist']

    # Choose how many bins you want here
    num_bins = 50

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(rlist, bins=num_bins, normed=True)
    #
    # # And finally plot the cdf
    # plt.plot(bin_edges[1:], counts)
    #
    # plt.show()

    # Now find the cdf
    cdf = np.cumsum(counts[::-1])[::-1]
    cdf /= cdf[0]

    def r_cdf(x):
        if x < bin_edges[0]:
            return 1.0
        if x > bin_edges[-1]:
            return 0.0
        for l, r, i in zip(bin_edges[:-1], bin_edges[1:], range(len(cdf))):
            if x > l and x < r:
                lv = cdf[i]
                rv = cdf[i+1]
                v = lv + (rv - lv) * (x - l) / (r - l)
                return v
    #
    # # And finally plot the cdf
    # plt.plot(bin_edges[1:], cdf)
    #
    # plt.show()

    z_mean = np.zeros([z_size])
    z_var = np.zeros([z_size])

    r_mean = 0
    r_var = 0

    for z in zlist:
        z_mean += np.sum(z, 0) / z.shape[0]
    z_mean /= len(zlist)

    for z in zlist:
        z_var += np.sum(np.power(z - z_mean, 2.0), 0) / z.shape[0]
    z_var /= len(zlist)

    z_std = np.power(z_var, 0.5)

    for r in rlist:
        r_mean += r
    r_mean /= len(rlist)

    for r in rlist:
        r_var += np.power(r - r_mean, 2.0)
    r_var /= len(rlist)

    r_std = np.power(r_var, 0.5)

    print("z_mean ", z_mean)
    print("z_std ", z_std)
    print("r_mean ", r_mean)
    print("r_std ", r_std)

    zlist = np.concatenate(zlist)
    # for i in range(32):
    #     plt.hist(zlist[:, i], bins='auto', histtype='step')
    # plt.title("Histogram with 'auto' bins")
    # plt.show()

    gennorm_param = np.zeros([3, z_size])
    for i in range(z_size):
        betta, loc, scale = scipy.stats.gennorm.fit(zlist[:, i])
        gennorm_param[0, i] = betta
        gennorm_param[1, i] = loc
        gennorm_param[2, i] = scale

    for it in range(len(mnist_test_x) // batch_size):
        x = Variable(extract_batch(mnist_test_x, it, batch_size).view(-1, 32 * 32).data, requires_grad=True)
        label = extract_batch_(mnist_test_y, it, batch_size)

        recon_batch, z, logvar = model(x.view(-1, 1, 32, 32))

        save_image(recon_batch.view(-1, 1, 32, 32), 'sample.png', nrow=1)
        J = compute_jacobian(x, z)

        J = J.cpu().numpy()

        z = z.cpu().detach().numpy()

        reconstruction_error = torch.norm(recon_batch.view(-1, 32 * 32) - x.view(-1, 32 * 32), 2.0,
                                          1).cpu().detach().numpy()

        recon_batch_torch = recon_batch
        recon_batch = recon_batch.squeeze().cpu().detach().numpy()
        x_torch = x
        x = x.squeeze().cpu().detach().numpy()

        error = 0

        for i in range(batch_size):
            print(label[i].item() in train_classes)
            u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=True)
            d = np.abs(np.prod(s))
            logD = np.log(d)
            #print(logD)

            #p = gaussian(z[i], z_std, z_mean)
            p = scipy.stats.gennorm.pdf(z[i], gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
            logPz = np.log(np.prod(p))

            #r = reconstruction_error[i]
            #rp = gaussian(r, r_std, r_mean)

            null = vh[z_size:]

            reconstructed_in_null = null * recon_batch[i].flatten()
            input_in_null = null * x[i].flatten()

            distance = np.linalg.norm(input_in_null - reconstructed_in_null)
            #print(r_cdf(distance))
            #print(np.log(r_cdf(distance)))
            logPe = np.log(r_cdf(distance))

            print("%f, %f %f %f" % (logD + logPz + logPe, logD, logPz, logPe))

            # if not label[i].item() in train_classes and p > 40:
            #     save_image(x_torch[i].view(1, 32, 32), 'falsePositive.png', nrow=1)
            #     save_image(recon_batch_torch[i].view(1, 32, 32), 'falsePositiveR.png', nrow=1)

            if (label[i].item() in train_classes) != (logD + logPz + logPe > -20):
                error += 1

        print(100 - 100 * error / batch_size)
        break


if __name__ == '__main__':
    main(0, 4)
