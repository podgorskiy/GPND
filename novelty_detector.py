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


def main(folding_id, opennessid, folds=5):
    batch_size = 64
    mnist_train = []
    mnist_valid = []


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

    model = VAE(32).to(device)
    model.cuda()
    model.load_state_dict(torch.load("model_s.pkl"))

    sample = torch.randn(64, 32).to(device)
    sample = model.decode(sample).cpu()
    save_image(sample.view(64, 1, 32, 32), 'sample.png')

    #model.eval()
    test_loss = 0

    z_mean = np.zeros([32])
    z_var = np.zeros([32])
    count = 0

    for it in range(len(mnist_train_x) // batch_size):
        x = Variable(extract_batch(mnist_train_x, it, batch_size).view(-1, 32 * 32).data, requires_grad=True)
        recon_batch, z, logvar = model(x.view(-1, 1, 32, 32))
        z = z.cpu().detach().numpy()
        z_mean += np.sum(z, 0)
        z_var += np.sum(np.power(z, 2.0), 0)
        count += 1
    z_var -= np.power(z_mean / count, 2.0)
    z_std = np.power(z_var / (count - 1), 0.5)
    z_mean /= count

    for it in range(len(mnist_test_x) // batch_size):
        x = Variable(extract_batch(mnist_test_x, it, batch_size).view(-1, 32 * 32).data, requires_grad=True)
        label = extract_batch_(mnist_test_y, it, batch_size)

        recon_batch, z, logvar = model(x.view(-1, 1, 32, 32))

        save_image(recon_batch.view(-1, 1, 32, 32), 'sample.png', nrow=1)
        J = compute_jacobian(x, z)

        J = J.cpu().numpy()

        z = z.cpu().detach().numpy()

        for i in range(batch_size):
            print(label[i].item() in train_classes)
            u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
            d = np.prod(s)
            print(np.log(d))
            #print(d)
            #p = gaussian(z[i], z_std, z_mean)
            #f = np.sum(np.log(p))
            #print(f + d)
        break



if __name__ == '__main__':
    main(0, 4)
