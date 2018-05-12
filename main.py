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


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))

    return BCE + KLD * 200.0


def extract_batch(data, it, batch_size):
    x = numpy2torch(data[it * batch_size:(it + 1) * batch_size, :, :]) / 255.0
    #x.sub_(0.5).div_(0.5)
    return Variable(x)


def train(model, optimizer, train_data, valid_data, batch_size, epoch, train_epoch):
    model.train()
    train_loss = 0
    epoch_start_time = time.time()

    def shuffle(X):
        np.take(X, np.random.permutation(X.shape[0]), axis=0, out=X)

    shuffle(train_data)
    #shuffle(valid_data)

    for it in range(len(train_data) // batch_size):
        x = extract_batch(train_data, it, batch_size).view(-1, 1, 32, 32)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(x)
        loss = loss_function(recon_batch, x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    train_loss /= len(train_data)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, loss: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime,
                                                 train_loss))


    model.eval()
    test_loss = 0
    with torch.no_grad():
        for it in range(len(valid_data) // batch_size):
            x = extract_batch(valid_data, it, batch_size).view(-1, 1, 32, 32)
            recon_batch, mu, logvar = model(x)
            test_loss += loss_function(recon_batch, x, mu, logvar).item()
            if it == 0:
                x = x[:32]
                recon_batch = recon_batch[:32]
                n = x.size(0)
                comparison = torch.cat([x, recon_batch])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(valid_data)
    print('====> Validation set loss: {:.4f}'.format(test_loss))


def main(folding_id, folds=5):
    batch_size = 512
    zsize = 32
    mnist_train = []
    mnist_valid = []

    class_data = json.load(open('class_table.txt'))

    train_classes = class_data[0]["train"]

    for i in range(folds):
        if i != folding_id:
            with open('data_fold_%d.pkl' % i, 'rb') as pkl:
                fold = pickle.load(pkl)
            if len(mnist_valid) == 0:
                mnist_valid = fold
            mnist_train += fold

    #keep only train classes
    mnist_train = [x for x in mnist_train if x[0] in train_classes]
    mnist_valid = [x for x in mnist_valid if x[0] in train_classes]

    random.shuffle(mnist_train)
    random.shuffle(mnist_valid)

    def list_of_pairs_to_numpy(l):
        return np.asarray([x[1] for x in l], np.float32), np.asarray([x[0] for x in l], np.int)

    print("Train set size:", len(mnist_train))

    mnist_train_x, mnist_train_y = list_of_pairs_to_numpy(mnist_train)
    mnist_valid_x, mnist_valid_y = list_of_pairs_to_numpy(mnist_valid)

    model = VAE(zsize).to(device)
    model.cuda()
    model.weight_init(mean=0, std=0.02)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_epoch = 100

    for epoch in range(train_epoch):
        train(model, optimizer, mnist_train_x, mnist_valid_x, batch_size, epoch, train_epoch)
        with torch.no_grad():
            sample = torch.randn(64, zsize).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 32, 32), 'results/sample_' + str(epoch) + '.png')

    print("Training finish!... save training results")
    torch.save(model.state_dict(), "model_s.pkl")

if __name__ == '__main__':
    main(0)
