from utils import mnist_reader
from utils.download import download
import random
import math
import pickle
import json

download(directory="mnist", url="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", extract_gz=True)
download(directory="mnist", url="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", extract_gz=True)
download(directory="mnist", url="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", extract_gz=True)
download(directory="mnist", url="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", extract_gz=True)

train_classes_count = 6
total_classes_count = 10

folds = 5

#Randomly pick train classes
all_classes = [x for x in range(total_classes_count)]
random.shuffle(all_classes)
train_classes = all_classes[:train_classes_count]
rest_classes = [x for x in all_classes if x not in train_classes]

print("Openness table:")

with open('class_table.txt', 'w') as outfile:

    table = []

    for i in range(total_classes_count - train_classes_count + 1):
        test_target_classes = train_classes + rest_classes[:i]
        openness = 1.0 - math.sqrt(2 * len(train_classes) / (len(train_classes) + len(test_target_classes)))
        print("\tOpenness: %f" % openness)
        table.append({"train": train_classes, "test_target": test_target_classes})

    json.dump(table, outfile, indent=4)

#Split mnist into 5 folds:
mnist = items_train = mnist_reader.Reader('mnist', train=True, test=True).items
class_bins = {}

random.shuffle(mnist)

for x in mnist:
    if x[0] not in class_bins:
        class_bins[x[0]] = []
    class_bins[x[0]].append(x)

mnist_folds = [[] for _ in range(folds)]

for _class, data in class_bins.items():
    count = len(data)
    print("Class %d count: %d" % (_class, count))

    count_per_fold = count // folds

    for i in range(folds):
        mnist_folds[i] += data[i * count_per_fold: (i + 1) * count_per_fold]


print("Folds sizes:")
for i in range(len(mnist_folds)):
    print(len(mnist_folds[i]))

    output = open('data_fold_%d.pkl' % i, 'wb')
    pickle.dump(mnist_folds[i], output)
    output.close()
