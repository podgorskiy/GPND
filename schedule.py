from save_to_csv import save_results
import logging
import sys
import multiprocessing
import os


full_run = False

percentages = [10, 20, 30, 40, 50]

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


results = []
mul = 0.25

settings = []

gpu_count = 4 #  torch.cuda.device_count()

cpu_count = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(max(1, int(cpu_count / gpu_count)))


def init(queue):
    global idx
    idx = queue.get()


def f(setting):
    global idx
    import torch
    import train_AAE
    import novelty_detector
    # train_AAE.main(fold, [i], i, 10)
    print("Running on GPU: %d" % idx)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(idx)
    device = torch.cuda.current_device()
    print("Running on ", torch.cuda.get_device_name(device))

    res = novelty_detector.main(setting['fold'], [setting['digit']], setting['digit'], 10, mul)
    return res


ids = range(gpu_count)
manager = multiprocessing.Manager()
idQueue = manager.Queue()

for i in ids:
    idQueue.put(i)

for fold in range(5 if full_run else 1):
    for i in range(0, 10):
        settings.append(dict(fold=fold, digit=i))

p = multiprocessing.Pool(gpu_count, init, (idQueue,))

results = p.map(f, settings)

save_results(results, "results_new3.csv")
