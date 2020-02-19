from save_to_csv import save_results
import logging
import sys
import utils.multiprocessing


full_run = True

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


mul = 0.2

settings = []

classes_count = 10

for fold in range(5 if full_run else 1):
    for i in range(classes_count):
        settings.append(dict(fold=fold, digit=i))


def f(setting):
    import train_AAE
    import novelty_detector

    fold_id = setting['fold']
    inliner_classes = setting['digit']

    train_AAE.train(fold_id, [inliner_classes], inliner_classes)

    res = novelty_detector.main(fold_id, [inliner_classes], inliner_classes, classes_count, mul)
    return res


gpu_count = utils.multiprocessing.get_gpu_count()

results = utils.multiprocessing.map(f, gpu_count, settings)

save_results(results, "results.csv")
