from save_to_csv import save_results
import logging
import sys
import utils.multiprocessing
from defaults import get_cfg_defaults
import os

full_run = False

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

if len(sys.argv) > 1:
    cfg_file = 'configs/' + sys.argv[1]
else:
    cfg_file = 'configs/' + 'mnist_alpha_tuning.yaml'  # input("Config file:")

mul = 0.25

settings = []

classes_count = 10

for fold in range(5 if full_run else 1):
    for i in range(classes_count):
        settings.append(dict(fold=fold, digit=i))

cfg = get_cfg_defaults()
cfg.merge_from_file(cfg_file)
cfg.freeze()


def f(setting):
    import train_AAE
    import novelty_detector_dump_for_offline

    fold_id = setting['fold']
    inliner_classes = setting['digit']

    # train_AAE.train(fold_id, [inliner_classes], inliner_classes, cfg=cfg)

    novelty_detector_dump_for_offline.main(fold_id, [inliner_classes], inliner_classes, classes_count, mul, cfg=cfg)


gpu_count = utils.multiprocessing.get_gpu_count()

results = utils.multiprocessing.map(f, gpu_count, settings)
