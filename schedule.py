from save_to_csv import save_results
import logging
import sys
import torch
import utils.multiprocessing


full_run = True

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


mul = 1.0


def run():
    import train_AAE
    import novelty_detector

    # train_AAE.train()

    novelty_detector.main(mul)
    # return res


torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.set_device(0)
device = torch.cuda.current_device()
print("Running on GPU: %d, %s" % (0, torch.cuda.get_device_name(device)))


run()


# save_results(results, "results.csv")
