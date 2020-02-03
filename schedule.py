import train_AAE
import novelty_detector
from save_to_csv import save_results
import logging
import sys


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

for fold in range(5 if full_run else 1):
    for i in range(0, 10):
        # train_AAE.main(fold, [i], i, 10)
        
        print("All")
        res = novelty_detector.main(fold, [i], i, 10, mul)

        results.append(res)

        save_results(results, "results_new3.csv")
