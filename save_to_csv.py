import csv
import numpy as np


def save_results(results, filename):
    percentages = list(results[0].keys())
    measures = list(list(results[0].values())[0].keys())

    f = open(filename, 'wt')
    writer = csv.writer(f)

    for m in measures:
        writer.writerow((m,))
        header = ['Percentage %d' % x for x in percentages]
        writer.writerow(header)

        for r in results:
            row = []
            for p in percentages:
                row.append(r[p][m])
            writer.writerow(tuple(row))

    f.close()

    mean_f1 = np.asarray([r[50]['f1'] for r in results]).mean()

    f = open(filename[:-4] + "_%.3f" % mean_f1, 'w')
    f.close()

    print('Mean F1 at 50%%: %.3f' % mean_f1)
