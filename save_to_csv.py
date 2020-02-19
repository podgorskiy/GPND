import csv


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

