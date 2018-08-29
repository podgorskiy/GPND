import train_AAE
import novelty_detector
import csv


full_run = True

percentages = [10, 20, 30, 40, 50]

def save_results(results):
    f = open("results.csv", 'wt')
    writer = csv.writer(f)
    writer.writerow(('F1',))
    writer.writerow(('Percentage 10', 'Percentage 20', 'Percentage 30', 'Percentage 40', 'Percentage 50'))
    maxlength = 0
    for percentage in percentages:
        list = results[percentage]
        maxlength = max(maxlength, len(list))

    for r in range(maxlength):
        row = []
        for percentage in percentages:
            list = results[percentage]
            res_f1 = [f1 for auc, f1, fpr95, error, auprin, auprout in list]
            row.append(res_f1[r] if len(list) > r else '')
        writer.writerow(tuple(row))

    writer.writerow(('AUC',))
    writer.writerow(('Percentage 10', 'Percentage 20', 'Percentage 30', 'Percentage 40', 'Percentage 50'))

    for r in range(maxlength):
        row = []
        for percentage in percentages:
            list = results[percentage]
            res_auc = [auc for auc, f1, fpr95, error, auprin, auprout in list]
            row.append(res_auc[r] if len(list) > r else '')
        writer.writerow(tuple(row))

    writer.writerow(('FPR',))
    writer.writerow(('Percentage 10', 'Percentage 20', 'Percentage 30', 'Percentage 40', 'Percentage 50'))

    for r in range(maxlength):
        row = []
        for percentage in percentages:
            list = results[percentage]
            res_fpr95 = [fpr95 for auc, f1, fpr95, error, auprin, auprout in list]
            row.append(res_fpr95[r] if len(list) > r else '')
        writer.writerow(tuple(row))

    writer.writerow(('error',))
    writer.writerow(('Percentage 10', 'Percentage 20', 'Percentage 30', 'Percentage 40', 'Percentage 50'))

    for r in range(maxlength):
        row = []
        for percentage in percentages:
            list = results[percentage]
            res_error = [error for auc, f1, fpr95, error, auprin, auprout in list]
            row.append(res_error[r] if len(list) > r else '')
        writer.writerow(tuple(row))

    writer.writerow(('auprin',))
    writer.writerow(('Percentage 10', 'Percentage 20', 'Percentage 30', 'Percentage 40', 'Percentage 50'))

    for r in range(maxlength):
        row = []
        for percentage in percentages:
            list = results[percentage]
            res_auprin = [auprin for auc, f1, fpr95, error, auprin, auprout in list]
            row.append(res_auprin[r] if len(list) > r else '')
        writer.writerow(tuple(row))

    writer.writerow(('auprout',))
    writer.writerow(('Percentage 10', 'Percentage 20', 'Percentage 30', 'Percentage 40', 'Percentage 50'))

    for r in range(maxlength):
        row = []
        for percentage in percentages:
            list = results[percentage]
            res_auprout = [auprout for auc, f1, fpr95, error, auprin, auprout in list]
            row.append(res_auprout[r] if len(list) > r else '')
        writer.writerow(tuple(row))
    f.close()

results = {}

for percentage in percentages:
    results[percentage] = []

for fold in range(5 if full_run else 1):
    for i in range(10):
        train_AAE.main(fold, [i], 10)
        res = novelty_detector.main(fold, [i], 10)

        for k, v in res.items():
            results[k].append(v)

        save_results(results)
