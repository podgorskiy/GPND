import numpy as np
from sklearn.metrics import roc_auc_score
import pickle
import os


def get_f1(true_positive, false_positive, false_negative):
    if true_positive == 0:
        return 0.0
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return 2.0 * precision * recall / (precision + recall)


def evaluate(logger, percentage_of_outliers, inliner_classes, prediction, threshold, gt_inlier):
    y = np.greater(prediction, threshold)

    gt_outlier = np.logical_not(gt_inlier)

    true_positive = np.sum(np.logical_and(y, gt_inlier))
    true_negative = np.sum(np.logical_and(np.logical_not(y), gt_outlier))
    false_positive = np.sum(np.logical_and(y, gt_outlier))
    false_negative = np.sum(np.logical_and(np.logical_not(y), gt_inlier))
    total_count = true_positive + true_negative + false_positive + false_negative

    accuracy = 100 * (true_positive + true_negative) / total_count

    y_true = gt_inlier
    y_scores = prediction

    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0

    logger.info("Percentage %f" % percentage_of_outliers)
    logger.info("Accuracy %f" % accuracy)
    f1 = get_f1(true_positive, false_positive, false_negative)
    logger.info("F1 %f" % get_f1(true_positive, false_positive, false_negative))
    logger.info("AUC %f" % auc)

    # return dict(auc=auc, f1=f1)

    # inliers
    X1 = [x[1] for x in zip(gt_inlier, prediction) if x[0]]

    # outliers
    Y1 = [x[1] for x in zip(gt_inlier, prediction) if not x[0]]

    minP = min(prediction) - 1
    maxP = max(prediction) + 1

    ##################################################################
    # FPR at TPR 95
    ##################################################################
    fpr95 = 0.0
    clothest_tpr = 1.0
    dist_tpr = 1.0
    for threshold in np.arange(minP, maxP, 0.2):
        tpr = np.sum(np.greater_equal(X1, threshold)) / np.float(len(X1))
        fpr = np.sum(np.greater_equal(Y1, threshold)) / np.float(len(Y1))
        if abs(tpr - 0.95) < dist_tpr:
            dist_tpr = abs(tpr - 0.95)
            clothest_tpr = tpr
            fpr95 = fpr

    logger.info("tpr: %f" % clothest_tpr)
    logger.info("fpr95: %f" % fpr95)

    ##################################################################
    # Detection error
    ##################################################################
    error = 1.0
    for threshold in np.arange(minP, maxP, 0.2):
        tpr = np.sum(np.less(X1, threshold)) / np.float(len(X1))
        fpr = np.sum(np.greater_equal(Y1, threshold)) / np.float(len(Y1))
        error = np.minimum(error, (tpr + fpr) / 2.0)

    logger.info("Detection error: %f" % error)

    ##################################################################
    # AUPR IN
    ##################################################################
    auprin = 0.0
    recallTemp = 1.0
    for threshold in np.arange(minP, maxP, 0.2):
        tp = np.sum(np.greater_equal(X1, threshold))
        fp = np.sum(np.greater_equal(Y1, threshold))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / np.float(len(X1))
        auprin += (recallTemp - recall) * precision
        recallTemp = recall
    auprin += recall * precision

    logger.info("auprin: %f" % auprin)

    ##################################################################
    # AUPR OUT
    ##################################################################
    minP, maxP = -maxP, -minP
    X1 = [-x for x in X1]
    Y1 = [-x for x in Y1]
    auprout = 0.0
    recallTemp = 1.0
    for threshold in np.arange(minP, maxP, 0.2):
        tp = np.sum(np.greater_equal(Y1, threshold))
        fp = np.sum(np.greater_equal(X1, threshold))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / np.float(len(Y1))
        auprout += (recallTemp - recall) * precision
        recallTemp = recall
    auprout += recall * precision

    logger.info("auprout: %f" % auprout)

    with open(os.path.join("results.txt"), "a") as file:
        file.write(
            "Class: %s\n Percentage: %d\n"
            "Error: %f\n F1: %f\n AUC: %f\nfpr95: %f"
            "\nDetection: %f\nauprin: %f\nauprout: %f\n\n" %
            ("_".join([str(x) for x in inliner_classes]), percentage_of_outliers, error, f1, auc, fpr95, error, auprin, auprout))

    return dict(auc=auc, f1=f1, fpr95=fpr95, error=error, auprin=auprin, auprout=auprout)
    # return auc, f1, fpr95, error, auprin, auprout
