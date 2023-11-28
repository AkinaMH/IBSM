import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *


def metric(y_test, y_pred, y_pred_pro):
    scores = np.array([])

    Recall = recall_score(y_test, y_pred)

    Precision = precision_score(y_test, y_pred)

    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    # scores = np.append(scores, F1)
    F1 = f1_score(y_test, y_pred)
    scores = np.append(scores, F1)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    Specificity = tn / (tn + fp)

    G_mean = (Recall * Specificity) ** 0.5
    scores = np.append(scores, G_mean)

    Auc = roc_auc_score(y_test, y_pred)
    # Auc = roc_auc_score(y_test, y_pred_pro)
    scores = np.append(scores, Auc)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_pro)
    roc_auc = auc(fpr, tpr)
    # print(f'Auc = {Auc}')
    # print(f'roc_auc = {roc_auc}')
    # AUC画图
    # display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
    # display.plot()
    # plt.show()

    return scores
