from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, average_precision_score
import numpy as np
import torch


def dec_metrics(outputs, labels):
    # 
    preds = torch.argmax(outputs, dim=1)

    # 
    labels_np = labels.cpu().detach().numpy()
    preds_np = preds.cpu().detach().numpy()
    scores_np = outputs[:, 1].cpu().detach().numpy()

    # 
    acc = accuracy_score(labels_np, preds_np)

    #  AUC
    auc = roc_auc_score(labels_np, scores_np)

    #  EER
    fpr, tpr, thresholds = roc_curve(labels_np, scores_np)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_index]

    #  Precision, Recall, F1
    precision = precision_score(labels_np, preds_np)
    recall = recall_score(labels_np, preds_np)
    f1 = f1_score(labels_np, preds_np)

    #  Average Precision (AP)
    ap = average_precision_score(labels_np, scores_np)

    return acc, auc, eer, precision, recall, f1, ap
