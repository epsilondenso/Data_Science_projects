import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score
                              ,roc_curve, precision_recall_curve, auc)

metrics = [accuracy_score, recall_score, precision_score, f1_score]

def metrics_stats(metrics_eval: dict[list], decimals: int = 4) -> pd.DataFrame:
    stats = pd.DataFrame()
    stats["Métrica"] = metrics_eval.keys()
    stats["Promedio"] = [np.round(np.mean(metrics_eval[metric_name]), decimals) for metric_name in metrics_eval.keys()]
    stats["Desv. Std."] = [np.round(np.std(metrics_eval[metric_name]), decimals) for metric_name in metrics_eval.keys()]
    return stats

def fold_eval_metrics(y_test: np.array, y_pred: np.array, y_proba: np.array,
                      metrics_dict: dict[list], 
                      roc_dict: dict[list],
                      prc_dict: dict[list], 
                      metrics:list = metrics):

    for i, metric_name in enumerate(metrics_dict.keys()):
        metric_value = metrics[i](y_test, y_pred)
        metrics_dict[metric_name].append(metric_value)
    #Calculamos las curvas ROC y PRC:
        #ROC
    roc = roc_curve(y_test, y_proba)
    for i, key in enumerate(roc_dict.keys()):
        roc_dict[key].append(roc[i])

        #PRC
    prc = precision_recall_curve(y_test, y_proba)
    for i, key in enumerate(prc_dict.keys()):
        prc_dict[key].append(prc[i])