import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score
                              ,roc_curve, precision_recall_curve, auc)
from .utils import copy_dicts
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

def train_eval_model(pipeline: BaseEstimator, fold_maker: StratifiedKFold, 
                     features: pd.DataFrame, target: pd.Series, 
                     metrics_dicts: tuple[dict] | None = None) -> tuple[dict]:
    
    num_metrics_dict, roc_dict, prc_dict = metrics_dicts if metrics_dicts else copy_dicts()

    for train_index, test_index in fold_maker.split(features, target):
        #train-test split:
        x_trn, y_trn = features.iloc[train_index], target.iloc[train_index]
        x_test, y_test = features.iloc[test_index], target.iloc[test_index]
        #Clonamos el pipeline en cada fold:
        pipe = clone(pipeline)
        #Entrenamos:
        pipe.fit(x_trn, y_trn)
        #Hacemos predicciones:
        y_pred = pipe.predict(x_test)
        y_proba = pipe.predict_proba(x_test)[:, 1]
        # Evaluamos métricas:
        fold_eval_metrics(y_test = y_test, y_pred = y_pred, y_proba = y_proba, 
                      metrics_dict= num_metrics_dict,
                      roc_dict= roc_dict,
                      prc_dict = prc_dict)
        
    return (num_metrics_dict, roc_dict, prc_dict)
