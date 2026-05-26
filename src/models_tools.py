import pandas as pd
import numpy as np


def metrics_stats(metrics_eval: dict[list], decimals: int = 4) -> pd.DataFrame:
    stats = pd.DataFrame()
    stats["Métrica"] = metrics_eval.keys()
    stats["Promedio"] = [np.round(np.mean(metrics_eval[metric_name]), decimals) for metric_name in metrics_eval.keys()]
    stats["Desv. Std."] = [np.round(np.std(metrics_eval[metric_name]), decimals) for metric_name in metrics_eval.keys()]
    return stats