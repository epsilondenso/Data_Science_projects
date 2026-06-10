from sympy import factorint
import copy

color1, color0 = ("#fb2e01", "#6fcb9f")

def get_subplt_dim(n: int = 100) -> tuple:
    factors = factorint(n)
    d_1 = 1
    if len(factors.items()) > 1:
        for i in range(len(factors.items())-1):
            t = list(factors.items())[i]
            d_1 *= t[0] ** t[1]
        d_2 = n // d_1
        rows = max(d_1, d_2)
        cols = min(d_1, d_2)
    else:
        fac_exp = list(factors.items())[0]
        rows = fac_exp[0]**(fac_exp[1] - 1)
        cols = fac_exp[0]
    return (rows, cols)

metrics_eval= {"accuracy": [], "recall": [], "precision": [], "f1": []}
roc_eval = {"fpr": [], "tpr": [], "thresholds": []}
prc_eval = {"precision": [], "recall": [], "thresholds": []}

def copy_dicts(orig_dicts: list[dict] = [metrics_eval, roc_eval, prc_eval]) -> tuple[dict]:
    """
    Creates deep copies of a list of dictionaries, returns the copies as a tuple.
    """
    copies = [copy.deepcopy(orig_dict) for orig_dict in orig_dicts]
    return tuple(copies)