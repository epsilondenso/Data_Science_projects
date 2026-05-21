from sympy import factorint

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