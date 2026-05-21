import pandas as pd
import numpy as np
import dcor 

def save_features_names(df: pd.DataFrame, path: str) -> None:
    """Guarda las columnas de un df en un txt, una por línea"""
    with open(path, "w", encoding="utf-8") as features_txt:
        for feature in df.columns:
            features_txt.write(f"{feature}\n")

def dcc_matrix(M: np.array, Round = 2, **kwargs) -> np.array:
  """
  Toma una matriz con N renglones, devuelve una matriz
  NxN con los DCC (Distance Correlation Coeficient)
  entre cada una de las N series
  """
  N = M.shape[0]
  DCC_m = np.zeros((N,N))
  for i in range(0, N, 1):
    for j in range(0, N, 1):
      DCC_m[i,j] = dcor.distance_correlation(M[i][0:], M[j][0:], **kwargs)

  return np.round(DCC_m, Round)

def effective_correlations(correlation_matrix: np.array, diagonal: bool = False, indices: bool = False) -> np.array:
    
    """ Devuelve un array con los valores de la matriz de correlación que se encuentran
        sobre la diagonal principal si diagonal = False, o incluyendo la diagonal si diagonal = True.
        Si indices=True, también devuelve los índices correspondientes a esos valores."""

    indices = np.triu_indices_from(correlation_matrix, k = 0 if diagonal else 1)

    return correlation_matrix[indices] if not indices else (correlation_matrix[indices], indices)