import pandas as pd
import numpy as np

def has_the_same_value(df: pd.DataFrame, column: str) -> bool:
    """ Verifica si una columna tiene el mismo valor en todas las filas
    PARÁMETROS:
      - df: pd.DataFrame - El DataFrame a verificar
      - column: str - El nombre de la columna a verificar
    RETORNA:
      - bool: True si la columna tiene el mismo valor en todas las filas, False en caso contrario"""
    return df[column].nunique() == 1

def get_duplicated_index(df: pd.DataFrame) -> np.array:
    """ Obtiene los índices de las filas (o columnas) duplicadas en un DataFrame
    PARÁMETROS:
      - df: pd.DataFrame - El DataFrame a verificar
    RETORNA:
      - np.array: Un array con los índices de las filas (o columnas) duplicadas, 
                  incluyendo la primera aparición de cada duplicado"""
    return np.where(df.duplicated(keep=False))[0]
