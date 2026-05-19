# Remover os atributos ['atributo_1', 'atributo_2', 'atributo_3']
# Remover duplicatas
# Remover entradas com valores ausentes
# Normalizar os atributos numéricos, exceto a coluna de classe

import pandas as pd
import numpy as np
import os

# Get the current file path
current_file_path = os.path.abspath(__file__)
# Get the parent directory path
parent_dir_path = os.path.dirname(os.path.dirname(current_file_path))
# Path to the dataset
dataset_path = os.path.join(parent_dir_path, 'dataset/base_sintetica_media.csv')

# Column names
ATRIBUTOS = ['atributo_1', 'atributo_2', 'atributo_3', 'atributo_4', 'atributo_5', 'atributo_6']
COL_CLASSE = 'classe'

# Read the dataset
df = pd.read_csv(dataset_path)


def remover_atributos(df: pd.DataFrame):
    df = df.drop(columns=['atributo_1', 'atributo_2', 'atributo_3'])
    return df

def remover_duplicatas(df: pd.DataFrame):
    df = df.drop_duplicates()
    return df

def remover_valores_ausentes(df: pd.DataFrame):
    df = df.dropna()
    return df

def normalizar_atributos(df: pd.DataFrame):
    """
    Normaliza os atributos numéricos, exceto a coluna de classe.
    """
    df_to_normalize = df.drop(columns=[COL_CLASSE])
    df_to_normalize = df_to_normalize.apply(lambda x: (x - x.mean()) / x.std() if x.dtype in ['int64', 'float64'] else x)
    df = pd.concat([df_to_normalize, df[[COL_CLASSE]]], axis=1)
    return df

def preprocessing(df: pd.DataFrame):
    df = remover_atributos(df)
    df = remover_duplicatas(df)
    df = remover_valores_ausentes(df)
    df = normalizar_atributos(df)
    return df

def executar_preprocessing():
    df = pd.read_csv(dataset_path)
    df = preprocessing(df)
    df.to_csv('dataset/base_sintetica_media_preprocessed.csv', index=False)

if __name__ == "__main__":
    executar_preprocessing()