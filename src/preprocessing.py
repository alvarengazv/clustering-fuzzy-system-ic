# Remover os atributos ['atributo_1', 'atributo_2', 'atributo_3']
# Remover duplicatas
# Remover entradas com valores ausentes
# Normalizar os atributos numéricos, exceto a coluna de classe

import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE

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

from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

def aplicar_balanceamento(df: pd.DataFrame):
    """
    Aplica técnicas de balanceamento. 
    Você pode comentar/descomentar a técnica que deseja testar.
    """
    X = df.drop(columns=[COL_CLASSE])
    y = df[COL_CLASSE]
    
    # OPÇÃO 1: Undersampling (Reduzir as maiores para igualar à menor - 1000 amostras cada)
    # Muito útil quando dados sintéticos geram ruído que confunde o C-Means
    # sampler = RandomUnderSampler(random_state=42)
    
    # OPÇÃO 2: SMOTE + ENN (Híbrido)
    # Ele cria dados sintéticos, mas depois apaga amostras (reais ou falsas) 
    # que estão nas fronteiras sobrepostas causando confusão (limpa o ruído).
    # sampler = SMOTEENN(random_state=42)
    
    # OPÇÃO 3: ADASYN 
    # Parecido com SMOTE, mas foca em gerar dados onde a rede tem mais dificuldade.
    from imblearn.over_sampling import ADASYN
    sampler = ADASYN(random_state=42)

    X_res, y_res = sampler.fit_resample(X, y)
    
    df_res = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=COL_CLASSE)], axis=1)
    return df_res

def preprocessing(df: pd.DataFrame):
    # df = remover_atributos(df)
    df = remover_duplicatas(df)
    df = remover_valores_ausentes(df)
    df = normalizar_atributos(df)
    df = aplicar_balanceamento(df)
    return df

def executar_preprocessing():
    df = pd.read_csv(dataset_path)
    df = preprocessing(df)
    df.to_csv('dataset/base_sintetica_media_preprocessed.csv', index=False)

if __name__ == "__main__":
    executar_preprocessing()