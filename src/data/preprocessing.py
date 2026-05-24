import pandas as pd
import os
import sys
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from imblearn.combine import SMOTEENN
import config
from config import *
from utils.plots import plotar_antes_depois_smote, plotar_dispersao_dados

# Remove duplicate entries
def remover_duplicatas(df: pd.DataFrame):
    df = df.drop_duplicates()
    return df

# Remove entries with missing values
def remover_valores_ausentes(df: pd.DataFrame):
    df = df.dropna()
    return df

# Normalize the attributes, except the class column
def normalizar_atributos(df: pd.DataFrame):
    df_to_normalize = df.drop(columns=[COL_CLASSE])
    df_to_normalize = df_to_normalize.apply(lambda x: (x - x.mean()) / x.std() if x.dtype in ['int64', 'float64'] else x)
    df = pd.concat([df_to_normalize, df[[COL_CLASSE]]], axis=1)
    return df

# Remove features with high pearson correlation
def remover_alta_correlacao(df: pd.DataFrame):
    if not config.REMOVER_CORRELACIONADOS:
        return df
        
    df_features = df.drop(columns=[COL_CLASSE])
    corr_matrix = df_features.corr().abs()
    
    # Pegar o triângulo superior da matriz de correlação
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Encontrar as colunas a serem descartadas (correlação > 0.8)
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    
    if len(to_drop) > 0:
        if config.PRINT_OPTION:
            print(f"  [Pré-processamento] Removendo atributos altamente correlacionados (>0.8): {to_drop}")
        df = df.drop(columns=to_drop)
        
    return df

# Apply balancing techniques
def aplicar_balanceamento(df: pd.DataFrame):
    X = df.drop(columns=[COL_CLASSE])
    y = df[COL_CLASSE]
    
    # SMOTE + ENN: Makes the dataset balanced, creates synthetic data and removes samples on overlapping borders
    sampler = SMOTEENN(random_state=42)
    
    X_res, y_res = sampler.fit_resample(X, y)
    df_res = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=COL_CLASSE)], axis=1)
    
    plotar_antes_depois_smote(df, df_res)
    
    return df_res

def remover_outliers_iqr(df: pd.DataFrame, colunas_atributos):
    limite = 1.5
    for col in colunas_atributos:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - limite * iqr
        limite_superior = q3 + limite * iqr
        
        # Filtra mantendo apenas o que está dentro dos limites
        df = df[(df[col] >= limite_inferior) & (df[col] <= limite_superior)]
    return df

# Preprocessing pipeline
def preprocessing(df: pd.DataFrame):
    df = remover_duplicatas(df)
    df = remover_valores_ausentes(df)
    df = normalizar_atributos(df)
    
    # Plotar a dispersão dos dados antes de mexer na quantidade de amostras (SMOTE/Outliers)
    plotar_dispersao_dados(df, "Dispersão após Normalização (Antes Balanceamento)", "dispersao_pre_balanceamento")

    # Remover alta correlação, se ativado
    df = remover_alta_correlacao(df)

    if config.APLICAR_BALANCEAMENTO:
        df = aplicar_balanceamento(df)
    else:
        cols = [c for c in df.columns if c != COL_CLASSE]
        df = remover_outliers_iqr(df, cols)
    return df

# Execute the preprocessing pipeline
def executar_preprocessing():
    df = pd.read_csv(config.DATASET_RAW_PATH)
    df = preprocessing(df)
    df.to_csv(config.DATASET_PREPROCESSED_PATH, index=False)

# Option to run the preprocessing (only appears if the base_sintetica_media_processed.csv file exists)
def preprocessing_option():
    if not config.PRINT_OPTION:
        if not os.path.exists(config.DATASET_PREPROCESSED_PATH):
            executar_preprocessing()
        return

    if os.path.exists(config.DATASET_RAW_PATH):
        if os.path.exists(config.DATASET_PREPROCESSED_PATH):
            print(f"O arquivo {os.path.basename(config.DATASET_PREPROCESSED_PATH)} já existe.")
            choice = input("Deseja rodar o preprocessing novamente? (s/n): ").strip()
            if choice.lower() == 's':
                executar_preprocessing()
        else:
            executar_preprocessing()
    else:
        print(f"O arquivo {config.DATASET_RAW_PATH} não foi encontrado na pasta {config.DATASET_DIR}")
        print(f"Por favor, baixe o dataset e salve-o na pasta {config.DATASET_DIR}.")
        sys.exit()