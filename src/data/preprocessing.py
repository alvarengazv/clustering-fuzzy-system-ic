import pandas as pd
import os
import sys
from imblearn.combine import SMOTEENN
from config import *

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

# Apply balancing techniques
def aplicar_balanceamento(df: pd.DataFrame):
    X = df.drop(columns=[COL_CLASSE])
    y = df[COL_CLASSE]
    
    # SMOTE + ENN: Makes the dataset balanced, creates synthetic data and removes samples on overlapping borders
    sampler = SMOTEENN(random_state=42)
    
    X_res, y_res = sampler.fit_resample(X, y)
    df_res = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=COL_CLASSE)], axis=1)
    return df_res

# Preprocessing pipeline
def preprocessing(df: pd.DataFrame):
    df = remover_duplicatas(df)
    df = remover_valores_ausentes(df)
    df = normalizar_atributos(df)
    df = aplicar_balanceamento(df)
    return df

# Execute the preprocessing pipeline
def executar_preprocessing():
    df = pd.read_csv(DATASET_RAW_PATH)
    df = preprocessing(df)
    df.to_csv(DATASET_PREPROCESSED_PATH, index=False)

# Option to run the preprocessing (only appears if the base_sintetica_media_processed.csv file exists)
def preprocessing_option():
    import config
    if not config.PRINT_OPTION:
        if not os.path.exists(DATASET_PREPROCESSED_PATH):
            executar_preprocessing()
        return

    if os.path.exists(DATASET_RAW_PATH):
        if os.path.exists(DATASET_PREPROCESSED_PATH):
            print("O arquivo base_sintetica_media_preprocessed.csv já existe.")
            choice = input("Deseja rodar o preprocessing novamente? (s/n): ")
            if choice.lower() == 's':
                executar_preprocessing()
        else:
            executar_preprocessing()
    else:
        print(f"O arquivo {DATASET_RAW_PATH} não foi encontrado na pasta {DATASET_DIR}")
        print(f"Por favor, baixe o dataset e salve-o na pasta {DATASET_DIR}.")
        sys.exit()