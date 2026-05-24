import os 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
APLICAR_BALANCEAMENTO = False # Define se aplica o balanceamento com SMOTE-ENN (True para sim, False para não)
METODO_VALIDACAO = 'holdout' # Método de validação: 'holdout' (80/20) ou 'kfold' (Stratified K-Fold)

DATASET_RAW_PATH = os.path.join(DATASET_DIR, 'base_sintetica_media.csv')
if APLICAR_BALANCEAMENTO:
    DATASET_PREPROCESSED_PATH = os.path.join(DATASET_DIR, 'base_sintetica_media_preprocessed.csv')
else:
    DATASET_PREPROCESSED_PATH = os.path.join(DATASET_DIR, 'base_sintetica_media_preprocessed_sem_smote.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'resultados')

ATRIBUTOS = ['atributo_1', 'atributo_2', 'atributo_3', 'atributo_4', 'atributo_5', 'atributo_6']
# ATRIBUTOS = ['atributo_4', 'atributo_5', 'atributo_6'] # Removidos atributo_1, atributo_2 e atributo_3 para teste de acurácia
COL_CLASSE = 'classe'

# Hiperparâmetros padrão
N_RULES = 4
M_FUZZ = 2.0
MAX_ITER_FCM = 300
N_FOLDS = 5          # 5-fold → 80% treino / 20% teste
RANDOM_STATE = 42
PRINT_OPTION = True
REMOVER_CORRELACIONADOS = False

def get_sufixo_config():
    sufixo_smote = "smote" if APLICAR_BALANCEAMENTO else "semsmote"
    sufixo_corr = "corr" if REMOVER_CORRELACIONADOS else "completos"
    return f"_{sufixo_smote}_{METODO_VALIDACAO}_{sufixo_corr}"
