import os 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
DATASET_RAW_PATH = os.path.join(DATASET_DIR, 'base_sintetica_media.csv')
DATASET_PREPROCESSED_PATH = os.path.join(DATASET_DIR, 'base_sintetica_media_preprocessed.csv')
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
