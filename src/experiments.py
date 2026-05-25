import numpy as np
import pandas as pd
import os
import config
from config import *
from utils.metrics import *
from utils.plots import plotar_analise_sensibilidade

def experimentar_hiperparametros(X, y):
    original_print_option = config.PRINT_OPTION
    
    if original_print_option:
        print(f"\n{'='*60}")
        print(f"  ANÁLISE DE SENSIBILIDADE DOS HIPERPARÂMETROS")
        if config.METODO_VALIDACAO == 'kfold':
            print(f"  (Stratified K-Fold {N_FOLDS}-Folds para cada configuração)")
        else:
            print(f"  (Holdout Treino 80% / Teste 20% para cada configuração)")
        print(f"{'='*60}")
        print(f"\n  ── Variação do expoente m (n_rules={N_RULES}) ──")
        print(f"    {'m':>7} | {'Acc micro':>10} | {'Acc macro':>10} | {'Recall':>10} | {'AUC':>10} | {'F1':>10}")
        print(f"    {'─'*60}")
    
    config.PRINT_OPTION = False
    
    resultados = []
    
    for m_val in [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0]:
        medias, desvios, _, _, _ = cross_validation(X, y, n_rules=N_RULES, m=m_val)
        resultados.append({
            'n_rules': N_RULES, 'm': m_val,
            'acuracia': medias['acuracia'], 'acuracia_macro': medias['acuracia_macro'],
            'recall': medias['recall'], 'auc': medias['auc'], 'f1_score': medias['f1_score'],
            'acuracia_std': desvios['acuracia'], 'acuracia_macro_std': desvios['acuracia_macro'],
            'recall_std': desvios['recall'], 'auc_std': desvios['auc'], 'f1_std': desvios['f1_score']
        })
        m = medias
        if original_print_option:
            print(f"    {m_val:>7.1f} | {m['acuracia']:>10.4f} | {m['acuracia_macro']:>10.4f} | {m['recall']:>10.4f} | "
                  f"{m['auc']:>10.4f} | {m['f1_score']:>10.4f}")
         
    config.PRINT_OPTION = original_print_option

    df_res = pd.DataFrame(resultados)
    plotar_analise_sensibilidade(df_res)

    return df_res

