import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import config
from config import *
from utils.metrics import *


def experimentar_hiperparametros(X, y):
    print(f"\n{'='*60}")
    print(f"  ANÁLISE DE SENSIBILIDADE DOS HIPERPARÂMETROS")
    print(f"  (Cross-validation {N_FOLDS}-fold para cada configuração)")
    print(f"{'='*60}")

    resultados = []

    # Variação do expoente de fuzzificação (n_rules=4)
    print(f"\n  ── Variação do expoente m (n_rules={N_RULES}) ──")
    print(f"    {'m':>7} | {'Acc micro':>10} | {'Acc macro':>10} | {'RSE':>10} | {'RMSE':>10} | {'F1':>10}")
    print(f"    {'─'*60}")
    
    # Store original print option and turn it off for the loop
    original_print_option = config.PRINT_OPTION
    config.PRINT_OPTION = False
    
    for m_val in [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0]:
        medias, desvios, _, _, _ = cross_validation(X, y, n_rules=N_RULES, m=m_val)
        resultados.append({
            'n_rules': N_RULES, 'm': m_val,
            'acuracia': medias['acuracia'], 'acuracia_macro': medias['acuracia_macro'],
            'rse': medias['rse'], 'rmse': medias['rmse'], 'f1_score': medias['f1_score'],
            'acuracia_std': desvios['acuracia'], 'acuracia_macro_std': desvios['acuracia_macro'],
            'rse_std': desvios['rse'], 'rmse_std': desvios['rmse'], 'f1_std': desvios['f1_score']
        })
        m = medias
        print(f"    {m_val:>7.1f} | {m['acuracia']:>10.4f} | {m['acuracia_macro']:>10.4f} | {m['rse']:>10.4f} | "
              f"{m['rmse']:>10.4f} | {m['f1_score']:>10.4f}")
              
    # Restore print option
    config.PRINT_OPTION = original_print_option

    # Plotar resultados
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_res = pd.DataFrame(resultados)
    metricas_plot = [
        ('acuracia', 'Acurácia (micro)', '#3498db'),
        ('acuracia_macro', 'Acurácia (macro)', '#9b59b6'),
        ('rse', 'RSE', '#e74c3c'),
        ('rmse', 'RMSE', '#2ecc71'),
        ('f1_score', 'F1-Score', '#f39c12')
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    df_m = df_res[df_res['n_rules'] == N_RULES]
    for idx, (col, label, color) in enumerate(metricas_plot):
        ax = axes[idx]
        ax.plot(df_m['m'], df_m[col], 'o-', color=color,
                linewidth=2, markersize=8, label=label)
        ax.fill_between(df_m['m'],
                        df_m[col] - df_m.get(f'{col}_std', 0),
                        df_m[col] + df_m.get(f'{col}_std', 0),
                        alpha=0.15, color=color)
        ax.set_xlabel('Expoente de Fuzzificação (m)', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label} vs Expoente m (n_rules={N_RULES})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Ocultar o último subplot (pois temos 5 plots em um grid de 2x3)
    axes[5].set_visible(False)

    plt.suptitle(f'Análise de Sensibilidade ({N_FOLDS}-Fold CV)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path_hp = os.path.join(RESULTS_DIR, 'analise_hiperparametros.png')
    plt.savefig(path_hp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [Gráfico] Análise de hiperparâmetros salva em: {path_hp}")

    # Gráfico consolidado: todas as métricas vs m
    fig, ax = plt.subplots(figsize=(10, 6))

    for col, label, color in metricas_plot:
        ax.plot(df_m['m'], df_m[col], 'o-',
                linewidth=2, markersize=8, label=label, color=color)
    ax.set_xlabel('Expoente de Fuzzificação (m)', fontsize=11)
    ax.set_ylabel('Valor da Métrica', fontsize=11)
    ax.set_title(f'Comparação de Métricas vs Expoente m (n_rules={N_RULES})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Comparação Consolidada ({N_FOLDS}-Fold CV)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path_cons = os.path.join(RESULTS_DIR, 'comparacao_consolidada.png')
    plt.savefig(path_cons, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Gráfico] Comparação consolidada salva em: {path_cons}")

    return df_res

