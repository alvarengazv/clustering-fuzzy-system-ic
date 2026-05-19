"""
Pipeline principal: Fuzzy C-Means + Takagi-Sugeno para classificação.

Etapas:
  1. Carrega dados preprocessados
  2. Cross-validation estratificado (5-fold, 80/20)
  3. Treina o modelo Takagi-Sugeno (que internamente usa FCM)
  4. Avalia com acurácia, RSE, RMSE, F1-Score, matriz de confusão (%)
  5. Compara hiperparâmetros com as 4 métricas
  6. Gera gráficos de análise
"""

import sys
import os

# Adicionar diretório src ao path para imports locais
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_squared_error
)

from fuzzy_cmeans import FuzzyCMeans
from takagi_sugeno import TakagiSugenoClassifier


# ────────────────────────────────────────────────────────────────
# Configuração
# ────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'base_sintetica_media_preprocessed.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'resultados')

ATRIBUTOS = ['atributo_4', 'atributo_5', 'atributo_6']
COL_CLASSE = 'classe'

# Hiperparâmetros padrão
N_RULES = 4
M_FUZZ = 2.0
MAX_ITER_FCM = 300
N_FOLDS = 5          # 5-fold → 80% treino / 20% teste
RANDOM_STATE = 42


# ────────────────────────────────────────────────────────────────
# Métricas auxiliares
# ────────────────────────────────────────────────────────────────

def calcular_metricas(y_true, y_pred):
    """Calcula acurácia, RSE, RMSE e F1-Score."""
    acc = accuracy_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    rse_val = np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return {'acuracia': acc, 'rse': rse_val, 'rmse': rmse, 'f1_score': f1}


# ────────────────────────────────────────────────────────────────
# Funções principais
# ────────────────────────────────────────────────────────────────

def carregar_dados():
    """Carrega o dataset preprocessado e separa X e y."""
    print(f"\n{'='*60}")
    print(f"  CARREGAMENTO DOS DADOS")
    print(f"{'='*60}")

    df = pd.read_csv(DATASET_PATH)
    print(f"\n  Arquivo   : {os.path.basename(DATASET_PATH)}")
    print(f"  Amostras  : {df.shape[0]:,}")
    print(f"  Atributos : {ATRIBUTOS}")
    print(f"  Classes   : {sorted(df[COL_CLASSE].unique())}")
    print(f"\n  Distribuição das classes:")
    for cls in sorted(df[COL_CLASSE].unique()):
        n = (df[COL_CLASSE] == cls).sum()
        print(f"    Classe {cls}: {n:>5,} ({n/len(df)*100:.1f}%)")

    X = df[ATRIBUTOS].values
    y = df[COL_CLASSE].values
    return X, y, df


def cross_validation(X, y, n_rules=N_RULES, m=M_FUZZ, verbose=True):
    """
    Executa cross-validation estratificado (5-fold, 80/20).
    Retorna métricas médias e o melhor modelo.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  CROSS-VALIDATION ({N_FOLDS}-Fold, 80/20)")
        print(f"{'='*60}")
        print(f"\n  Hiperparâmetros: n_rules={n_rules}, m={m}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    metricas_folds = []
    melhor_acc = -1
    melhor_modelo = None
    melhor_dados = None
    cm_total = None

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        modelo = TakagiSugenoClassifier(
            n_rules=n_rules, m=m,
            max_iter_fcm=MAX_ITER_FCM, random_state=RANDOM_STATE
        )
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        metricas = calcular_metricas(y_test, y_pred)
        metricas_folds.append(metricas)

        # Acumular matriz de confusão
        classes = sorted(np.unique(y))
        cm_fold = confusion_matrix(y_test, y_pred, labels=classes)
        if cm_total is None:
            cm_total = cm_fold
        else:
            cm_total = cm_total + cm_fold

        if verbose:
            print(f"\n  Fold {fold_idx}/{N_FOLDS}:")
            print(f"    Treino: {len(train_idx):,} | Teste: {len(test_idx):,}")
            print(f"    Acurácia: {metricas['acuracia']:.4f} | RSE: {metricas['rse']:.4f} "
                  f"| RMSE: {metricas['rmse']:.4f} | F1: {metricas['f1_score']:.4f}")

        if metricas['acuracia'] > melhor_acc:
            melhor_acc = metricas['acuracia']
            melhor_modelo = modelo
            melhor_dados = (X_train, X_test, y_train, y_test, y_pred)

    # Médias e desvios
    df_metricas = pd.DataFrame(metricas_folds)
    medias = df_metricas.mean()
    desvios = df_metricas.std()

    if verbose:
        print(f"\n  {'─'*50}")
        print(f"  RESULTADOS MÉDIOS ({N_FOLDS} folds):")
        print(f"  {'─'*50}")
        print(f"    Acurácia : {medias['acuracia']:.4f} ± {desvios['acuracia']:.4f}")
        print(f"    RSE      : {medias['rse']:.4f} ± {desvios['rse']:.4f}")
        print(f"    RMSE     : {medias['rmse']:.4f} ± {desvios['rmse']:.4f}")
        print(f"    F1-Score : {medias['f1_score']:.4f} ± {desvios['f1_score']:.4f}")

    # Relatório e matriz de confusão do melhor fold
    if verbose:
        X_train, X_test, y_train, y_test, y_pred = melhor_dados
        classes = sorted(np.unique(y))

        print(f"\n  {'─'*50}")
        print(f"  Relatório de Classificação (melhor fold):")
        print(f"  {'─'*50}")
        report = classification_report(
            y_test, y_pred,
            target_names=[f"Classe {c}" for c in classes],
            zero_division=0
        )
        for line in report.split('\n'):
            print(f"  {line}")

        # Matriz de confusão acumulada em porcentagem
        cm_pct = cm_total.astype(float)
        cm_pct = cm_pct / cm_pct.sum(axis=1, keepdims=True) * 100

        print(f"\n  {'─'*50}")
        print(f"  Matriz de Confusão Acumulada (%, {N_FOLDS} folds):")
        print(f"  {'─'*50}")
        print(f"  {'':>12}", end="")
        for c in classes:
            print(f"  Pred {c:>2}", end="")
        print()
        for i, c in enumerate(classes):
            print(f"  Real {c:>5} ", end="")
            for j in range(len(classes)):
                print(f"  {cm_pct[i,j]:>5.1f}%", end="")
            print()

    return medias.to_dict(), desvios.to_dict(), melhor_modelo, melhor_dados, cm_total


def plotar_resultados(modelo, X_test, y_test, y_pred_test, cm_total):
    """Gera e salva gráficos de análise."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    classes = sorted(np.unique(y_test))
    class_labels = [f"Classe {c}" for c in classes]

    # ── 1. Matriz de Confusão em % (Heatmap) ──
    cm_pct = cm_total.astype(float)
    cm_pct = cm_pct / cm_pct.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_pct, annot=True, fmt='.1f', cmap='Blues',
        xticklabels=class_labels, yticklabels=class_labels,
        ax=ax, linewidths=0.5, vmin=0, vmax=100,
        annot_kws={'size': 12}
    )
    # Adicionar símbolo % nas anotações
    for text in ax.texts:
        text.set_text(text.get_text() + '%')
    ax.set_xlabel('Classe Predita', fontsize=12)
    ax.set_ylabel('Classe Real', fontsize=12)
    ax.set_title(f'Matriz de Confusão (%) — Takagi-Sugeno ({N_FOLDS}-Fold CV)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path_cm = os.path.join(RESULTS_DIR, 'matriz_confusao.png')
    plt.savefig(path_cm, dpi=150)
    plt.close()
    print(f"\n  [Gráfico] Matriz de confusão salva em: {path_cm}")

    # ── 2. Dispersão 2D — real vs predito ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    scatter1 = axes[0].scatter(
        X_test[:, 0], X_test[:, 1],
        c=y_test, cmap='Set1', alpha=0.6, s=15, edgecolors='none'
    )
    axes[0].set_xlabel(ATRIBUTOS[0], fontsize=11)
    axes[0].set_ylabel(ATRIBUTOS[1], fontsize=11)
    axes[0].set_title('Classes Reais', fontsize=13, fontweight='bold')
    plt.colorbar(scatter1, ax=axes[0], label='Classe')

    scatter2 = axes[1].scatter(
        X_test[:, 0], X_test[:, 1],
        c=y_pred_test, cmap='Set1', alpha=0.6, s=15, edgecolors='none'
    )
    axes[1].set_xlabel(ATRIBUTOS[0], fontsize=11)
    axes[1].set_ylabel(ATRIBUTOS[1], fontsize=11)
    axes[1].set_title('Classes Preditas (Takagi-Sugeno)', fontsize=13, fontweight='bold')
    plt.colorbar(scatter2, ax=axes[1], label='Classe')

    for ax in axes:
        for i, center in enumerate(modelo.fcm_.centers_):
            ax.plot(center[0], center[1], 'kx', markersize=12, markeredgewidth=2)
            ax.annotate(f'C{i+1}', (center[0], center[1]),
                       fontsize=9, fontweight='bold',
                       xytext=(5, 5), textcoords='offset points')

    plt.suptitle('Comparação: Classes Reais vs Preditas', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path_scatter = os.path.join(RESULTS_DIR, 'dispersao_real_vs_predito.png')
    plt.savefig(path_scatter, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Gráfico] Dispersão real vs predito salva em: {path_scatter}")

    # ── 3. Funções de pertinência Gaussianas ──
    fig, axes = plt.subplots(1, len(ATRIBUTOS), figsize=(6 * len(ATRIBUTOS), 5))
    if len(ATRIBUTOS) == 1:
        axes = [axes]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    for d, (ax, attr) in enumerate(zip(axes, ATRIBUTOS)):
        x_range = np.linspace(X_test[:, d].min() - 0.5, X_test[:, d].max() + 0.5, 500)
        for i in range(modelo.fcm_.n_clusters):
            center = modelo.fcm_.centers_[i, d]
            sigma = modelo.fcm_.sigmas_[i, d]
            mu = np.exp(-((x_range - center) ** 2) / (2 * sigma ** 2))
            color = colors[i % len(colors)]
            ax.plot(x_range, mu, color=color, linewidth=2,
                    label=f'Regra {i+1} (c={center:.2f}, σ={sigma:.2f})')
            ax.axvline(center, color=color, linestyle='--', alpha=0.4)
        ax.set_xlabel(attr, fontsize=11)
        ax.set_ylabel('Grau de Pertinência', fontsize=11)
        ax.set_title(f'Funções de Pertinência — {attr}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Funções de Pertinência Gaussianas (derivadas do FCM)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path_mf = os.path.join(RESULTS_DIR, 'funcoes_pertinencia.png')
    plt.savefig(path_mf, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Gráfico] Funções de pertinência salvas em: {path_mf}")

    # ── 4. Dispersão 3D ──
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2],
                c=y_test, cmap='Set1', alpha=0.5, s=10)
    ax1.set_xlabel(ATRIBUTOS[0]); ax1.set_ylabel(ATRIBUTOS[1]); ax1.set_zlabel(ATRIBUTOS[2])
    ax1.set_title('Classes Reais (3D)', fontweight='bold')
    for center in modelo.fcm_.centers_:
        ax1.scatter(*center, c='black', marker='X', s=100, edgecolors='white', linewidths=1)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2],
                c=y_pred_test, cmap='Set1', alpha=0.5, s=10)
    ax2.set_xlabel(ATRIBUTOS[0]); ax2.set_ylabel(ATRIBUTOS[1]); ax2.set_zlabel(ATRIBUTOS[2])
    ax2.set_title('Classes Preditas (3D)', fontweight='bold')
    for center in modelo.fcm_.centers_:
        ax2.scatter(*center, c='black', marker='X', s=100, edgecolors='white', linewidths=1)

    plt.suptitle('Visualização 3D — Real vs Predito', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path_3d = os.path.join(RESULTS_DIR, 'dispersao_3d.png')
    plt.savefig(path_3d, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Gráfico] Dispersão 3D salva em: {path_3d}")


def experimentar_hiperparametros(X, y):
    """Testa diferentes hiperparâmetros com cross-validation. Métricas: Acurácia, RSE, RMSE, F1."""
    print(f"\n{'='*60}")
    print(f"  ANÁLISE DE SENSIBILIDADE DOS HIPERPARÂMETROS")
    print(f"  (Cross-validation {N_FOLDS}-fold para cada configuração)")
    print(f"{'='*60}")

    resultados = []

    # Variação do número de regras
    print(f"\n  ── Variação do número de regras (m={M_FUZZ}) ──")
    print(f"    {'n_rules':>7} | {'Acurácia':>10} | {'RSE':>10} | {'RMSE':>10} | {'F1':>10}")
    print(f"    {'─'*60}")
    for n_rules in [2, 3, 4, 5, 6, 8]:
        medias, desvios, _, _, _ = cross_validation(X, y, n_rules=n_rules, m=M_FUZZ, verbose=False)
        resultados.append({
            'n_rules': n_rules, 'm': M_FUZZ,
            'acuracia': medias['acuracia'], 'rse': medias['rse'],
            'rmse': medias['rmse'], 'f1_score': medias['f1_score'],
            'acuracia_std': desvios['acuracia'], 'rse_std': desvios['rse'],
            'rmse_std': desvios['rmse'], 'f1_std': desvios['f1_score']
        })
        m = medias
        print(f"    {n_rules:>7} | {m['acuracia']:>10.4f} | {m['rse']:>10.4f} | "
              f"{m['rmse']:>10.4f} | {m['f1_score']:>10.4f}")

    # Variação do expoente de fuzzificação
    print(f"\n  ── Variação do expoente m (n_rules={N_RULES}) ──")
    print(f"    {'m':>7} | {'Acurácia':>10} | {'RSE':>10} | {'RMSE':>10} | {'F1':>10}")
    print(f"    {'─'*60}")
    for m_val in [1.5, 2.0, 2.5, 3.0, 3.5]:
        medias, desvios, _, _, _ = cross_validation(X, y, n_rules=N_RULES, m=m_val, verbose=False)
        resultados.append({
            'n_rules': N_RULES, 'm': m_val,
            'acuracia': medias['acuracia'], 'rse': medias['rse'],
            'rmse': medias['rmse'], 'f1_score': medias['f1_score'],
            'acuracia_std': desvios['acuracia'], 'rse_std': desvios['rse'],
            'rmse_std': desvios['rmse'], 'f1_std': desvios['f1_score']
        })
        m = medias
        print(f"    {m_val:>7.1f} | {m['acuracia']:>10.4f} | {m['rse']:>10.4f} | "
              f"{m['rmse']:>10.4f} | {m['f1_score']:>10.4f}")

    # Plotar resultados
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_res = pd.DataFrame(resultados)
    metricas_plot = [
        ('acuracia', 'Acurácia', '#3498db'),
        ('rse', 'RSE', '#e74c3c'),
        ('rmse', 'RMSE', '#2ecc71'),
        ('f1_score', 'F1-Score', '#f39c12')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Gráficos para variação de n_rules
    df_rules = df_res[df_res['m'] == M_FUZZ]
    for idx, (col, label, color) in enumerate(metricas_plot):
        ax = axes[idx]
        ax.plot(df_rules['n_rules'], df_rules[col], 'o-', color=color,
                linewidth=2, markersize=8, label=f'{label} (n_rules)')
        ax.fill_between(df_rules['n_rules'],
                        df_rules[col] - df_rules.get(f'{col}_std', 0),
                        df_rules[col] + df_rules.get(f'{col}_std', 0),
                        alpha=0.15, color=color)
        # Sobrepor variação de m
        df_m = df_res[df_res['n_rules'] == N_RULES]
        ax.plot(df_m['m'], df_m[col], 's--', color=color, alpha=0.5,
                linewidth=2, markersize=8, label=f'{label} (m)')
        ax.set_xlabel('Hiperparâmetro', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label} vs Hiperparâmetros', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Análise de Sensibilidade ({N_FOLDS}-Fold CV)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path_hp = os.path.join(RESULTS_DIR, 'analise_hiperparametros.png')
    plt.savefig(path_hp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [Gráfico] Análise de hiperparâmetros salva em: {path_hp}")

    # Gráfico consolidado: n_rules
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for col, label, color in metricas_plot:
        axes[0].plot(df_rules['n_rules'], df_rules[col], 'o-',
                     linewidth=2, markersize=8, label=label, color=color)
    axes[0].set_xlabel('Número de Regras', fontsize=11)
    axes[0].set_ylabel('Valor da Métrica', fontsize=11)
    axes[0].set_title(f'Métricas vs Nº de Regras (m={M_FUZZ})', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    df_m = df_res[df_res['n_rules'] == N_RULES]
    for col, label, color in metricas_plot:
        axes[1].plot(df_m['m'], df_m[col], 'o-',
                     linewidth=2, markersize=8, label=label, color=color)
    axes[1].set_xlabel('Expoente de Fuzzificação (m)', fontsize=11)
    axes[1].set_ylabel('Valor da Métrica', fontsize=11)
    axes[1].set_title(f'Métricas vs Expoente m (n_rules={N_RULES})', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Comparação Consolidada ({N_FOLDS}-Fold CV)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path_cons = os.path.join(RESULTS_DIR, 'comparacao_consolidada.png')
    plt.savefig(path_cons, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Gráfico] Comparação consolidada salva em: {path_cons}")

    return df_res


# ────────────────────────────────────────────────────────────────
# Pipeline principal
# ────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  SISTEMA FUZZY: FCM + TAKAGI-SUGENO PARA CLASSIFICAÇÃO")
    print("═" * 60)

    # 1. Carregar dados
    X, y, df = carregar_dados()

    # 2. Cross-validation com configuração padrão
    medias, desvios, modelo, melhor_dados, cm_total = cross_validation(X, y)

    X_train, X_test, y_train, y_test, y_pred_test = melhor_dados

    # 3. Imprimir base de regras
    modelo.print_rules(feature_names=ATRIBUTOS)

    # 4. Gerar gráficos
    plotar_resultados(modelo, X_test, y_test, y_pred_test, cm_total)

    # 5. Análise de sensibilidade com CV
    df_resultados = experimentar_hiperparametros(X, y)

    # Resumo final
    print(f"\n{'═'*60}")
    print(f"  RESUMO FINAL")
    print(f"{'═'*60}")
    print(f"\n  Modelo: Takagi-Sugeno (1ª ordem) com FCM")
    print(f"  Validação: {N_FOLDS}-Fold Cross-Validation (80/20)")
    print(f"  Configuração: {N_RULES} regras, m={M_FUZZ}")
    print(f"\n  Métricas médias ({N_FOLDS} folds):")
    print(f"    Acurácia : {medias['acuracia']:.4f} ± {desvios['acuracia']:.4f}")
    print(f"    RSE      : {medias['rse']:.4f} ± {desvios['rse']:.4f}")
    print(f"    RMSE     : {medias['rmse']:.4f} ± {desvios['rmse']:.4f}")
    print(f"    F1-Score : {medias['f1_score']:.4f} ± {desvios['f1_score']:.4f}")
    print(f"\n  Gráficos salvos em: {RESULTS_DIR}/")
    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()