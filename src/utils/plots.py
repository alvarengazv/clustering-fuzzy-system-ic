import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import *
import config

# Generate plots for analysis
def plotar_resultados(modelo, X_test, y_test, y_pred_test, cm_total):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    classes = sorted(np.unique(y_test))
    class_labels = [f"Classe {c}" for c in classes]

    # Graphic of the confusion matrix
    cm_pct = cm_total.astype(float)
    cm_pct = cm_pct / cm_pct.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_pct, annot=True, fmt='.1f', cmap='Blues',
        xticklabels=class_labels, yticklabels=class_labels,
        ax=ax, linewidths=0.5, vmin=0, vmax=100,
        annot_kws={'size': 12}
    )
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
    if config.PRINT_OPTION:
        print(f"\n  [Gráfico] Matriz de confusão salva em: {path_cm}")

    # Graphic of the confusion matrix 2d
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
    if config.PRINT_OPTION:
        print(f"  [Gráfico] Dispersão real vs predito salva em: {path_scatter}")

    # Gaussian membership functions
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
    if config.PRINT_OPTION:
        print(f"  [Gráfico] Funções de pertinência salvas em: {path_mf}")

    # Graphic of the dispersion 3d
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2],
                c=y_test, cmap='Set1', alpha=0.5, s=10)
    ax1.set_xlabel(ATRIBUTOS[0]); ax1.set_ylabel(ATRIBUTOS[1]); ax1.set_zlabel(ATRIBUTOS[2])
    ax1.set_title('Classes Reais (3D)', fontweight='bold')
    for center in modelo.fcm_.centers_:
        ax1.scatter(*center[:3], c='black', marker='X', s=100, edgecolors='white', linewidths=1)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2],
                c=y_pred_test, cmap='Set1', alpha=0.5, s=10)
    ax2.set_xlabel(ATRIBUTOS[0]); ax2.set_ylabel(ATRIBUTOS[1]); ax2.set_zlabel(ATRIBUTOS[2])
    ax2.set_title('Classes Preditas (3D)', fontweight='bold')
    for center in modelo.fcm_.centers_:
        ax2.scatter(*center[:3], c='black', marker='X', s=100, edgecolors='white', linewidths=1)

    plt.suptitle('Visualização 3D — Real vs Predito', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path_3d = os.path.join(RESULTS_DIR, 'dispersao_3d.png')
    plt.savefig(path_3d, dpi=150, bbox_inches='tight')
    plt.close()
    if config.PRINT_OPTION:
        print(f"  [Gráfico] Dispersão 3D salva em: {path_3d}")

# Generate plots for hyperparameter sensitivity analysis
def plotar_analise_sensibilidade(df_res: pd.DataFrame):
    os.makedirs(RESULTS_DIR, exist_ok=True)
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

    # Hide the last subplot (since we have 5 plots in a 2x3 grid)
    axes[5].set_visible(False)

    plt.suptitle(f'Análise de Sensibilidade ({N_FOLDS}-Fold CV)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path_hp = os.path.join(RESULTS_DIR, 'analise_hiperparametros.png')
    plt.savefig(path_hp, dpi=150, bbox_inches='tight')
    plt.close()
    if config.PRINT_OPTION:
        print(f"\n  [Gráfico] Análise de hiperparâmetros salva em: {path_hp}")

    # Consolidated chart: all metrics vs m
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
    if config.PRINT_OPTION:
        print(f"  [Gráfico] Comparação consolidada salva em: {path_cons}")


