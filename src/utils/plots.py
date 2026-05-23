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
    if config.PRINT_OPTION:
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
    if config.PRINT_OPTION:
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
    if config.PRINT_OPTION:
        print(f"  [Gráfico] Funções de pertinência salvas em: {path_mf}")

    # ── 4. Dispersão 3D ──
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

