import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_squared_error, balanced_accuracy_score
)
import config
from config import *
from sklearn.model_selection import StratifiedKFold
from models.takagi_sugeno import TakagiSugenoClassifier

# Calculate Metrics (Micro, Macro, RSE, RMSE, F1-Score)
def calcular_metricas(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    acc_macro = balanced_accuracy_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    rse_val = np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return {
        'acuracia': acc,
        'acuracia_macro': acc_macro,
        'rse': rse_val,
        'rmse': rmse,
        'f1_score': f1
    }

# Cross-validation (5-fold, 80/20) and Evaluation
def cross_validation(X, y, n_rules=N_RULES, m=M_FUZZ):
    if config.PRINT_OPTION:
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

        if config.PRINT_OPTION:
            print(f"\n  Fold {fold_idx}/{N_FOLDS}:")
            print(f"    Treino: {len(train_idx):,} | Teste: {len(test_idx):,}")
            print(f"    Acurácia (micro): {metricas['acuracia']:.4f} | Acurácia (macro): {metricas['acuracia_macro']:.4f} "
                  f"| RSE: {metricas['rse']:.4f} | RMSE: {metricas['rmse']:.4f} | F1: {metricas['f1_score']:.4f}")

        if metricas['acuracia'] > melhor_acc:
            melhor_acc = metricas['acuracia']
            melhor_modelo = modelo
            melhor_dados = (X_train, X_test, y_train, y_test, y_pred)

    # Médias e desvios
    df_metricas = pd.DataFrame(metricas_folds)
    medias = df_metricas.mean()
    desvios = df_metricas.std()

    if config.PRINT_OPTION:
        print(f"\n  {'─'*50}")
        print(f"  RESULTADOS MÉDIOS ({N_FOLDS} folds):")
        print(f"  {'─'*50}")
        print(f"    Acurácia (micro): {medias['acuracia']:.4f} ± {desvios['acuracia']:.4f}")
        print(f"    Acurácia (macro): {medias['acuracia_macro']:.4f} ± {desvios['acuracia_macro']:.4f}")
        print(f"    RSE             : {medias['rse']:.4f} ± {desvios['rse']:.4f}")
        print(f"    RMSE            : {medias['rmse']:.4f} ± {desvios['rmse']:.4f}")
        print(f"    F1-Score        : {medias['f1_score']:.4f} ± {desvios['f1_score']:.4f}")

    # Relatório e matriz de confusão do melhor fold
    if config.PRINT_OPTION:
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
