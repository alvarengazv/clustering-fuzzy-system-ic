import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, recall_score, roc_auc_score, balanced_accuracy_score
)
import config
from config import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from models.takagi_sugeno import TakagiSugenoClassifier

def calcular_metricas(y_true, y_pred, y_pred_proba):
    acc = accuracy_score(y_true, y_pred)
    acc_macro = balanced_accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    except ValueError:
        auc = 0.0

    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'acuracia': acc,
        'acuracia_macro': acc_macro,
        'recall': recall,
        'auc': auc,
        'f1_score': f1
    }

def cross_validation(X, y, n_rules=N_RULES, m=M_FUZZ):
    if config.PRINT_OPTION:
        print(f"\n{'='*60}")
        if config.METODO_VALIDACAO == 'kfold':
            print(f"  AVALIAÇÃO - STRATIFIED K-FOLD ({N_FOLDS} Folds)")
        else:
            print(f"  AVALIAÇÃO - HOLDOUT (Treino 80% / Teste 20%)")
        print(f"{'='*60}")
        print(f"\n  Hiperparâmetros: n_rules={n_rules}, m={m}")

    classes = sorted(np.unique(y))

    if config.METODO_VALIDACAO == 'kfold':
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
            
            y_pred_proba = modelo.predict_proba(X_test)
            y_pred = modelo.predict(X_test)
            metricas = calcular_metricas(y_test, y_pred, y_pred_proba)
            metricas_folds.append(metricas)
            
            cm_fold = confusion_matrix(y_test, y_pred, labels=classes)
            if cm_total is None:
                cm_total = cm_fold
            else:
                cm_total = cm_total + cm_fold
                
            if config.PRINT_OPTION:
                print(f"\n  Fold {fold_idx}/{N_FOLDS}:")
                print(f"    Treino: {len(train_idx):,} | Teste: {len(test_idx):,}")
                print(f"    Acurácia (micro): {metricas['acuracia']:.4f} | Acurácia (macro): {metricas['acuracia_macro']:.4f} "
                      f"| Recall: {metricas['recall']:.4f} | AUC: {metricas['auc']:.4f} | F1: {metricas['f1_score']:.4f}")
                      
            if metricas['acuracia'] > melhor_acc:
                melhor_acc = metricas['acuracia']
                melhor_modelo = modelo
                melhor_dados = (X_train, X_test, y_train, y_test, y_pred, y_pred_proba)
               
        df_metricas = pd.DataFrame(metricas_folds)
        medias = df_metricas.mean()
        desvios = df_metricas.std()
        
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
        )
        
        modelo = TakagiSugenoClassifier(
            n_rules=n_rules, m=m,
            max_iter_fcm=MAX_ITER_FCM, random_state=RANDOM_STATE
        )
        modelo.fit(X_train, y_train)
        
        y_pred_proba = modelo.predict_proba(X_test)
        y_pred = modelo.predict(X_test)
        metricas = calcular_metricas(y_test, y_pred, y_pred_proba)
        
        cm_total = confusion_matrix(y_test, y_pred, labels=classes)
        
        if config.PRINT_OPTION:
            print(f"\n  Divisão Treino/Teste:")
            print(f"    Treino: {len(X_train):,} | Teste: {len(X_test):,}")
            print(f"    Acurácia (micro): {metricas['acuracia']:.4f} | Acurácia (macro): {metricas['acuracia_macro']:.4f} "
                  f"| Recall: {metricas['recall']:.4f} | AUC: {metricas['auc']:.4f} | F1: {metricas['f1_score']:.4f}")
                  
        medias = pd.Series(metricas)
        desvios = pd.Series({k: 0.0 for k in metricas.keys()})
        melhor_modelo = modelo
        melhor_dados = (X_train, X_test, y_train, y_test, y_pred, y_pred_proba)

    if config.PRINT_OPTION:
        print(f"\n  {'─'*50}")
        if config.METODO_VALIDACAO == 'kfold':
            print(f"  RESULTADOS MÉDIOS ({N_FOLDS}-Folds):")
        else:
            print(f"  RESULTADOS (Holdout 80/20):")
        print(f"  {'─'*50}")
        print(f"    Acurácia (micro): {medias['acuracia']:.4f}" + (f" ± {desvios['acuracia']:.4f}" if config.METODO_VALIDACAO == 'kfold' else ""))
        print(f"    Acurácia (macro): {medias['acuracia_macro']:.4f}" + (f" ± {desvios['acuracia_macro']:.4f}" if config.METODO_VALIDACAO == 'kfold' else ""))
        print(f"    Recall          : {medias['recall']:.4f}" + (f" ± {desvios['recall']:.4f}" if config.METODO_VALIDACAO == 'kfold' else ""))
        print(f"    AUC             : {medias['auc']:.4f}" + (f" ± {desvios['auc']:.4f}" if config.METODO_VALIDACAO == 'kfold' else ""))
        print(f"    F1-Score        : {medias['f1_score']:.4f}" + (f" ± {desvios['f1_score']:.4f}" if config.METODO_VALIDACAO == 'kfold' else ""))

    if config.PRINT_OPTION:
        print(f"\n  {'─'*50}")
        print(f"  Relatório de Classificação (Melhor Fold/Modelo):" if config.METODO_VALIDACAO == 'kfold' else "  Relatório de Classificação:")
        print(f"  {'─'*50}")
        _, X_test_best, _, y_test_best, y_pred_best, _ = melhor_dados
        report = classification_report(
            y_test_best, y_pred_best,
            target_names=[f"Classe {c}" for c in classes],
            zero_division=0
        )
        for line in report.split('\n'):
            print(f"  {line}")

        cm_pct = cm_total.astype(float)
        cm_pct = cm_pct / cm_pct.sum(axis=1, keepdims=True) * 100

        print(f"\n  {'─'*50}")
        print(f"  Matriz de Confusão (%, Acumulada/Teste):" if config.METODO_VALIDACAO == 'kfold' else "  Matriz de Confusão (%, Teste):")
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
