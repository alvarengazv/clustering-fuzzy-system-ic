import config
from config import *
from experiments import *
from data.eda import *
from data.preprocessing import *
from utils.metrics import *
from utils.plots import *
import sys

# Clear the terminal
def clear_terminal():
    for _ in range(100):
        print()
    os.system('cls' if os.name == 'nt' else 'clear')        

# Option of print 
def print_option():
    print("\nDeseja imprimir o andamento completo do programa (1) ou apenas as informações principais (2)? ")
    choice = input().strip()  
    if choice == "1":
        config.PRINT_OPTION = True
        return True
    elif choice == "2":
        config.PRINT_OPTION = False
        return False
    else:
        print("Opção inválida!")
        return print_option() 

# Option of SMOTE-ENN
def smote_option():
    print("\nDeseja aplicar o balanceamento SMOTE-ENN?")
    print("  1 - Sim (Com SMOTE-ENN)")
    print("  2 - Não (Sem SMOTE-ENN, com remoção de outliers)")
    choice = input("Opção: ").strip()
    if choice == "1":
        config.APLICAR_BALANCEAMENTO = True
        config.DATASET_PREPROCESSED_PATH = os.path.join(config.DATASET_DIR, 'base_sintetica_media_preprocessed.csv')
        return True
    elif choice == "2":
        config.APLICAR_BALANCEAMENTO = False
        config.DATASET_PREPROCESSED_PATH = os.path.join(config.DATASET_DIR, 'base_sintetica_media_preprocessed_sem_smote.csv')
        return False
    else:
        print("Opção inválida!")
        return smote_option()

# Option of validation method
def validation_option():
    print("\nEscolha o método de validação:")
    print("  1 - Holdout (Treino 80% / Teste 20%)")
    print("  2 - K-Fold Cross-Validation (5 Folds)")
    choice = input("Opção: ").strip()
    if choice == "1":
        config.METODO_VALIDACAO = 'holdout'
        return 'holdout'
    elif choice == "2":
        config.METODO_VALIDACAO = 'kfold'
        return 'kfold'
    else:
        print("Opção inválida!")
        return validation_option()

# Prepare data and handle preprocessing choices
def preparar_dados_e_preproc():
    if not os.path.exists(config.DATASET_RAW_PATH):
        print(f"O arquivo {config.DATASET_RAW_PATH} não foi encontrado na pasta {config.DATASET_DIR}")
        print(f"Por favor, baixe o dataset e salve-o na pasta {config.DATASET_DIR}.")
        sys.exit()

    path_com_smote = os.path.join(config.DATASET_DIR, 'base_sintetica_media_preprocessed.csv')
    path_sem_smote = os.path.join(config.DATASET_DIR, 'base_sintetica_media_preprocessed_sem_smote.csv')
    
    existe_com = os.path.exists(path_com_smote)
    existe_sem = os.path.exists(path_sem_smote)
    
    rodar_novamente = 's'
    if existe_com or existe_sem:
        print("\nDeseja rodar o pré-processamento novamente? (s/n)")
        rodar_novamente = input("Opção: ").strip().lower()
        while rodar_novamente not in ['s', 'n']:
            print("Opção inválida! Digite 's' para sim ou 'n' para não.")
            rodar_novamente = input("Opção: ").strip().lower()

    if rodar_novamente == 's':
        smote_option()
        executar_preprocessing()
    else:
        print("\nQual base de dados pré-processada existente deseja utilizar?")
        print("  1 - Carregar base COM SMOTE-ENN")
        print("  2 - Carregar base SEM SMOTE-ENN (Com remoção de outliers via IQR)")
        escolha = input("Opção: ").strip()
        while escolha not in ['1', '2']:
            print("Opção inválida! Escolha 1 ou 2.")
            escolha = input("Opção: ").strip()
            
        if escolha == '1':
            config.APLICAR_BALANCEAMENTO = True
            config.DATASET_PREPROCESSED_PATH = path_com_smote
            if not existe_com:
                print("\n[Aviso] O arquivo COM SMOTE-ENN não foi encontrado. Executando pré-processamento...")
                executar_preprocessing()
        else:
            config.APLICAR_BALANCEAMENTO = False
            config.DATASET_PREPROCESSED_PATH = path_sem_smote
            if not existe_sem:
                print("\n[Aviso] O arquivo SEM SMOTE-ENN não foi encontrado. Executando pré-processamento...")
                executar_preprocessing()

# Loading dataset and separate X and y
def carregar_dados():
    if config.PRINT_OPTION:
        print(f"\n{'='*60}")
        print(f"  CARREGAMENTO DOS DADOS")
        print(f"{'='*60}")

    df = pd.read_csv(config.DATASET_PREPROCESSED_PATH)
    
    if config.PRINT_OPTION:
        print(f"\n  Arquivo   : {os.path.basename(config.DATASET_PREPROCESSED_PATH)}")
        print(f"  Amostras  : {df.shape[0]:,}")
        print(f"  Atributos : {ATRIBUTOS}")
        print(f"  Classes   : {sorted(df[COL_CLASSE].unique())}")
        print(f"\n  Distribuição das classes:")
        for cls in sorted(df[COL_CLASSE].unique()):
            n = (df[COL_CLASSE] == cls).sum()
            print(f"    Classe {cls}: {n:>5,} ({n/len(df)*100:.1f}%)")

    # Update config.ATRIBUTOS dynamically based on the dataset
    config.ATRIBUTOS = [c for c in df.columns if c != COL_CLASSE]

    X = df[config.ATRIBUTOS].values
    y = df[COL_CLASSE].values
    return X, y, df

# Option of Correlation
def correlacao_option():
    print("\nDeseja remover atributos com alta correlação (Pearson > 0.8)?")
    print("  1 - Não (Manter todos)")
    print("  2 - Sim (Remover alta correlação)")
    choice = input("Opção: ").strip()
    if choice == "1":
        config.REMOVER_CORRELACIONADOS = False
        return False
    elif choice == "2":
        config.REMOVER_CORRELACIONADOS = True
        return True
    else:
        print("Opção inválida!")
        return correlacao_option()

# Option of execution mode
def modo_execucao_option():
    print("\nEscolha o modo de execução:")
    print("  1 - Interativo (Escolher 1 configuração de pré-processamento e validação)")
    print("  2 - Lote (Rodar TODAS as 8 combinações e exportar resultados)")
    choice = input("Opção: ").strip()
    if choice == "1":
        return 'interativo'
    elif choice == "2":
        return 'lote'
    else:
        print("Opção inválida!")
        return modo_execucao_option()

def executar_pipeline(modo_lote=False):
    # Loading data
    X, y, df = carregar_dados()

    # Cross-validation with default parameters
    medias, desvios, modelo, melhor_dados, cm_total = cross_validation(X, y)

    X_train, X_test, y_train, y_test, y_pred_test, y_pred_proba_test = melhor_dados

    # Print rules if not in batch mode or if print option is enabled
    if not modo_lote or config.PRINT_OPTION:
        modelo.print_rules(feature_names=config.ATRIBUTOS)

    # Generating plots
    plotar_resultados(modelo, X_test, y_test, y_pred_test, y_pred_proba_test, cm_total)

    # Sensitivity analysis with CV
    df_resultados = experimentar_hiperparametros(X, y)

    # Final summary
    if not modo_lote or config.PRINT_OPTION:
        print(f"\n{'═'*60}")
        print(f"  RESUMO FINAL")
        print(f"{'═'*60}")
        print(f"\n  Modelo: Takagi-Sugeno (1ª ordem) com FCM")
        if config.METODO_VALIDACAO == 'kfold':
            print(f"  Validação: Stratified {N_FOLDS}-Fold Cross-Validation")
        else:
            print(f"  Validação: Holdout Treino 80% / Teste 20%")
        print(f"  Configuração: {N_RULES} regras, m={M_FUZZ}")
        print(f"\n  Métricas no conjunto de teste:")
        print(f"    Acurácia (micro): {medias['acuracia']:.4f}")
        print(f"    Acurácia (macro): {medias['acuracia_macro']:.4f}")
        print(f"    Recall          : {medias['recall']:.4f}")
        print(f"    AUC             : {medias['auc']:.4f}")
        print(f"    F1-Score        : {medias['f1_score']:.4f}")
        print(f"\n  Gráficos salvos em: {RESULTS_DIR}/")
        print(f"\n{'═'*60}\n")
        
    return {
        'SMOTE-ENN': 'Sim' if config.APLICAR_BALANCEAMENTO else 'Não',
        'Validacao': config.METODO_VALIDACAO.capitalize(),
        'Remove_Corr': 'Sim' if config.REMOVER_CORRELACIONADOS else 'Não',
        'Acuracia': medias['acuracia'],
        'Acuracia_Macro': medias['acuracia_macro'],
        'Recall': medias['recall'],
        'AUC': medias['auc'],
        'F1_Score': medias['f1_score']
    }

def salvar_resultados_consolidados(resultados_lista):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_res = pd.DataFrame(resultados_lista)
    path_csv = os.path.join(RESULTS_DIR, 'consolidado_resultados.csv')
    df_res.to_csv(path_csv, index=False)
    print(f"\n[Sucesso] Resultados consolidados salvos em: {path_csv}")

# Main function
def main():
    # Clear the terminal
    clear_terminal()
    
    print("\n" + "═" * 60)
    print("  SISTEMA FUZZY: FCM + TAKAGI-SUGENO PARA CLASSIFICAÇÃO")
    print("═" * 60)

    modo = modo_execucao_option()
    
    if modo == 'interativo':
        # Option of print
        print_option()

        # Prepare data and handle preprocessing choices (re-runs or selects existing)
        preparar_dados_e_preproc()
        
        # Option of correlation
        correlacao_option()
        
        # Re-run preprocessing if correlation option changes
        if config.REMOVER_CORRELACIONADOS:
            executar_preprocessing()

        # Option of Validation method
        validation_option()

        # EDA
        eda_option()

        if config.PRINT_OPTION:
            # Clear the terminal again if we want to clean up prompts
            clear_terminal()

        res = executar_pipeline(modo_lote=False)
        salvar_resultados_consolidados([res])
        
    else:
        # Batch Mode
        print("\nIniciando execução em LOTE para as 8 configurações...")
        config.PRINT_OPTION = False # Desativa prints extensos no modo lote por padrão
        
        resultados_totais = []
        
        combinacoes = [
            (True, 'holdout', False),
            (True, 'kfold', False),
            (False, 'holdout', False),
            (False, 'kfold', False),
            (True, 'holdout', True),
            (True, 'kfold', True),
            (False, 'holdout', True),
            (False, 'kfold', True)
        ]
        
        for aplicar_smote, metodo_val, remover_corr in combinacoes:
            config.APLICAR_BALANCEAMENTO = aplicar_smote
            config.METODO_VALIDACAO = metodo_val
            config.REMOVER_CORRELACIONADOS = remover_corr
            
            sufixo_dataset = ""
            if not aplicar_smote:
                sufixo_dataset += "_sem_smote"
            if remover_corr:
                sufixo_dataset += "_corr"
                
            config.DATASET_PREPROCESSED_PATH = os.path.join(config.DATASET_DIR, f'base_sintetica_media_preprocessed{sufixo_dataset}.csv')
            
            # Generate dataset for this specific combination if it doesn't exist
            if not os.path.exists(config.DATASET_PREPROCESSED_PATH):
                executar_preprocessing()
                
            print(f"\nExecutando -> SMOTE: {'Sim' if aplicar_smote else 'Não'} | Validação: {metodo_val} | Remover Corr: {'Sim' if remover_corr else 'Não'}")
            res = executar_pipeline(modo_lote=True)
            resultados_totais.append(res)
            
        salvar_resultados_consolidados(resultados_totais)
        
        print("\nTodas as combinações foram executadas. Verifique a pasta 'resultados' para os gráficos.")

if __name__ == "__main__":
    main()