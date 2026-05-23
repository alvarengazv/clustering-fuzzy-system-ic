import config
from config import *
from experiments import *
from data.eda import *
from data.preprocessing import *
from utils.metrics import *
from utils.plots import *

# Clear the terminal
def clear_terminal():
    for _ in range(100):
        print()
    os.system('cls' if os.name == 'nt' else 'clear')        

# Option of print 
def print_option():
    print("\nDeseja imprimir o andamento completo do programa (1) ou apenas as informações principais (2)? ")
    choice = input()  
    if choice == "1":
        config.PRINT_OPTION = True
        return True
    elif choice == "2":
        config.PRINT_OPTION = False
        return False
    else:
        print("Opção inválida!")
        return print_option() 

# Loading dataset and separate X and y
def carregar_dados():
    print(f"\n{'='*60}")
    print(f"  CARREGAMENTO DOS DADOS")
    print(f"{'='*60}")

    df = pd.read_csv(DATASET_PREPROCESSED_PATH)
    print(f"\n  Arquivo   : {os.path.basename(DATASET_PREPROCESSED_PATH)}")
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

# Main function
def main():
    # Clear the terminal
    clear_terminal()
    
    print("\n" + "═" * 60)
    print("  SISTEMA FUZZY: FCM + TAKAGI-SUGENO PARA CLASSIFICAÇÃO")
    print("═" * 60)

    # Preprocessing
    preprocessing_option()

    # EDA
    eda_option()

    # Option of print
    print_option()

    # Clear the terminal
    clear_terminal()

    # Loading data
    X, y, df = carregar_dados()

    # Cross-validation with default parameters
    medias, desvios, modelo, melhor_dados, cm_total = cross_validation(X, y)

    X_train, X_test, y_train, y_test, y_pred_test = melhor_dados

    # Print rules
    modelo.print_rules(feature_names=ATRIBUTOS)

    # Generating plots
    plotar_resultados(modelo, X_test, y_test, y_pred_test, cm_total)

    # Sensitivity analysis with CV
    df_resultados = experimentar_hiperparametros(X, y)

    # Final summary
    print(f"\n{'═'*60}")
    print(f"  RESUMO FINAL")
    print(f"{'═'*60}")
    print(f"\n  Modelo: Takagi-Sugeno (1ª ordem) com FCM")
    print(f"  Validação: {N_FOLDS}-Fold Cross-Validation (80/20)")
    print(f"  Configuração: {N_RULES} regras, m={M_FUZZ}")
    print(f"\n  Métricas médias ({N_FOLDS} folds):")
    print(f"    Acurácia (micro): {medias['acuracia']:.4f} ± {desvios['acuracia']:.4f}")
    print(f"    Acurácia (macro): {medias['acuracia_macro']:.4f} ± {desvios['acuracia_macro']:.4f}")
    print(f"    RSE             : {medias['rse']:.4f} ± {desvios['rse']:.4f}")
    print(f"    RMSE            : {medias['rmse']:.4f} ± {desvios['rmse']:.4f}")
    print(f"    F1-Score        : {medias['f1_score']:.4f} ± {desvios['f1_score']:.4f}")
    print(f"\n  Gráficos salvos em: {RESULTS_DIR}/")
    print(f"\n{'═'*60}\n")

if __name__ == "__main__":
    main()