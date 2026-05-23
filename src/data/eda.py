import pandas as pd
import os
from config import *

def separador(titulo: str):
    print(f"\n  -- {titulo} ---------------")
    
def classificar_atributo(serie: pd.Series) -> str:
    serie = serie.dropna()
    n_unicos = serie.nunique()

    if pd.api.types.is_bool_dtype(serie) or n_unicos == 2:
        return "Binário"
        
    if (
        pd.api.types.is_object_dtype(serie) 
        or pd.api.types.is_string_dtype(serie) 
        or isinstance(serie.dtype, pd.CategoricalDtype)
    ):
        if isinstance(serie.dtype, pd.CategoricalDtype) and serie.dtype.ordered:
            return "Ordinal"
        return "Nominal"

    if pd.api.types.is_numeric_dtype(serie):
        if serie.min() < 0:
            return "Intervalo"
        if pd.api.types.is_integer_dtype(serie):
            return "Discreto"
            
        if pd.api.types.is_float_dtype(serie):
            return "Contínuo"

    return "Desconhecido"



def informacoes(df: pd.DataFrame):
    # Get the columns of the dataset
    print(f"The columns of the dataset are: {df.columns.tolist()}")

    # Get the class distribution
    print(f"The class distribution is: \n {df[COL_CLASSE].value_counts().sort_index()}")

    # Get the descriptive statistics of the dataset
    print(f"The descriptive statistics of the dataset are: \n {df.describe()}")

    # Get the number of samples in the dataset
    print(f"The number of samples in the dataset is: {df.shape[0]}")

    # Check if it has any missing values
    missing_values_columns = df.isnull().sum()
    missing_values_columns = missing_values_columns[missing_values_columns > 0]
    print(f"The number of missing values in each column is:\n{missing_values_columns}")

    # Get the percentage of missing values in each column
    missing_values_percentage = missing_values_columns / df.shape[0] * 100
    print(f"The percentage of missing values in each column is:\n{missing_values_percentage}")

    # Get the lines with missing values
    missing_values_lines = df[df.isnull().any(axis=1)]
    print(f"The lines with missing values are:\n{missing_values_lines}")

    # How much lines has missing values
    print(f"The number of lines with missing values is: {missing_values_lines.shape[0]}")

    # How much lines doesn't have missing values
    print(f"The number of lines without missing values is: {df.shape[0] - missing_values_lines.shape[0]}")

    # How much lines per class with missing values
    if not missing_values_lines.empty:
        print(f"The number of lines per class with missing values is:\n{missing_values_lines['classe'].value_counts().sort_index()}")
    else:
        print("No lines with missing values per class.")

def caracterizacao(df: pd.DataFrame):
    separador("CARACTERIZAÇÃO GERAL DO DATASET")

    print(f"\n  Instâncias : {df.shape[0]:>10,}")
    print(f"  Atributos  : {df.shape[1]:>10,}")
    print(f"  Memória    : {df.memory_usage(deep=True).sum() / 1e6:>10.2f} MB")

    print("\n\n  --> TAXONOMIA DOS ATRIBUTOS")
    print(f"  {'Atributo':<25} {'Dtype':<12} {'Tipo':<12} {'#Únicos':>8} {'%Nulos':>8}")
    print("  " + "-" * 70)

    linhas = []
    for col in df.columns:
        tipo = classificar_atributo(df[col])
        n_unicos = df[col].nunique()
        pct_nulo = df[col].isna().mean() * 100
        print(f"  {col:<25} {str(df[col].dtype):<12} {tipo:<12} "
              f"{n_unicos:>8,} {pct_nulo:>7.2f}%")
        linhas.append({"Atributo": col, "Tipo": tipo})

    df_tax = pd.DataFrame(linhas)
    contagem = df_tax["Tipo"].value_counts()
    print("\n  --> RESUMO:")
    for tipo, qtd in contagem.items():
        print(f"    {tipo:<12}: {qtd} atributo(s)")

    print("\n\n\n")

    return df_tax

def variavel_alvo(df: pd.DataFrame):
    separador(f"ANÁLISE DA VARIÁVEL ALVO - \"{COL_CLASSE}\"")

    vc = df[COL_CLASSE].value_counts().sort_index()
    vc_pct = df[COL_CLASSE].value_counts(normalize=True).sort_index() * 100
 
    print(f"\n  {'Classe':<10} {'Contagem':>12} {'%':>8}")
    print("  " + "-" * 34)
    for cls in vc.index:
        print(f"  {str(cls):<10} {vc[cls]:>12,} {vc_pct[cls]:>7.2f}%")
 
    razao = vc.max() / vc.min()
    print(f"\n  Razão de desbalanceamento (max/min): {razao:.1f}:1")
    if razao > 3:
        print(f"  --> A variável alvo ( {COL_CLASSE} ) está desbalanceada")
    elif razao > 1.5:
        print(f"  --> A variável alvo ( {COL_CLASSE} ) apresenta algum desbalanceamento.")
    else:
        print(f"  A variável alvo ( {COL_CLASSE} ) parece razoavelmente balanceada.")

    print("\n\n\n")

def distribuicoes_numericas(df: pd.DataFrame):
    separador("DISTRIBUIÇÕES DOS ATRIBUTOS NUMÉRICOS")

    num_cols = [c for c in ATRIBUTOS if c in df.columns]

    if not num_cols:
        print("  Nenhum atributo numérico encontrado.")
        return

    print("\n  Assimetria (skewness) dos atributos numéricos:")
    print(f"    {'Atributo':<25} {'Skewness':>10}  {'Interpretação'}")
    print("    " + "-" * 55)
    for col in num_cols:
        sk = df[col].skew()
        if abs(sk) < 0.5:
            interp = "simétrico"
        elif abs(sk) < 1:
            interp = "assimétrico moderado"
        else:
            interp = "assimétrico forte"
        print(f"    {col:<25} {sk:>10.4f}  {interp}")

    print("\n  Curtose (kurtosis) dos atributos numéricos:")
    print(f"    {'Atributo':<25} {'Kurtosis':>10}  {'Interpretação'}")
    print("    " + "-" * 55)
    for col in num_cols:
        kt = df[col].kurtosis()
        if abs(kt) < 1:
            interp = "mesocúrtica (normal)"
        elif kt > 1:
            interp = "leptocúrtica (caudas pesadas)"
        else:
            interp = "platicúrtica (caudas leves)"
        print(f"    {col:<25} {kt:>10.4f}  {interp}")

    print("\n  Estatísticas por classe:")
    for col in num_cols:
        print(f"\n    --> {col}:")
        print(f"      {'Classe':<10} {'Média':>12} {'Desvio':>12} {'Mín':>12} {'Máx':>12}")
        print("      " + "-" * 52)
        for cls in sorted(df[COL_CLASSE].unique()):
            subset = df[df[COL_CLASSE] == cls][col]
            print(f"      {str(cls):<10} {subset.mean():>12.4f} {subset.std():>12.4f} {subset.min():>12.4f} {subset.max():>12.4f}")

    print("\n\n\n")

def qualidade_dados(df: pd.DataFrame):
    separador("QUALIDADE DOS DADOS")

    print("\n  --> VALORES AUSENTES")
    df_nulos = (pd.DataFrame({
        "Ausentes": df.isna().sum(),
        "%": df.isna().mean() * 100
    }).query("Ausentes > 0").sort_values("%", ascending=False))

    if df_nulos.empty:
        print("    Sem valores ausentes.")
    else:
        print(f"    {len(df_nulos)} coluna(s) com ausentes:")
        print(f"    {'Atributo':<25} {'Ausentes':>12}  {'%':>8}")
        print("    " + "-" * 47)
        for col_name, row in df_nulos.iterrows():
            ausentes = int(row['Ausentes'])
            pct = row['%']
            print(f"    {col_name:<25} {ausentes:>12,}  {pct:>7.2f}%")

    print("\n\n  --> DUPLICATAS")
    dup = df.duplicated().sum()
    if dup > 0:
        print(f"    {dup:,} linhas duplicadas ({dup/len(df)*100:.3f}%)")
    else:
        print("    Nenhuma duplicata.")

    print("\n\n  --> DETECÇÃO DE OUTLIERS (Regra 3-sigma)")
    num_cols = [c for c in ATRIBUTOS if c in df.columns]
    dados_outliers = []

    for col in num_cols:
        serie = df[col].dropna()
        mu = serie.mean()
        sigma = serie.std()
        outliers = ((serie < mu - 3 * sigma) | (serie > mu + 3 * sigma)).sum()
        pct = (outliers / len(serie)) * 100
        
        if outliers > 0: 
            dados_outliers.append({ "Atributo": col, "Outliers": outliers, "%": pct })

    if dados_outliers:
        df_out = pd.DataFrame(dados_outliers).sort_values("%", ascending=False)
        print(f"  Encontrados outliers em {len(df_out)} colunas:")
        print("  " + "-" * 50)
        print(df_out.to_string(index=False))
    else:
        print("    Nenhum outlier severo detectado pela regra 3-sigma.")

    print("\n  --> DETECÇÃO DE OUTLIERS (IQR)")
    dados_outliers_iqr = []

    for col in num_cols:
        serie = df[col].dropna()
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((serie < lower) | (serie > upper)).sum()
        pct = (outliers / len(serie)) * 100

        if outliers > 0:
            dados_outliers_iqr.append({"Atributo": col, "Outliers": outliers, "%": pct})

    if dados_outliers_iqr:
        df_out_iqr = pd.DataFrame(dados_outliers_iqr).sort_values("%", ascending=False)
        print(f"  Encontrados outliers em {len(df_out_iqr)} colunas (IQR):")
        print("  " + "-" * 50)
        print(df_out_iqr.to_string(index=False))
    else:
        print("    Nenhum outlier detectado pela regra IQR.")

    print("\n\n\n")

def correlacoes(df: pd.DataFrame):
    separador("CORRELAÇÕES ENTRE ATRIBUTOS")

    num_cols = [c for c in ATRIBUTOS if c in df.columns]

    corr = df[num_cols].corr()

    print("\n  Matriz de Correlação (Pearson):")
    print(corr.to_string())

    print("\n  Pares com |correlação| > 0.8 (possíveis redundâncias):")
    alto_corr = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            val = abs(corr.iloc[i, j])
            if val > 0.8:
                alto_corr.append((num_cols[i], num_cols[j], corr.iloc[i, j]))
                print(f"    {num_cols[i]:<25} ↔ {num_cols[j]:<25} : {corr.iloc[i, j]:+.4f}")

    if not alto_corr:
        print("  Nenhum par com |correlação| > 0.8 encontrado.")

    print("\n  Pares com |correlação| > 0.5:")
    medio_corr = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            val = abs(corr.iloc[i, j])
            if 0.5 < val <= 0.8:
                medio_corr.append((num_cols[i], num_cols[j], corr.iloc[i, j]))
                print(f"    {num_cols[i]:<25} ↔ {num_cols[j]:<25} : {corr.iloc[i, j]:+.4f}")

    if not medio_corr:
        print("  Nenhum par com 0.5 < |correlação| <= 0.8 encontrado.")

    print("\n\n\n")

def sintese_preprocessamento(df: pd.DataFrame):
    separador("SÍNTESE DA ANÁLISE E PLANO DE AÇÃO")

    n_linhas, n_cols = df.shape
    num_cols = [c for c in ATRIBUTOS if c in df.columns]
    
    nulos = df.isna().sum()
    cols_com_nulos = nulos[nulos > 0].index.tolist()
    dups = df.duplicated().sum()
    
    vc = df[COL_CLASSE].value_counts()
    razao = vc.max() / vc.min()
    n_classes = df[COL_CLASSE].nunique()

    print("    --> RESUMO GERAL DO DATASET:")
    print(f"      Dimensões             : {n_linhas:,} registros x {n_cols} colunas")
    print(f"      Atributos numéricos   : {len(num_cols)}")
    print(f"      Nº de classes         : {n_classes}")
    print(f"      Saúde dos Dados       : {dups} duplicatas | {len(cols_com_nulos)} coluna(s) com nulos")
    print(f"      Desbalanceamento      : {razao:.1f}:1 (max/min)")

    print("\n    --> DISTRIBUIÇÃO POR CLASSE:")
    for cls in sorted(vc.index):
        print(f"      Classe {cls}: {vc[cls]:>6,} amostras ({vc[cls]/n_linhas*100:.1f}%)")

    print("\n\n\n")

def executar_eda():
    # Read the dataset
    df = pd.read_csv(DATASET_RAW_PATH)

    print("=" * 60)
    print("  ANÁLISE EXPLORATÓRIA DE DADOS")
    print(f"  Dataset: base_sintetica_media.csv")
    print(f"  Atributos: {', '.join(ATRIBUTOS)}")
    print(f"  Variável alvo: {COL_CLASSE} (4 classes)")
    print("=" * 60)

    informacoes(df)
    caracterizacao(df)
    variavel_alvo(df)
    distribuicoes_numericas(df)
    qualidade_dados(df)
    correlacoes(df)
    sintese_preprocessamento(df)

    print("Análise Exploratória de Dados concluída!")


# Option to run the eda
def eda_option():
    choice = input("Deseja rodar a EDA (Análise Exploratória de Dados)? (s/n): ")
    if choice.lower() == 's':
        executar_eda()
    else:
        print("EDA não executada.")