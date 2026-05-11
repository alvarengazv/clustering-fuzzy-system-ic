import pandas as pd
import os

# Get the current file path
current_file_path = os.path.abspath(__file__)
# Get the parent directory path
parent_dir_path = os.path.dirname(os.path.dirname(current_file_path))
# Path to the dataset
dataset_path = os.path.join(parent_dir_path, '../dataset/base_sintetica_media.csv')

# Read the dataset
df = pd.read_csv(dataset_path)

# Get the columns of the dataset
print(f"The columns of the dataset are: {df.columns.tolist()}")

# Get the class distribution
print(f"The class distribution is: \n {df['classe'].value_counts()}")

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

# how much lines per class with missing values
print(f"The number of lines per class with missing values is:\n{missing_values_lines['classe'].value_counts()}")

# Cut randomly 2000 lines of classes 1, 3 and 4, because class 2 has only 1000 samples
# From class 4, ensure the missing values lines are included

df_res = df[df['classe'] == 2]

for cls in [1, 3, 4]:
    df_cls = df[df['classe'] == cls]
    if cls == 4:
        df_missing = df_cls[df_cls.isnull().any(axis=1)]
        df_valid = df_cls.dropna()
        df_res = pd.concat([df_res, df_missing, df_valid.sample(1000 - len(df_missing))])
    else:
        df_res = pd.concat([df_res, df_cls.sample(1000)])

# Check the number of samples per class in the resulting dataset
print(f"The class distribution is: \n {df_res['classe'].value_counts()}")