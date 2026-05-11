import pandas as pd
import numpy as np

# Create dummy data
data = {
    'A': np.random.rand(10000),
    'classe': [1]*3000 + [2]*1000 + [3]*3000 + [4]*3000
}
df = pd.DataFrame(data)

# Inject some missing values in class 4
df.loc[9000:9089, 'A'] = np.nan

df_res = df[df['classe'] == 2]

for cls in [1, 3, 4]:
    df_cls = df[df['classe'] == cls]
    if cls == 4:
        df_missing = df_cls[df_cls.isnull().any(axis=1)]
        df_valid = df_cls.dropna()
        df_res = pd.concat([df_res, df_missing, df_valid.sample(2000 - len(df_missing))])
    else:
        df_res = pd.concat([df_res, df_cls.sample(2000)])

print(df_res['classe'].value_counts())
print(df_res.isnull().sum())
