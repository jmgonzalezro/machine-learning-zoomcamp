import pandas as pd
import numpy as np


df = pd.read_csv('laptops.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')
cols = [
    'ram',
    'storage',
    'screen',
    'final_price',
]
df = df[cols]
# q1
print(df.isna().sum())

# q2
print(df.ram.mean())
print(df.ram.quantile(0.5))

# q3
np.random.seed(42)
n = len(df)

n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - (n_val + n_test)

idx = np.arange(n)
np.random.shuffle(idx)

df_shuffled = df.iloc[idx]

df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
df_test = df_shuffled.iloc[n_train+n_val:].copy()
