import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# 0.  CONSTANTS
# ---------------------------------------------------------------------
RAW_PATH      = "cross_statement_ratio.csv"
CLEANED_PATH  = "crossstatement_ratio_df_cleaned.csv"
RANDOM_STATE  = 42
MISSING_THR   = 0.50     # drop col if >50 % missing
TARGET_FLAG   = 1

# ---------------------------------------------------------------------
# 1.  LOAD & BASIC FIXES
# ---------------------------------------------------------------------
df = pd.read_csv(RAW_PATH)

print(df.head())
print(df.shape)
print(df['Instrument'].unique().shape)

# ถ้ามีคอลัมน์ Unnamed: 0 ให้ลบทิ้ง (errors='ignore' ป้องกันกรณีไม่มีคอลัมน์นี้)
df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

# -- parse date
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])            # ถ้า Date เพี้ยนให้ทิ้งทั้งแถว

# -- drop very‑sparse cols (> MISSING_THR)
df = df.loc[:, df.isna().mean() < MISSING_THR]

# ---------------------------------------------------------------------
# 2.  IMPUTE NUMERIC BY COMPANY‑MEAN
# ---------------------------------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns

# mean‑impute per Instrument (vectorized)
means = df.groupby('Instrument')[numeric_cols].transform('mean')
df[numeric_cols] = df[numeric_cols].fillna(means)
df[numeric_cols] = df[numeric_cols].fillna(0)      # fallback ถ้ายัง NaN

print(df.isnull().sum())
print(df.shape)

# ---------------------------------------------------------------------
# 3.  TARGET LABEL
# ---------------------------------------------------------------------
fraud_com = {
    'ACAPm.BK',
    'AIE.BK',
    'AJA.BK',
    'EA.BK',
    'GGC.BK',
    'GJS.BK',
    'GL.BK',
    'GSTEEL.BK',
    'IEC.BK^G19',
    'IFEC.BK^G24',
    'KC.BK',
    'NATION.BK',
    'PACE.BK',
    'POLAR.BK^A25',
    'PRO.BK^A25',
    'RICH.BK^H20',
    'STARK.BK^I24',
    'STELLA.BK',
    'STOWERm.BK',
    'TRITN.BK',
    'TUCC.BK^G17',
    'WORLD.BK^I24',
    'EARTH.BK^I19', 
    'EASTW.BK'}


df['Target'] = 0

df.loc[df['Instrument'].isin(fraud_com), 'Target'] = TARGET_FLAG

print(df['Target'].value_counts())

# ---------------------------------------------------------------------
# 4.  SAVE CLEANED CSV
# ---------------------------------------------------------------------
df.to_csv(CLEANED_PATH, index=False)

# ---------------------------------------------------------------------
# 5.  TRAIN / VAL / TEST SPLIT
# ---------------------------------------------------------------------
X = df.drop(columns=['Target'])
y = df['Target']

# first hold‑out test 20 %
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

# then val 20 % of remaining 80 % → 16 % of total
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val)


print(X.shape, y.shape)

## Check the sizes of the datasets
print("Training set size:", X_train.shape[0])
print("Validation set size:", X_val.shape[0])
print("Testing set size:", X_test.shape[0])

print("Train fraud (y=1):", y_train.sum())
print("Val fraud (y=1):", y_val.sum())
print("Test fraud (y=1):", y_test.sum())


# ---------------------------------------------------------------------
# (Option) EXPORT SCALED MATRICES
# ---------------------------------------------------------------------
X_train.to_csv("X_train_CrossStatement_ratio.csv", index=False)
X_val.to_csv("X_val_CrossStatement_ratio.csv", index=False)
X_test.to_csv("X_test_CrossStatement_ratio.csv", index=False)

print("ถึงบรรทัดสุดท้ายแล้วจ้า")