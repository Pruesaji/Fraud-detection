import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# 0.  CONSTANTS
# ---------------------------------------------------------------------
RAW_PATH      = "CF_df.csv"
CLEANED_PATH  = "CF_df_cleaned.csv"
RANDOM_STATE  = 42
MISSING_THR   = 0.50     # drop col if >50 % missing
TARGET_FLAG   = 1

# ---------------------------------------------------------------------
# 1.  LOAD & BASIC FIXES
# ---------------------------------------------------------------------
df = pd.read_csv(RAW_PATH)

# print(df.shape)
# print(df['Instrument'].unique().shape)

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
print("Training set size:", X_train.shape)
print("Validation set size:", X_val.shape)
print("Testing set size:", X_test.shape)

print("Train fraud (y=1):", y_train.shape)
print("Val fraud (y=1):", y_val.shape)
print("Test fraud (y=1):", y_test.shape)


def zscore_by_group(train_df, apply_df, group_col, num_cols):
    apply_df = apply_df.copy() # avoid modifying original DataFrame
    stats = train_df.groupby(group_col)[num_cols].agg(['mean', 'std'])
    stats.columns = pd.Index(stats.columns.map('_'.join))     
    merged = apply_df.join(stats, on=group_col)
    for col in num_cols:
        apply_df[col] = (
            (apply_df[col] - merged[f'{col}_mean']) /
            merged[f'{col}_std'].replace(0, 1)              
        )
    return apply_df[num_cols]

# create copies to keep original Instrument col
X_train_scaled = X_train.copy()
X_val_scaled   = X_val.copy()
X_test_scaled  = X_test.copy()

X_train_scaled[numeric_cols] = zscore_by_group(X_train, X_train_scaled, 'Instrument', numeric_cols)
X_val_scaled[numeric_cols]   = zscore_by_group(X_train, X_val_scaled,   'Instrument', numeric_cols)
X_test_scaled[numeric_cols]  = zscore_by_group(X_train, X_test_scaled,  'Instrument', numeric_cols)

# ---------------------------------------------------------------------
# 6. EXPORT SCALED MATRICES
# ---------------------------------------------------------------------
X_train_scaled.to_csv("X_train_scaled_CF.csv", index=False)
X_val_scaled.to_csv("X_val_scaled_CF.csv", index=False)
X_test_scaled.to_csv("X_test_scaled_CF.csv", index=False)

print("ถึงบรรทัดสุดท้ายแล้วจ้า")