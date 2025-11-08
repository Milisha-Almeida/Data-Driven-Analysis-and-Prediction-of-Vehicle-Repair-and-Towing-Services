# DATA CLEANING 

import pandas as pd
import numpy as np

# LOAD & INSPECT DATASET
df = pd.read_csv("sorted_motor_vehicle_dataset.csv", low_memory=False)

print("Initial shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nMissing values (top 10):")
print(df.isnull().sum().sort_values(ascending=False).head(10))
print("\nPreview:")
print(df.head())

# HANDLE MISSING VALUES
thresh = int(0.1 * len(df))  # keep columns with at least 10% non-null
df = df.dropna(axis=1, thresh=thresh)

num_cols = df.select_dtypes(include='number').columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# CLEAN & STANDARDIZE TEXT DATA
for col in df.select_dtypes(include='object').columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r'\s+', ' ', regex=True)
    )

# CONVERT DATE / TIME COLUMNS TO REAL datetime64[ns]
date_cols = ['Repair Date', 'Towing Date']

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        print(f"{col} converted successfully — dtype: {df[col].dtype}")

# Drop rows where both dates failed to convert (both NaT)
df = df.dropna(subset=['Repair Date', 'Towing Date'], how='all')

# FIX DATE INCONSISTENCIES (Towing after Repair)
if set(['Repair Date', 'Towing Date']).issubset(df.columns):
    inconsistent_dates = df[df["Towing Date"] > df["Repair Date"]].shape[0]
    if inconsistent_dates > 0:
        print(f"Found {inconsistent_dates} rows with Towing Date after Repair Date — fixing...")
        # Option 1: Swap the dates if repair < towing
        mask = df["Towing Date"] > df["Repair Date"]
        df.loc[mask, ["Repair Date", "Towing Date"]] = df.loc[mask, ["Towing Date", "Repair Date"]].values
        print("Date inconsistencies fixed by swapping values.")

# FIX NUMERIC OUTLIERS (IQR METHOD) + NON-NEGATIVE
numeric_cols = [
    'Total Cost', 'Estimated Cost', 'Service Duration Hours',
    'Mileage at Service', 'Tow Distance Miles'
]
for col in numeric_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
        df[col] = df[col].clip(lower=0)

# REMOVE DUPLICATE ROWS
df.drop_duplicates(inplace=True)
print("After removing duplicates:", df.shape)

# FIX BOOLEAN / CATEGORICAL TYPES
yes_no_map = {
    'yes': True, 'no': False,
    'true': True, 'false': False,
    '1': True, '0': False,
    'y': True, 'n': False,
    'on': True, 'off': False
}
for col in ['Is Warranty Valid', 'Insurance Claim Used', 'Towing Required', 'Parts In Warranty', 'Follow-up Needed']:
    if col in df.columns:
        df[col] = df[col].map(yes_no_map)

# DROP UNINFORMATIVE / UNWANTED COLUMNS
single_val_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
if single_val_cols:
    df.drop(columns=single_val_cols, inplace=True)
    print("Dropped single-value columns:", single_val_cols)

for col in ['Service Status', 'Customer Feedback']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        print(f"Dropped column: {col}")

# FINAL CHECK
print("\nCleaned Data Summary:")
print(df.info())
print("\nMissing values after cleaning (top 10):")
print(df.isnull().sum().sort_values(ascending=False).head(10))
print("\nPreview of cleaned data:")
print(df.head())

# Keep first 15,000 rows
df = df.head(15000)

# FINAL VALIDATION
missing_total = int(df.isna().sum().sum())
dup_total = int(df.duplicated().sum())
uniform_cols_after = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
date_types = {c: str(df[c].dtype) for c in date_cols if c in df.columns}
print("\n=== FINAL VALIDATION ===")
print(f"Missing values (total): {missing_total}")
print(f"Duplicate rows: {dup_total}")
print(f"Uniform columns (should be empty): {uniform_cols_after}")
print(f"Date dtypes: {date_types}")
print("========================\n")

# SAVE CLEANED DATASET
# -----------------------------------------------------------
output_path = "cleaned_motor_vehicle_dataset.csv"
df.to_csv(output_path, index=False, date_format="%Y-%m-%d")

print(f"Cleaned dataset (first 15,000 rows) saved to: {output_path}")
print("Dates converted to datetime64[ns] and stored in ISO format (yyyy-mm-dd).")
print("On reload, use parse_dates=['Repair Date','Towing Date'] to preserve datetime dtypes.")