# FEATURE ENGINEERING 

import pandas as pd
import numpy as np

# Load
df = pd.read_csv("cleaned_motor_vehicle_dataset.csv")
print("Shape (raw):", df.shape)

# Cleaning
df.columns = df.columns.str.strip()
df = df.drop_duplicates().reset_index(drop=True)

# Dates → datetime
for c in ["Repair Date", "Towing Date"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# Simple date features
if "Repair Date" in df.columns:
    df["Repair_Year"] = df["Repair Date"].dt.year
    df["Repair_Month"] = df["Repair Date"].dt.month
    df["Repair_Weekday"] = df["Repair Date"].dt.weekday

# Time difference (days)
if {"Repair Date", "Towing Date"}.issubset(df.columns):
    df["Time_Diff_Days"] = (df["Repair Date"] - df["Towing Date"]).dt.days
    # if negative (rare), mark as missing
    df.loc[df["Time_Diff_Days"] < 0, "Time_Diff_Days"] = np.nan

# Cost features
if {"Estimated Cost", "Total Cost"}.issubset(df.columns):
    df["Estimated Cost"] = pd.to_numeric(df["Estimated Cost"], errors="coerce")
    df["Total Cost"] = pd.to_numeric(df["Total Cost"], errors="coerce")
    df["Cost_Difference"] = df["Estimated Cost"] - df["Total Cost"]
    df["Cost_Ratio"] = df["Total Cost"] / df["Estimated Cost"].replace(0, np.nan)

# Mileage features
if "Mileage at Service" in df.columns:
    df["Mileage at Service"] = pd.to_numeric(df["Mileage at Service"], errors="coerce")
    df.loc[df["Mileage at Service"] < 0, "Mileage at Service"] = np.nan  # no negatives
    df["Mileage_Level"] = pd.cut(
        df["Mileage at Service"],
        bins=[0, 20000, 60000, 120000, np.inf],
        labels=["Low", "Medium", "High", "Very High"],
        right=True
    )

# Minimal binary normalization → 0/1
def to01(s):
    return (s.astype(str).str.strip().str.lower()
            .map({"true":1,"yes":1,"y":1,"1":1,"false":0,"no":0,"n":0,"0":0}))
for col in ["Insurance Claim Used", "Follow-up Needed"]:
    if col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
        else:
            df[col] = to01(df[col])

# Coerce key numerics
for c in ["Total Cost","Estimated Cost","Service Duration Hours",
        "Technician Rating","Mileage at Service","Time_Diff_Days",
        "Cost_Difference","Cost_Ratio"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# non-negative for natural-≥0 fields
for c in ["Total Cost","Estimated Cost","Service Duration Hours",
        "Mileage at Service","Time_Diff_Days","Cost_Difference"]:
    if c in df.columns:
        df[c] = df[c].clip(lower=0)

# rating within [0,5]
if "Technician Rating" in df.columns:
    df["Technician Rating"] = df["Technician Rating"].clip(lower=0, upper=5)

# Missing values (simple, consistent)
num_cols = df.select_dtypes(include=["number"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

df[num_cols] = df[num_cols].apply(lambda s: s.fillna(s.median()))
for c in cat_cols:
    if df[c].isna().any():
        mode = df[c].mode(dropna=True)
        df[c] = df[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")

# Save
df.to_csv("feature_engineered_dataset.csv", index=False)
print("Saved → feature_engineered_dataset.csv")
print("Shape (final):", df.shape)
print("Total missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))