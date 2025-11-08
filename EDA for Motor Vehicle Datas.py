#Exploratory Data Analysis (EDA)
#Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load dataset
df = pd.read_csv("cleaned_motor_vehicle_dataset.csv")
print("Dataset shape:", df.shape)

# Convert date columns to datetime
df["Repair Date"] = pd.to_datetime(df["Repair Date"], errors="coerce")
df["Towing Date"] = pd.to_datetime(df["Towing Date"], errors="coerce")

# Basic info
print("\n--- BASIC INFO ---")
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())

# DESCRIPTIVE STATISTICS
os.makedirs("cleaned_data", exist_ok=True)

exclude_cols = ["Service ID", "Repair Date", "Towing Date"]
numeric_cols = df.select_dtypes(include=["number"]).columns.difference(exclude_cols)

desc = df[numeric_cols].describe().T
try:
    desc["mode"] = df[numeric_cols].mode().iloc[0]
except Exception:
    desc["mode"] = pd.NA

print("\n=== DESCRIPTIVE STATISTICS (numeric only) ===")
print(desc.round(2))
desc.to_csv("cleaned_data/descriptive_statistics.csv")
print("Saved descriptive stats to cleaned_data/descriptive_statistics.csv")

# Correlation Analysis
print("\n--- CORRELATION ANALYSIS ---")
corr = df[numeric_cols].corr(numeric_only=True)
print("\nCorrelation with Total Cost:\n", corr["Total Cost"].sort_values(ascending=False))

# SIMPLIFIED STATISTICAL ANALYSIS
print("\n--- STATISTICAL ANALYSIS ---")

stats_results = []  # to collect test results for CSV output

# ANOVA: Urgency Level → Total Cost
try:
    model = ols('Q("Total Cost") ~ C(Q("Urgency Level"))', data=df).fit()
    anova_urgency = sm.stats.anova_lm(model, typ=2)
    p = anova_urgency["PR(>F)"].iloc[0]
    print(f"\n[ANOVA] Urgency Level vs Total Cost: p = {p:.5f}")
    stats_results.append(["ANOVA", "Urgency Level vs Total Cost", p, "Significant" if p < 0.05 else "Not Significant"])
except Exception:
    print("\n[ANOVA] Skipped (missing or invalid columns).")

# T-Test: Insurance Claim Used → Total Cost
try:
    ic = df["Insurance Claim Used"]
    if ic.dtype == "O":
        ic = ic.astype(str).str.strip().str.lower().map({"yes": True, "no": False})
    tmp = df.assign(_icu=ic).dropna(subset=["_icu", "Total Cost"])
    insured = tmp.loc[tmp["_icu"] == True, "Total Cost"]
    not_insured = tmp.loc[tmp["_icu"] == False, "Total Cost"]
    t, p = stats.ttest_ind(insured, not_insured, equal_var=False, nan_policy="omit")
    print(f"\n[T-Test] Insurance Claim Used vs Total Cost: p = {p:.5f}")
    stats_results.append(["T-Test", "Insurance Claim Used vs Total Cost", p, "Significant" if p < 0.05 else "Not Significant"])
except Exception:
    print("\n[T-Test] Skipped (missing or invalid columns).")

# ANOVA: Vehicle Type → Service Duration Hours
try:
    model2 = ols('Q("Service Duration Hours") ~ C(Q("Vehicle Type"))', data=df).fit()
    anova_vehicle = sm.stats.anova_lm(model2, typ=2)
    p2 = anova_vehicle["PR(>F)"].iloc[0]
    print(f"\n[ANOVA] Vehicle Type vs Service Duration Hours: p = {p2:.5f}")
    stats_results.append(["ANOVA", "Vehicle Type vs Service Duration Hours", p2, "Significant" if p2 < 0.05 else "Not Significant"])
except Exception:
    print("\n[ANOVA] Skipped (missing or invalid columns).")

# Chi-Square: Customer Type ↔ Service Type
try:
    ct = pd.crosstab(df["Customer Type"], df["Service Type"])
    chi2, p, dof, ex = stats.chi2_contingency(ct)
    print(f"\n[Chi-Square] Customer Type ↔ Service Type: p = {p:.5f}")
    stats_results.append(["Chi-Square", "Customer Type ↔ Service Type", p, "Significant" if p < 0.05 else "Not Significant"])
except Exception:
    print("\n[Chi-Square] Skipped (missing or invalid columns).")

# Save statistical results summary
results_df = pd.DataFrame(stats_results, columns=["Test Type", "Comparison", "p-value", "Result"])
results_df.to_csv("cleaned_data/statistical_test_results.csv", index=False)
print("\nStatistical test results saved to cleaned_data/statistical_test_results.csv")

print("\n Done! EDA visuals shown above; all numeric and statistical outputs saved in 'cleaned_data/'.")