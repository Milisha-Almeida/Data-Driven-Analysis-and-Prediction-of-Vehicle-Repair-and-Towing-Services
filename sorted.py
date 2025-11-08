import pandas as pd

# Load the dataset
df = pd.read_csv("enhanced_motor_vehicle_repair_towing_dataset.csv")

# Convert date columns to datetime
for col in ["Repair Date", "Towing Date"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# Sort in descending order by both dates
df_sorted = df.sort_values(by=["Repair Date", "Towing Date"], ascending=[False, False])

# Keep only the first 15,000 rows
df_top_15k = df_sorted.head(15000)

# Save the result
df_top_15k.to_csv("sorted_motor_vehicle_dataset.csv", index=False)
print("Sorted dataset saved as sorted_motor_vehicle_dataset.csv")