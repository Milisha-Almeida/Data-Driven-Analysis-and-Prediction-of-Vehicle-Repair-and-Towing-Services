# REGRESSION MODEL TRAINING AND EVALUATION

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#LOAD & SPLIT DATA
df = pd.read_csv("feature_engineered_dataset.csv")
print("Dataset loaded successfully!")
print("Shape:", df.shape)

# Target variable
target = "Total Cost"

# Separate features and target
X = df.drop(columns=[target])
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n Data successfully split!")
print("Training set shape :", X_train.shape)
print("Testing set shape  :", X_test.shape)
print("Target variable    :", target)
print("Training target size:", y_train.shape)
print("Testing target size :", y_test.shape)

# LINEAR REGRESSION MODEL

# Identify column types
cat_cols = X_train.select_dtypes(include="object").columns.tolist()
num_cols = X_train.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop"
)

# Pipeline: preprocessing + Linear Regression
lr_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

print("\nTraining Linear Regression model...")
lr_model.fit(X_train, y_train)
print("Linear Regression model training complete!")

# Predict
y_pred_lr = lr_model.predict(X_test)

# Evaluate
mae_lr  = mean_absolute_error(y_test, y_pred_lr)
mse_lr  = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr   = r2_score(y_test, y_pred_lr)

print("\n=== LINEAR REGRESSION RESULTS ===")
print(f"MAE  : {mae_lr:.2f}")
print(f"MSE  : {mse_lr:.2f}")
print(f"RMSE : {rmse_lr:.2f}")
print(f"R²   : {r2_lr:.4f}")

# RANDOM FOREST REGRESSOR MODEL

# Preprocessing for tree model (no scaling)
rf_preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder="drop"
)

# Random Forest pipeline
rf_model = Pipeline(steps=[
    ("preprocessor", rf_preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    ))
])

print("\nTraining Random Forest Regressor model...")
rf_model.fit(X_train, y_train)
print("Random Forest Regressor training complete!")

# Predict
y_pred_rf = rf_model.predict(X_test)

# Evaluate
mae_rf  = mean_absolute_error(y_test, y_pred_rf)
mse_rf  = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf   = r2_score(y_test, y_pred_rf)

print("\n=== RANDOM FOREST REGRESSOR RESULTS ===")
print(f"MAE  : {mae_rf:.2f}")
print(f"MSE  : {mse_rf:.2f}")
print(f"RMSE : {rmse_rf:.2f}")
print(f"R²   : {r2_rf:.4f}")

print("\nBoth models trained and evaluated successfully!")


# SIMPLE COMPARISON

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [mae_lr, mae_rf],
    "MSE": [mse_lr, mse_rf],
    "RMSE": [rmse_lr, rmse_rf],
    "R2 Score": [r2_lr, r2_rf]
})

print("\nMODEL PERFORMANCE COMPARISON:")
print(results.round(3).to_string(index=False))

best_model = results.loc[results["R2 Score"].idxmax(), "Model"]
print(f"\n Best performing model based on R²: {best_model}")

# SHORT INTERPRETATION
def interpret_regression_results(results_df):
    best_row = results_df.loc[results_df["R2 Score"].idxmax()]
    best_model_name = best_row["Model"]
    r2_best, mae_best, rmse_best = best_row["R2 Score"], best_row["MAE"], best_row["RMSE"]

    other_row = results_df[results_df["Model"] != best_model_name].iloc[0]
    other_model = other_row["Model"]
    r2_other, mae_other, rmse_other = other_row["R2 Score"], other_row["MAE"], other_row["RMSE"]

    print("\nINTERPRETATION (Auto):")
    print(f"- {best_model_name} is the better model for predicting Total Cost.")
    print(f"- Higher R² ({r2_best:.3f} vs {r2_other:.3f}) and lower error.")
    print(f"- MAE: {mae_best:.2f} vs {mae_other:.2f}")
    print(f"- RMSE: {rmse_best:.2f} vs {rmse_other:.2f}")
    if "Random Forest" in best_model_name:
        print("- Non-linear relationships captured better by tree-based models.")
    else:
        print("- Linear relationships dominate the dataset.")

interpret_regression_results(results)

#  EXPORT RESULTS TO CSV

os.makedirs("regression_exports", exist_ok=True)

# Save model comparison table
results.to_csv("regression_exports/regression_model_comparison.csv", index=False)
print("Saved regression_model_comparison.csv")

# Save predictions
pred_df = pd.DataFrame({
    "Actual": y_test.values,
    "Linear_Regression_Pred": y_pred_lr,
    "Random_Forest_Pred": y_pred_rf
})
pred_df.to_csv("regression_exports/regression_predictions.csv", index=False)
print("Saved regression_predictions.csv")

# Save feature importance (Random Forest only)
rf_feature_importance = pd.DataFrame({
    "Feature": rf_model.named_steps["preprocessor"].get_feature_names_out(),
    "Importance": rf_model.named_steps["regressor"].feature_importances_
}).sort_values("Importance", ascending=False)

rf_feature_importance.to_csv("regression_exports/rf_feature_importance.csv", index=False)
print("Saved rf_feature_importance.csv")

# Save best model name
pd.DataFrame({"Best_Model": [best_model]}).to_csv("regression_exports/best_regression_model.csv", index=False)
print("Saved best_regression_model.csv")

print("\nAll regression results exported successfully!")