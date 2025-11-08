# CLASSIFICATION MODEL TRAINING AND EVALUATION
    
import pandas as pd    
from sklearn.model_selection import train_test_split    
from sklearn.compose import ColumnTransformer    
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder    
from sklearn.pipeline import Pipeline    
from sklearn.metrics import classification_report, confusion_matrix    
from sklearn.linear_model import LogisticRegression    
from catboost import CatBoostClassifier    
import warnings    
warnings.filterwarnings("ignore")
import os      
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score    
    
#  Load dataset      
df = pd.read_csv("feature_engineered_dataset.csv")    
print(" Dataset loaded successfully! Shape:", df.shape)    

# Define Target Variable    
target = "Insurance Claim Used"      
    
if target not in df.columns:    
    raise ValueError(f"Target column '{target}' not found in dataset.")    
    
df = df.dropna(subset=[target])    
    
y = df[target]    
X = df.drop(columns=[target])    
    
# Identify column types        
cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()    
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()    
    
print(f"\nFeatures: {len(X.columns)},  Numeric: {len(num_cols)},  Categorical: {len(cat_cols)}")    

# Preprocessing    
preprocessor = ColumnTransformer(    
    transformers=[    
        ("num", StandardScaler(), num_cols),    
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)    
    ],    
    remainder="drop"    
)    
    
# Encode target if not numeric    
if y.dtype == "object":    
    le = LabelEncoder()    
    y = le.fit_transform(y)    
    label_names = le.classes_    
else:    
    label_names = None    
    
# Train-Test Split        
X_train, X_test, y_train, y_test = train_test_split(    
    X, y, test_size=0.2, random_state=42, stratify=y    
)    
print(f"\nTrain: {X_train.shape},  Test: {X_test.shape}")    
    
#  MODEL 1: Logistic Regression   
log_model = Pipeline(steps=[    
    ("preprocessor", preprocessor),    
    ("classifier", LogisticRegression(max_iter=1000))    
])    
log_model.fit(X_train, y_train)    
pred_log = log_model.predict(X_test)    
print("\nLOGISTIC REGRESSION RESULTS")    
print(confusion_matrix(y_test, pred_log))    
print(classification_report(y_test, pred_log, target_names=label_names))    
    
# MODEL 2: CatBoost       
cat_features = [X.columns.get_loc(col) for col in cat_cols]    
cat_model = CatBoostClassifier(    
    iterations=300,    
    learning_rate=0.05,    
    depth=6,    
    random_seed=42,    
    verbose=0    
)    
cat_model.fit(X_train, y_train, cat_features=cat_features)    
pred_cat = cat_model.predict(X_test)    
print("\nCATBOOST RESULTS")    
print(confusion_matrix(y_test, pred_cat))    
print(classification_report(y_test, pred_cat, target_names=label_names))    
    
# SIMPLE POST-EVALUATION COMPARISON       
    
acc_log  = accuracy_score(y_test, pred_log)    
prec_log = precision_score(y_test, pred_log, average="weighted", zero_division=0)    
rec_log  = recall_score(y_test, pred_log, average="weighted", zero_division=0)    
f1_log   = f1_score(y_test, pred_log, average="weighted", zero_division=0)    
    
acc_cat  = accuracy_score(y_test, pred_cat)    
prec_cat = precision_score(y_test, pred_cat, average="weighted", zero_division=0)    
rec_cat  = recall_score(y_test, pred_cat, average="weighted", zero_division=0)    
f1_cat   = f1_score(y_test, pred_cat, average="weighted", zero_division=0)    
    
comparison_df = pd.DataFrame({    
    "Model": ["Logistic Regression", "CatBoost"],    
    "Accuracy": [acc_log, acc_cat],    
    "Precision (weighted)": [prec_log, prec_cat],    
    "Recall (weighted)": [rec_log, rec_cat],    
    "F1 (weighted)": [f1_log, f1_cat]    
})    
    
print("\nMODEL PERFORMANCE COMPARISON (Test Set):")    
print(comparison_df.round(3).to_string(index=False))    
    
best_idx = comparison_df["F1 (weighted)"].idxmax()    
if comparison_df["F1 (weighted)"].iloc[0] == comparison_df["F1 (weighted)"].iloc[1]:    
    best_idx = comparison_df["Accuracy"].idxmax()    
    
best_model = comparison_df.loc[best_idx, "Model"]    
    
print(f"\nBest performing model: {best_model}")    
    
print("\nInterpretation:")    
print("- Higher Accuracy = more overall correct predictions.")    
print("- Weighted Precision/Recall/F1 account for class imbalance.")    
print(f"- Based on these metrics, {best_model} performed better.")    
    
# SIMPLE INTERPRETATION (Feature Importance)     
print("\n TOP 10 MOST IMPORTANT FEATURES (CatBoost):")    
feature_importance = pd.DataFrame({    
    "Feature": X.columns,    
    "Importance": cat_model.get_feature_importance()    
}).sort_values(by="Importance", ascending=False)    
    
print(feature_importance.head(10).to_string(index=False))    
    
print("\n Interpretation:")    
print("These are the features that most strongly influence insurance claim usage.")

# EXPORT RESULTS TO CSV
os.makedirs("classification_exports", exist_ok=True)

# Save model comparison table
comparison_df.to_csv("classification_exports/classification_model_comparison.csv", index=False)
print("Saved classification_model_comparison.csv")

# Save confusion matrices
cm_log = pd.DataFrame(confusion_matrix(y_test, pred_log))
cm_cat = pd.DataFrame(confusion_matrix(y_test, pred_cat))

cm_log.to_csv("classification_exports/logistic_regression_confusion_matrix.csv", index=False)
cm_cat.to_csv("classification_exports/catboost_confusion_matrix.csv", index=False)

print("Saved confusion matrices")

# Save feature importance
feature_importance.to_csv("classification_exports/catboost_feature_importance.csv", index=False)
print("Saved catboost_feature_importance.csv")

# Save predictions
pred_df = pd.DataFrame({
    "Actual": y_test,
    "Logistic_Pred": pred_log,
    "CatBoost_Pred": pred_cat
})
pred_df.to_csv("classification_exports/classification_predictions.csv", index=False)
print("Saved classification_predictions.csv")

# Save best model
pd.DataFrame({"Best_Model": [best_model]}).to_csv("classification_exports/best_classification_model.csv", index=False)
print("Saved best_classification_model.csv")

print("\nAll classification results exported successfully!")