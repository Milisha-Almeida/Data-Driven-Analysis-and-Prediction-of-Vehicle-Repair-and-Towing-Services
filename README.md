# Data-Driven-Analysis-and-Prediction-of-Vehicle-Repair-and-Towing-Services
**MSc Data Science (Part 1)** | **Semester I** | **Year:** 2025-26  
**Institution:** Goa Business School,Goa University  
 
---  
 **Team Members**

| Name               | Roll No. |
|-------------------|----------|
| Milisha Almeida     | 2502     |
|shravani Desai    | 2509     |
| Janhavi Naik    | 2511    |
| Reena Koranga    | 2504     |

---
## Project Overview
This project analyzes vehicle service data to understand the key drivers of repair costs and workshop efficiency. Using data analysis, machine learning, and clustering, it provides insights to:

- Predict vehicle service costs
- Classify insurance claim usage
- Identify service clusters for workflow optimization

The project helps service centers improve cost estimation, plan workloads smarter, and deliver better customer experiences.

---
## Tools and Technologies Used
- Python 3.12.0
- Pandas, NumPy
- SciPy, Statsmodels
- Matplotlib
- Scikit-learn, CatBoost
- OS (file handling), Warnings (to manage system warnings)
- Streamlit (for interactive dashboard)
- ---
## Dataset
- The main dataset is `enhanced_motor_vehicle_repair_towing_dataset.csv`
- Additional CSVs are generated :
  - Sorted dataset: `sorted_motor_vehicle_dataset` (latest 15,000 records)
  - Cleaned dataset: `cleaned_motor_vehicle_dataset.csv` (removing duplicates, fixing missing values, correcting dates, and handling outliers which is used for EDA and statistical tests)
  - Feature Engineered Dataset – `feature_engineered_dataset.csv` (new features such as Cost Difference, Cost Ratio, Time_Diff_Days, and Mileage Level)
  - EDA (descriptive_statistics.csv , numeric_correlation.csv, statistical_test_results.csv)
  - Regression results (best_regression_model.csv, regression_model_comparison.csv, regression_predictions.csv, rf_feature_importance.csv)
  - Classification results (best_classification_model.csv, catboost_confusion_matrix.csv, catboost_feature_importance.csv, classification_model_comparison.csv, classification_predictions.csv,logistic_regression_confusion_matrix.csv)
  - Clustering results (cluster_interpretation.csv, cluster_model_comparison.csv, final_clustered_dataset.csv, gmm_cluster_means.csv, kmeans_cluster_means.csv)
---
## Project Features
- **Dataset Overview:** Explore dataset shape, missing values, and column details.
- **EDA (Exploratory Data Analysis):** 
  - Descriptive statistics and correlation analysis
  - Visualizations: histograms, bar plots, pie charts, line plots, heatmaps
- **Regression:** 
  - Predict total service cost
  - Linear Regression and Random Forest models
  - Compare models with MAE, MSE, RMSE, R²
- **Classification:** 
  - Predict insurance claim usage
  - Logistic Regression & CatBoost
  - Confusion matrix and feature importance analysis
- **Clustering:** 
  - Group similar service records using K-Means and GMM
  - Identify patterns in service cost, duration, and mileage
  - Visualize cluster means and distributions
- **Final Insights:** 
  - Key factors influencing service cost
  - Cluster-based workflow recommendations
  - Model performance and actionable business strategies
---
## Key Insights
- Estimated Cost, Service Duration, and Mileage are main drivers of Total Cost.
- Cluster 0: Routine, low-cost services → ideal for fast-track workflow.
- Cluster 1: Complex, high-cost services → needs expert technicians and proper planning.
- Random Forest outperformed Linear Regression for cost prediction.
- CatBoost outperformed Logistic Regression for classification of insurance usage.
- ML insights can optimize staffing, parts inventory, and workflow segmentation.

