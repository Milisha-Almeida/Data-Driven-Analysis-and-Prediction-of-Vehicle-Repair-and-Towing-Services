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
- source: https://www.kaggle.com/datasets/aryan208/motor-vehicle-repair-and-towing-dataset
- Additional CSVs are generated :
  - Sorted dataset: `sorted_motor_vehicle_dataset.csv` (latest 15,000 records)
  - Cleaned dataset: `cleaned_motor_vehicle_dataset.csv` (removing duplicates, fixing missing values, correcting dates, and handling outliers which is used for EDA and statistical tests)
  - Feature Engineered Dataset – `feature_engineered_dataset.csv` (new features such as Cost Difference, Cost Ratio, Time_Diff_Days, and Mileage Level)
  - EDA (`descriptive_statistics.csv` , `numeric_correlation.csv` , `statistical_test_results.csv`)
  - Regression results (`best_regression_model.csv`, `regression_model_comparison.csv`, `regression_predictions.csv`, `rf_feature_importance.csv`)
  - Classification results (`best_classification_model.csv` , `catboost_confusion_matrix.csv` , `catboost_feature_importance.csv` , `classification_model_comparison.csv`, `classification_predictions.csv` ,`logistic_regression_confusion_matrix.csv`)
  - Clustering results (`cluster_interpretation.csv` , `cluster_model_comparison.csv` , `final_clustered_dataset.csv` , `gmm_cluster_means.csv` , `kmeans_cluster_means.csv`)
---
## Project Features
- **Dataset Overview:** Explore dataset shape, missing values, and column details.
- **EDA (Exploratory Data Analysis):** 
  - Descriptive statistics , correlation analysis and statistical analysis
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
  - Group similar service records (estimated cost, total cost, service duration hours, mileage at service) using K-Means and GMM
  - Visualize cluster means and distributions
- **Final Insights:** 
  - Key factors influencing service cost
  - Cluster-based workflow recommendations
  - Model performance and actionable business strategies
---
**Key Insights**
|  No | Insight                                                                                                                                                    | Derived From|                                   
| ----|------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------- |
|  *1* | Estimated Cost, Service Duration Hours, and Mileage at Service are the main factors influencing Total Cost.                                           |  Exploratory Data Analysis (EDA) & Regression Modeling   |
|  *2* | High-cost jobs are often underestimated — a large gap exists between Estimated and Total Cost for expensive, complex repairs.                              | Feature Engineering (Cost Difference feature) &  EDA     |
|  *3* | Vehicle Type affects cost and duration — trucks/buses have higher repair costs and longer service times than cars or bikes.                                | EDA (grouped analysis & ANOVA tests)                         |
|  *4* | Towed vehicles cost more on average — towing indicates higher severity or complexity.                                                                      | EDA (mean comparison between Towed vs Non-Towed jobs)        |
|  *5* | Two main service clusters identified:  <br>• Cluster 0 → Routine, low-cost, short-duration jobs. <br>• Cluster 1 → Complex, high-cost, long-duration jobs. | Clustering Analysis (K-Means & Gaussian Mixture Models)      |
|  *6* | Random Forest Regressor performed best for predicting Total Cost — handles non-linear patterns better than Linear Regression.                              | Regression Model Evaluation                                  |
|  *7* | CatBoost Classifier performed best for predicting Insurance Claim usage — handled categorical data more effectively than Logistic Regression.              | Classification Model Evaluation                              |
|  *8* | Insurance usage doesn’t significantly change Total Cost — claims are used across various job types, not just high-cost ones.                               | EDA & Classification Results|                       
| *9* | Top-rated technicians complete complex jobs faster, showing the value of skilled allocation.                                                               | EDA (Technician Rating vs Duration/Cost correlation) |
