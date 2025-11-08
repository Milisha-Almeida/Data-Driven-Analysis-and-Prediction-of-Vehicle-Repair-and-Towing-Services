# Data-Driven-Analysis-and-Prediction-of-Vehicle-Repair-and-Towing-Services
**MSc Data Science(Part 1) | Semester I |Year: 2025**  
**Institutions:** Goa University  
**Faculty Guide:** Department of Computer Science

👨‍💻 **Team Members**

| Name               | Roll No. |
|-------------------|----------|
| Milisha Almeida     | 2502     |
|shravani Desai    | 2509     |
| Janhavi Naik    | 2511    |
| Reena Koranga    | 2504     |


## Project Overview
This project analyzes vehicle service data to understand the key drivers of repair costs and workshop efficiency. Using data analysis, machine learning, and clustering, it provides insights to:

- Predict vehicle service costs
- Classify insurance claim usage
- Identify service clusters for workflow optimization

The project helps service centers improve cost estimation, plan workloads smarter, and deliver better customer experiences.

## Technologies Used
- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- Streamlit (for interactive dashboard)
- Scikit-learn (for regression, classification, clustering)
- CatBoost (for classification)

## Dataset
- The main dataset is `feature_engineered_dataset.csv` (vehicle service records including total cost, service duration, vehicle type, mileage, urgency, etc.).
- Additional CSVs are generated for:
  - EDA (descriptive statistics, correlation matrix)
  - Regression results (model predictions, comparison)
  - Classification results (confusion matrix, feature importance)
  - Clustering results (cluster means, final clustered dataset)

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

## Key Insights
- Estimated Cost, Service Duration, and Mileage are main drivers of Total Cost.
- Cluster 0: Routine, low-cost services → ideal for fast-track workflow.
- Cluster 1: Complex, high-cost services → needs expert technicians and proper planning.
- Random Forest outperformed Linear Regression for cost prediction.
- CatBoost outperformed Logistic Regression for classification of insurance usage.
- ML insights can optimize staffing, parts inventory, and workflow segmentation.

