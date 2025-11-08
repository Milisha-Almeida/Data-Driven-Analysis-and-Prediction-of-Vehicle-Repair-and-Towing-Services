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

|  **#** | **Insight**                                                                                                                                                    | **Derived From**                                                  |
| :----: | :------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------- |
|  **1** | **Estimated Cost, Service Duration Hours, and Mileage at Service** are the **main factors influencing Total Cost.**                                            | 🔹 *Exploratory Data Analysis (EDA)* & 🔹 *Regression Modeling*   |
|  **2** | **High-cost jobs are often underestimated** — a large gap exists between Estimated and Total Cost for expensive, complex repairs.                              | 🔹 *Feature Engineering* (Cost Difference feature) & 🔹 *EDA*     |
|  **3** | **Vehicle Type affects cost and duration** — trucks/buses have higher repair costs and longer service times than cars or bikes.                                | 🔹 *EDA* (grouped analysis & ANOVA tests)                         |
|  **4** | **Towed vehicles cost more on average** — towing indicates higher severity or complexity.                                                                      | 🔹 *EDA* (mean comparison between Towed vs Non-Towed jobs)        |
|  **5** | **Two main service clusters identified:**  <br>• Cluster 0 → Routine, low-cost, short-duration jobs. <br>• Cluster 1 → Complex, high-cost, long-duration jobs. | 🔹 *Clustering Analysis* (K-Means & Gaussian Mixture Models)      |
|  **6** | **Random Forest Regressor performed best** for predicting Total Cost — handles non-linear patterns better than Linear Regression.                              | 🔹 *Regression Model Evaluation*                                  |
|  **7** | **CatBoost Classifier performed best** for predicting Insurance Claim usage — handled categorical data more effectively than Logistic Regression.              | 🔹 *Classification Model Evaluation*                              |
|  **8** | **Insurance usage doesn’t significantly change Total Cost** — claims are used across various job types, not just high-cost ones.                               | 🔹 *EDA* & 🔹 *Classification Results*                            |
|  **9** | **High mileage + long repair duration = higher final cost** — older vehicles require more labor and parts.                                                     | 🔹 *Regression Model Insights* & 🔹 *EDA Correlation Matrix*      |
| **10** | **Daily/weekly service trends are consistent**, showing predictable busy periods useful for staffing and parts management.                                     | 🔹 *EDA Time Analysis (Repair Date trends)*                       |
| **11** | **Top-rated technicians complete complex jobs faster**, showing the value of skilled allocation.                                                               | 🔹 *EDA* (Technician Rating vs Duration/Cost correlation)         |
| **12** | **Clusters can be used for workflow segmentation:** fast-track lane for routine jobs and expert lane for complex ones.                                         | 🔹 *Clustering Results* & 🔹 *Final Insights Section (Dashboard)* |
