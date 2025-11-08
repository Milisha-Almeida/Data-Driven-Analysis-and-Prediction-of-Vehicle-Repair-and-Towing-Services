# CLUSTERING MODEL TRAINING AND EVALUATION
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Configuration
FEATURES = ["Total Cost", "Estimated Cost", "Service Duration Hours", "Mileage at Service"]
RANDOM_STATE = 42

# Load and select features
df = pd.read_csv("feature_engineered_dataset.csv")

missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise ValueError(f"These required columns are missing from the CSV: {missing}")

X_df = df[FEATURES].copy()
print("Using features:", FEATURES)

# Impute and scale
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X_df)
X = StandardScaler().fit_transform(X)

#  Helper to print metrics
def show_metrics(model_name, labels):
    if len(set(labels)) <= 1:
        print(f"\n {model_name}: only one cluster or all noise.")
        return
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    print(f"\n {model_name} METRICS:")
    print(f"Silhouette Score:       {sil:.3f}")
    print(f"Davies-Bouldin Index:   {db:.3f}")
    print(f"Calinski-Harabasz:      {ch:.3f}")
    print(f"{model_name} Cluster Counts:")
    print(pd.Series(labels).value_counts().sort_index())

# KMeans (k=2)
kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE)
kmeans_labels = kmeans.fit_predict(X)
show_metrics("K-Means", kmeans_labels)

# Gaussian Mixture (n_components=2)
gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE)
gmm_labels = gmm.fit_predict(X)
show_metrics("Gaussian Mixture", gmm_labels)

# Compare Models Side-by-Side
results = []

def get_metrics_dict(model_name, labels):
    if len(set(labels)) <= 1:
        return {"Model": model_name, "Silhouette": None, "DBI": None, "CHI": None}
    return {
        "Model": model_name,
        "Silhouette": silhouette_score(X, labels),
        "DBI": davies_bouldin_score(X, labels),
        "CHI": calinski_harabasz_score(X, labels)
    }

results.append(get_metrics_dict("K-Means", kmeans_labels))
results.append(get_metrics_dict("Gaussian Mixture", gmm_labels))

comparison_df = pd.DataFrame(results)
print("\nMODEL COMPARISON:")
print(comparison_df)

# Select Best Model
best_model = comparison_df.loc[comparison_df["Silhouette"].idxmax()]
print("\n BEST MODEL BASED ON SILHOUETTE:")
print(best_model)

# Interpretation 

cluster_profiles = X_df.copy()
cluster_profiles["KMeans_Cluster"] = kmeans_labels
cluster_profiles["GMM_Cluster"] = gmm_labels

print("\n K-MEANS CLUSTER PROFILES (Feature Means):")
kmeans_means = cluster_profiles.groupby("KMeans_Cluster").mean(numeric_only=True)
print(kmeans_means)

print("\n GMM CLUSTER PROFILES (Feature Means):")
gmm_means = cluster_profiles.groupby("GMM_Cluster").mean(numeric_only=True)
print(gmm_means)

def auto_interpret(summary_df, model_name):
    print(f"\n AUTOMATIC INTERPRETATION — {model_name}")

    insights = {}
    labels = {}

    for cluster in summary_df.index:
        row = summary_df.loc[cluster]
        interpretation = []

        if row["Total Cost"] > summary_df["Total Cost"].mean():
            interpretation.append("High Total Cost")
        else:
            interpretation.append("Low Total Cost")

        if row["Mileage at Service"] > summary_df["Mileage at Service"].mean():
            interpretation.append("High Mileage Vehicles")
        else:
            interpretation.append("Low Mileage Vehicles")

        if row["Service Duration Hours"] > summary_df["Service Duration Hours"].mean():
            interpretation.append("Long Repair Duration")
        else:
            interpretation.append("Short Repair Duration")

        if row["Estimated Cost"] < row["Total Cost"]:
            interpretation.append("Underestimated Jobs")
        else:
            interpretation.append("Accurately Estimated Jobs")

        insights[cluster] = interpretation

        label = []
        label.append("High-Cost" if "High Total Cost" in interpretation else "Low-Cost")
        label.append("Long" if "Long Repair Duration" in interpretation else "Short")
        label.append("High-Mileage" if "High Mileage Vehicles" in interpretation else "Low-Mileage")

        final_label = " ".join(label) + " Services"
        labels[cluster] = final_label

    for cluster in insights:
        print(f"\nCluster {cluster}:")
        for line in insights[cluster]:
            print(f" - {line}")
        print(f"Assigned Label → {labels[cluster]}")

    return insights, labels

kmeans_insights, kmeans_labels_map = auto_interpret(kmeans_means, "K-Means")
gmm_insights, gmm_labels_map = auto_interpret(gmm_means, "Gaussian Mixture")

cluster_profiles["KMeans_Label"] = cluster_profiles["KMeans_Cluster"].map(kmeans_labels_map)
cluster_profiles["GMM_Label"] = cluster_profiles["GMM_Cluster"].map(gmm_labels_map)

print("\n FINAL INSIGHTS:")
if best_model["Model"] == "K-Means":
    print("-  K-Means produced better-separated clusters.")
else:
    print("-  Gaussian Mixture performed better.")

#  CSV EXPORTS
os.makedirs("clustering_exports", exist_ok=True)

# Save full dataset with cluster labels
final_output = df.copy()
final_output["KMeans_Cluster"] = kmeans_labels
final_output["GMM_Cluster"] = gmm_labels
final_output["KMeans_Label"] = final_output["KMeans_Cluster"].map(kmeans_labels_map)
final_output["GMM_Label"] = final_output["GMM_Cluster"].map(gmm_labels_map)
final_output.to_csv("clustering_exports/final_clustered_dataset.csv", index=False)
print(" Saved final_clustered_dataset.csv")

# Save cluster means
kmeans_means.to_csv("clustering_exports/kmeans_cluster_means.csv")
gmm_means.to_csv("clustering_exports/gmm_cluster_means.csv")
print("Saved cluster means CSV files")

# Save comparison table
comparison_df.to_csv("clustering_exports/cluster_model_comparison.csv", index=False)
print("Saved cluster_model_comparison.csv")

# Save interpretation text
with open("clustering_exports/cluster_interpretation.txt", "w") as f:
    f.write("KMeans Interpretation:\n")
    for c, vals in kmeans_insights.items():
        f.write(f"\nCluster {c}:\n")
        for v in vals:
            f.write(f"- {v}\n")

    f.write("\n\nGMM Interpretation:\n")
    for c, vals in gmm_insights.items():
        f.write(f"\nCluster {c}:\n")
        for v in vals:
            f.write(f"- {v}\n")

print("Saved cluster_interpretation.txt")

print("\nAll clustering CSV files exported successfully!")