import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Vehicle Service Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
# --- Custom Modern Styling ---
st.markdown("""
<style>
    /* 🌟 Global Page Styling */
    body {
        background-color: #ffffff;  /* Keep main content white for clarity */
        color: #1e293b;
        font-family: 'Segoe UI', sans-serif;
    }

    /* 🧭 Sidebar Styling — Soft Charcoal Theme */
    section[data-testid="stSidebar"] {
        background-color: #374151 !important;  /* dark gray, softer than blue */
        color: #f9fafb !important;
        border-right: 1px solid #475569;
    }

    /* Sidebar headers and text */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label {
        color: #f9fafb !important;  /* pure white text */
        font-weight: 500 !important;
    }

    /* ✨ Sidebar Radio Buttons (for navigation) */
    div[data-testid="stRadio"] label {
        background: none !important;
        border: none !important;
        color: #f9fafb !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        padding: 6px 12px !important;
        border-radius: 6px;
        transition: all 0.2s ease-in-out;
    }

    div[data-testid="stRadio"] label:hover {
        color: #60a5fa !important; /* Soft blue hover */
        transform: translateX(3px);
    }

    div[data-testid="stRadio"] label[data-checked="true"] {
        color: #93c5fd !important; /* Active state brighter blue */
        font-weight: 600 !important;
        border-left: 3px solid #3b82f6 !important;
        background: none !important;
    }
    /* 🏷️ Headers */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-color) !important;  /* Automatically adapts to light/dark mode */
    font-weight: 700;
}



    /* 📊 Metric Cards */
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 10px;
        transition: transform 0.2s ease-in-out;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    }

    /* 📁 Section Banner */
    .section-title {
        background-color: #2563eb10;
        color: #1d4ed8;
        padding: 10px 15px;
        border-left: 5px solid #2563eb;
        border-radius: 8px;
        margin-top: 15px;
        font-weight: 600;
    }

    /* 💬 Info Box */
    .info-box {
        background-color: #e0f2fe;
        border-left: 5px solid #2563eb;
        color: #0c4a6e;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 12px;
    }

    /* Hide Streamlit footer */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)



st.set_page_config(page_title=" Data-Driven Analysis and Prediction of Vehicle Repair and Towing Services.", layout="wide")
st.title("Data-Driven Analysis and Prediction of Vehicle Repair and Towing Services.")
st.write("This project analyzes vehicle service data to understand what drives repair costs and workshop efficiency. It uses data analysis and machine learning to predict service costs, classify job priorities, and group similar repair types inturn helping service centers improve cost estimation, plan workloads smarter, and deliver better customer experiences.")

# Sidebar inputs
DATASET_PATH = st.sidebar.text_input("Main dataset path", value="feature_engineered_dataset.csv")
EDA_DIR = st.sidebar.text_input("EDA CSV folder", value="cleaned_data")
REG_DIR = st.sidebar.text_input("Regression CSV folder", value="regression_exports")
CLS_DIR = st.sidebar.text_input("Classification CSV folder", value="classification_exports")
CLU_DIR = st.sidebar.text_input("Clustering CSV folder", value="clustering_exports")

# Default to Dataset Overview on first load
PAGES = [
    "Dataset Overview",
    "EDA",
    "Regression Results",
    "Classification Results",
    "Clustering Results",
    "Final Insights",
]
if "active_page" not in st.session_state:
    st.session_state.active_page = PAGES[0]  # default

menu = st.sidebar.radio(
    "📌 Navigate",
    PAGES,
    index=PAGES.index(st.session_state.active_page),
    key="menu_radio",
)
st.session_state.active_page = menu

# Helpers
@st.cache_data(show_spinner=False)
def load_df(path: str):
    return pd.read_csv(path)

def load_csv(path: str, name: str):
    if not os.path.exists(path):
        st.warning(f"Missing: {name} — expected at `{path}`.")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read {name}: {e}")
        return None

#  Dataset Overview
if menu == "Dataset Overview":
    st.header(" Dataset Overview")
    df = None
    try:
        df = load_df(DATASET_PATH)
    except Exception as e:
        st.warning(f"Could not load dataset at `{DATASET_PATH}`: {e}")

    if df is None:
        st.info("Provide a valid dataset path in the sidebar.")
    else:
        st.dataframe(df.head(10), width="stretch")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", df.shape[0])
            st.metric("Columns", df.shape[1])
        with col2:
            st.write("**Column Names:**")
            st.write(list(df.columns))
        st.subheader(" Missing Values")
        st.dataframe(df.isnull().sum().reset_index().rename(columns={"index": "Column", 0: "Missing"}))

#  EDA 
elif menu == "EDA":
    st.header("Exploratory Data Analysis")
    st.markdown("""
•	The dataset cleaned_motor_vehicle_dataset.csv was loaded, inspected for structure, shape, and missing values to ensure data quality.\n
•	Date columns (Repair Date, Towing Date) were converted to proper datetime format for consistency and time-based analysis.\n
•	Descriptive statistics (mean, median, std, min, max) were computed for numeric features like Total Cost, Estimated Cost, Service Duration, and Mileage, while non-numeric fields were excluded.\n
•	Correlation analysis revealed strong positive links between Estimated Cost, Service Duration, and Total Cost — key drivers of repair expenses.\n
•	Inferential tests (ANOVA, T-test, Chi-Square) were applied to identify statistically significant relationships, such as higher costs for urgent services and longer durations for larger vehicles.\n
•	All summary tables and test results (including p-values and significance outcomes) were saved in the cleaned_data directory for reference and reporting.\n
•	The EDA provided comprehensive insights into cost behavior, service efficiency, and customer patterns — forming a solid foundation for predictive modeling and business decision-making.\n\n
""")

    desc = load_csv(os.path.join(EDA_DIR, "descriptive_statistics.csv"), "descriptive_statistics.csv")
    if desc is not None:
        st.markdown("\n**Descriptive Statistics**")
        st.dataframe(desc.round(3), width="stretch")

        corr_path = os.path.join(EDA_DIR, "numeric_correlation.csv")
        corr = load_csv(corr_path, "numeric_correlation.csv")

    if corr is not None and not corr.empty:
        st.markdown("**Numeric Correlation Matrix**")
        st.dataframe(corr.round(3), width="stretch")

    stats = load_csv(os.path.join(EDA_DIR, "statistical_test_results.csv"), "statistical_test_results.csv")
    if stats is not None:
        st.markdown("**Statistical Test Results**")
        st.dataframe(stats, width="stretch")

    #  Visualizations 
    st.subheader("EDA Visualizations")
    df = pd.read_csv("cleaned_motor_vehicle_dataset.csv")

    # Univariate — numeric
    fig, ax = plt.subplots()
    df["Total Cost"].hist(bins=30, ax=ax)
    ax.set_title("Distribution of Total Cost")
    ax.set_xlabel("Total Cost")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    df["Service Duration Hours"].hist(bins=30)
    plt.title("Service Duration Hours")
    plt.xlabel("Hours"); plt.ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig)

    # Univariate — categorical
    fig, ax = plt.subplots()
    df["Vehicle Type"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Vehicle Type Frequency")
    ax.set_xlabel("Vehicle Type"); ax.set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig)

    # Pie — Vehicle Type distribution
    fig, ax = plt.subplots()
    df["Vehicle Type"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90, ax=ax)
    ax.set_title("Vehicle Type Distribution (Pie Chart)")
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)

    # Bivariate
    fig, ax = plt.subplots()
    df.groupby("Vehicle Type")["Total Cost"].mean().plot(kind="bar", ax=ax)
    ax.set_title("Average Total Cost by Vehicle Type")
    ax.set_ylabel("Average Cost")
    plt.tight_layout()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    df.groupby("Urgency Level")["Total Cost"].mean().plot(kind="bar", ax=ax)
    ax.set_title("Average Total Cost by Urgency Level")
    ax.set_ylabel("Average Cost")
    plt.tight_layout()
    st.pyplot(fig)

    # Daily trends
    fig, ax = plt.subplots()
    if {"Repair Date", "Total Cost"}.issubset(df.columns):
        daily_cost = df.dropna(subset=["Repair Date"]).groupby("Repair Date")["Total Cost"].mean()
        daily_cost.plot(kind="line", marker="o", linewidth=2, ax=ax)
        ax.set_title("Average Total Cost Per Day")
        ax.set_xlabel("Date"); ax.set_ylabel("Average Total Cost")
        ax.grid(True); plt.tight_layout()
        st.pyplot(fig)

    # HEATMAP 
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) >= 2:
        corr_mat = df[num_cols].corr(numeric_only=True)
        os.makedirs("cleaned_data", exist_ok=True)
        corr_mat.to_csv(os.path.join("cleaned_data", "numeric_correlation.csv"), index=True)

        fig, ax = plt.subplots(figsize=(5, 5))
        cax = ax.matshow(corr_mat.values.astype(float), cmap="coolwarm")
        ax.set_title("Numeric Correlation Heatmap", pad=20)
        fig.colorbar(cax)
        plt.tight_layout()
        st.pyplot(fig)

# Regression Results

elif menu == "Regression Results":
    st.header(" Regression Results")
    st.markdown("""
•	The main goal was to predict the continuous numeric outcome — Total Cost of vehicle repair or service.\n
•	The dataset feature_engineered_dataset.csv was loaded and split into features (X) and the target variable (y = Total Cost).\n
•	Data was divided into training (80%) and testing (20%) sets for model training and evaluation.\n
•	Numeric and categorical columns were identified for proper preprocessing.\n
•	A Linear Regression pipeline was built with feature scaling and one-hot encoding to capture simple, linear relationships.\n
•	A Random Forest Regressor pipeline was also created with encoding (no scaling) to handle non-linear interactions.\n
•	Both models were trained and evaluated using MAE, MSE, RMSE, and R² metrics, then compared in a summary table.\n
•	The best model (based on the highest R²) was identified, providing insights into which factors most affect cost and supporting data-driven pricing and service optimization.\n
""")

    # Paths
    best_path  = os.path.join(REG_DIR, "best_regression_model.csv")
    cmp_path   = os.path.join(REG_DIR, "regression_model_comparison.csv")
    preds_path = os.path.join(REG_DIR, "regression_predictions.csv")

    # Helper to map variant column names
    def pick_col(df, candidates):
        cols_lower = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in cols_lower:
                return cols_lower[c.lower()]
        return None

    # Best model
    best_df = load_csv(best_path, "best_regression_model.csv")
    if best_df is not None and not best_df.empty:
        st.subheader(" Best Regression Model")
        st.dataframe(best_df, width="stretch")
        try:
            if "Model" in best_df.columns:
                st.success(f"Selected model: **{best_df.loc[0, 'Model']}**")
        except Exception:
            pass

    # Model comparison table
    cmp_df = load_csv(cmp_path, "regression_model_comparison.csv")
    if cmp_df is not None and not cmp_df.empty:
        st.subheader("Model Comparison")
        try:
            num_cols = cmp_df.select_dtypes(include="number").columns
            st.dataframe(
                cmp_df.assign(**{c: cmp_df[c].round(3) for c in num_cols}),
                width="stretch"
            )
            if "R2 Score" in cmp_df.columns:
                best_row = cmp_df.iloc[cmp_df["R2 Score"].idxmax()]
                st.info(f"Best by R²: **{best_row['Model']}** (R² = {best_row['R2 Score']:.3f})")
        except Exception:
            st.dataframe(cmp_df, width="stretch")

    # Predictions (Actual vs Predicted)
    preds_df = load_csv(preds_path, "regression_predictions.csv")
    if preds_df is not None and not preds_df.empty:
        st.subheader(" Actual vs Predicted ")
        st.dataframe(preds_df.head(200), width="stretch")

        actual_col = "Actual" if "Actual" in preds_df.columns else None
        candidate_preds = ["Random_Forest_Pred", "Linear_Regression_Pred"]
        pred_cols = [c for c in candidate_preds if c in preds_df.columns]

        if actual_col and len(pred_cols) > 0:
            x_all = pd.to_numeric(preds_df[actual_col], errors="coerce")

            fig, ax = plt.subplots(figsize=(7, 4.5))
            lo_vals, hi_vals = [], []
            markers = ["o", "x", "s", "D", "^"]

            for idx, col in enumerate(pred_cols):
                y = pd.to_numeric(preds_df[col], errors="coerce")
                mask = x_all.notna() & y.notna()
                x, y = x_all[mask], y[mask]
                if len(x) == 0:
                    continue
                ax.scatter(x, y, alpha=0.6, s=18, marker=markers[idx % len(markers)],
                        label=col.replace("_", " "))
                lo_vals.append(min(x.min(), y.min()))
                hi_vals.append(max(x.max(), y.max()))

            if lo_vals and hi_vals:
                lo, hi = min(lo_vals), max(hi_vals)
                ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2, label="Perfect Prediction")

            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted (Multiple Models)")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning(
                "Expected columns not found. Need 'Actual' and at least one of "
                "'Random_Forest_Pred' or 'Linear_Regression_Pred'."
            )

    st.caption(f" Loaded from: {REG_DIR}")

# Classification Results
elif menu == "Classification Results":
    st.header(" Classification Results (from CSV)")
    st.markdown("""
•	The goal was to classify whether an insurance claim was used or not (Insurance Claim Used), a categorical target variable.\n
•	The dataset was loaded, cleaned, and split into features (X) and target (y), then further divided into training (80%) and testing (20%) sets.\n
•	Numeric and categorical features were identified and preprocessed using scaling and one-hot encoding for model compatibility.\n
•	Two models were built and trained — Logistic Regression (for linear patterns) and CatBoost Classifier (for complex, non-linear relationships).\n
•	Both models were evaluated using accuracy, precision, recall, and F1-score, and the best model was selected based on performance.\n
•	Feature importance from CatBoost was analyzed to identify which factors most influence insurance claim usage, helping in decision-making and risk assessment.\n
""")

    #  Helper: safe confusion-matrix plotting (numeric-only)
    def plot_confusion_matrix(cm_df, title: str):
        try:
            cm_numeric = cm_df.apply(pd.to_numeric, errors="coerce").to_numpy().astype(float)
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            im = ax.imshow(cm_numeric, cmap="Blues")
            fig.colorbar(im, ax=ax)
            ax.set_title(title, pad=12)
            ax.set_xticks(range(cm_df.shape[1]))
            ax.set_yticks(range(cm_df.shape[0]))
            ax.set_xticklabels(list(cm_df.columns.astype(str)), rotation=45, ha="right")
            ax.set_yticklabels(list(cm_df.index.astype(str)))
            for i in range(cm_numeric.shape[0]):
                for j in range(cm_numeric.shape[1]):
                    ax.text(j, i, f"{cm_numeric[i, j]:.0f}", ha="center", va="center")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not plot confusion matrix: {e}")
            st.dataframe(cm_df)

    # Paths
    best_cls_path   = os.path.join(CLS_DIR, "best_classification_model.csv")
    cmp_cls_path    = os.path.join(CLS_DIR, "classification_model_comparison.csv")
    preds_cls_path  = os.path.join(CLS_DIR, "classification_predictions.csv")
    lr_cm_path      = os.path.join(CLS_DIR, "logistic_regression_confusion_matrix.csv")
    cb_cm_path      = os.path.join(CLS_DIR, "catboost_confusion_matrix.csv")
    cb_fi_path      = os.path.join(CLS_DIR, "catboost_feature_importance.csv")

    # Best model 
    best_cls = load_csv(best_cls_path, "best_classification_model.csv")
    if best_cls is not None and not best_cls.empty:
        st.subheader(" Best Classification Model")
        st.dataframe(best_cls, width="stretch")
        try:
            if "Model" in best_cls.columns:
                st.success(f"Selected model: **{best_cls.loc[0, 'Model']}**")
        except Exception:
            pass

    # Model comparison table 
    cls_cmp = load_csv(cmp_cls_path, "classification_model_comparison.csv")
    if cls_cmp is not None and not cls_cmp.empty:
        st.subheader("Model Comparison")
        try:
            num_cols = cls_cmp.select_dtypes(include="number").columns
            st.dataframe(
                cls_cmp.assign(**{c: cls_cmp[c].round(3) for c in num_cols}),
                width="stretch"
            )
            if "Accuracy" in cls_cmp.columns:
                best_row = cls_cmp.iloc[cls_cmp["Accuracy"].idxmax()]
                st.info(f"Best by Accuracy: **{best_row['Model']}** (Accuracy = {best_row['Accuracy']:.3f})")
            elif "F1" in cls_cmp.columns:
                best_row = cls_cmp.iloc[cls_cmp["F1"].idxmax()]
                st.info(f"Best by F1: **{best_row['Model']}** (F1 = {best_row['F1']:.3f})")
        except Exception:
            st.dataframe(cls_cmp, width="stretch")

    # Predictions sample 
    preds_cls = load_csv(preds_cls_path, "classification_predictions.csv")
    if preds_cls is not None and not preds_cls.empty:
        st.subheader("Predictions (sample)")
        st.dataframe(preds_cls.head(200), width="stretch")

    # Confusion matrices 
    st.subheader(" Confusion Matrices")
    lr_cm = load_csv(lr_cm_path, "logistic_regression_confusion_matrix.csv")
    if lr_cm is not None and not lr_cm.empty:
        plot_confusion_matrix(lr_cm, "Logistic Regression — Confusion Matrix")

    cb_cm = load_csv(cb_cm_path, "catboost_confusion_matrix.csv")
    if cb_cm is not None and not cb_cm.empty:
        plot_confusion_matrix(cb_cm, "CatBoost — Confusion Matrix")

    # CatBoost feature importance
    cb_fi = load_csv(cb_fi_path, "catboost_feature_importance.csv")
    if cb_fi is not None and not cb_fi.empty:
        st.subheader(" CatBoost — Feature Importance")
        try:
            if "Importance" in cb_fi.columns:
                cb_fi = cb_fi.sort_values("Importance", ascending=False)
            st.dataframe(cb_fi.head(15), width="stretch")

            if {"Feature", "Importance"}.issubset(cb_fi.columns):
                plot_df = cb_fi.head(15).copy()
                plot_df["Importance"] = pd.to_numeric(plot_df["Importance"], errors="coerce")
                fig, ax = plt.subplots(figsize=(6, 4.5))
                ax.barh(plot_df["Feature"].astype(str), plot_df["Importance"].astype(float))
                ax.set_title("Top Feature Importances")
                ax.set_xlabel("Importance")
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not render feature importance chart: {e}")
            st.dataframe(cb_fi, width="stretch")

    st.caption(f"Loaded from: {CLS_DIR}")

# Clustering Results
elif menu == "Clustering Results":
    st.header(" Clustering Results ")
    st.markdown("""
•	The goal was to group similar vehicle service records based on numeric features like Total Cost, Estimated Cost, Service Duration, and Mileage — to uncover hidden service patterns.\n
•	The dataset was loaded, relevant numeric features were selected, and missing values were handled using median imputation, followed by standard scaling for normalization.\n
•	Two clustering models — K-Means and Gaussian Mixture Model (GMM) — were trained to form clusters of similar service records.\n
•	Each model’s quality was evaluated using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score to identify the better-performing model.\n
•	Cluster profiles were analyzed to interpret characteristics like high/low cost, mileage, and repair duration, generating insightful cluster labels (e.g., “High-Cost Long Services”).\n
•	The best clustering model and its insights were exported, providing a data-driven segmentation useful for identifying service patterns, optimizing operations, and targeting specific customer groups.\n
""")

    # Helper: try multiple candidate filenames 
    def load_first_existing(base_dir, candidates, friendly_name):
        for fname in candidates:
            path = os.path.join(base_dir, fname)
            if os.path.exists(path):
                return load_csv(path, fname)
        st.warning(f" Missing: {friendly_name} — tried {', '.join(candidates)} in `{base_dir}`.")
        return None

    # Helper to pick a cluster label column
    def pick_cluster_col(df):
        return next((c for c in df.columns if c.lower() in
                    ["cluster", "label", "cluster_label", "cluster id", "cluster_id", "segment"]), None)

    # Simple heatmap for cluster means (numeric-only, defensive against dtype issues)
    def plot_means_heatmap(means_df, title):
        try:
            dfm = means_df.copy()

            # If there is an obvious ID/Cluster column, make it the index
            id_like = next((c for c in dfm.columns if c.lower() in
                            ["cluster", "cluster_id", "cluster id", "label", "segment", "cluster_label"]), None)
            if id_like is not None:
                dfm = dfm.set_index(id_like, drop=True)

            # Keep only numeric columns; coerce where possible
            for c in dfm.columns:
                if not pd.api.types.is_numeric_dtype(dfm[c]):
                    dfm[c] = pd.to_numeric(dfm[c], errors="coerce")

            num_df = dfm.select_dtypes(include=["number"])
            if num_df.empty:
                st.info(f"No numeric columns to plot in '{title}'.")
                return

            fig, ax = plt.subplots(figsize=(6, 4.5))
            im = ax.imshow(num_df.to_numpy().astype(float), aspect="auto", cmap="Blues")
            fig.colorbar(im, ax=ax)
            ax.set_title(title, pad=12)
            ax.set_xticks(range(num_df.shape[1]))
            ax.set_xticklabels(list(num_df.columns.astype(str)), rotation=45, ha="right")
            ax.set_yticks(range(num_df.shape[0]))
            ax.set_yticklabels(list(num_df.index.astype(str)))
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not render heatmap for '{title}': {e}")

    # Files 
    cmp_df = load_first_existing(
        CLU_DIR,
        ["cluster_model_comparison.csv", "clustering_model_comparison.csv"],
        "cluster model comparison"
    )
    km_means = load_first_existing(
        CLU_DIR,
        ["kmeans_cluster_means.csv", "kmeans_means.csv"],
        "KMeans cluster means"
    )
    gmm_means = load_first_existing(
        CLU_DIR,
        ["gmm_cluster_means.csv", "gmm_means.csv"],
        "GMM cluster means"
    )
    final_clustered = load_first_existing(
        CLU_DIR,
        ["final_clustered_dataset.csv"],
        "final clustered dataset"
    )

    # Model comparison table
    if cmp_df is not None and not cmp_df.empty:
        st.subheader(" Model Comparison")
        try:
            num_cols = cmp_df.select_dtypes(include="number").columns
            st.dataframe(cmp_df.assign(**{c: cmp_df[c].round(3) for c in num_cols}), width="stretch")
        except Exception:
            st.dataframe(cmp_df, width="stretch")

    # Cluster means (tables + heatmaps if numeric)
    cols = st.columns(2)
    with cols[0]:
        if km_means is not None and not km_means.empty:
            st.subheader("KMeans — Cluster Means")
            try:
                num_cols = km_means.select_dtypes(include="number").columns
                st.dataframe(km_means.assign(**{c: km_means[c].round(2) for c in num_cols}), width="stretch")
            except Exception:
                st.dataframe(km_means, width="stretch")
            plot_means_heatmap(km_means, "KMeans Means (Heatmap)")
    with cols[1]:
        if gmm_means is not None and not gmm_means.empty:
            st.subheader("GMM — Cluster Means")
            try:
                num_cols = gmm_means.select_dtypes(include="number").columns
                st.dataframe(gmm_means.assign(**{c: gmm_means[c].round(2) for c in num_cols}), width="stretch")
            except Exception:
                st.dataframe(gmm_means, width="stretch")
            plot_means_heatmap(gmm_means, "GMM Means (Heatmap)")

    # Final clustered dataset — sample, counts + minimal visuals
    if final_clustered is not None and not final_clustered.empty:
        st.subheader(" Final Clustered Dataset (sample)")
        st.dataframe(final_clustered.head(200), width="stretch")

        # Cluster counts (table + bar)
        cluster_col = pick_cluster_col(final_clustered)
        if cluster_col:
            st.subheader(" Cluster Counts")
            counts = final_clustered[cluster_col].value_counts().sort_index()
            st.dataframe(counts.rename("Count").to_frame(), width="stretch")

            # Bar chart
            try:
                fig, ax = plt.subplots(figsize=(5.5, 3.5))
                ax.bar(counts.index.astype(str), counts.values)
                ax.set_xlabel(cluster_col)
                ax.set_ylabel("Count")
                ax.set_title("Cluster Size Distribution")
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not plot cluster counts: {e}")

        # 2D scatter using the first two numeric features (if present), colored by Cluster
        if cluster_col:
            num_cols = final_clustered.select_dtypes(include="number").columns.tolist()
            # remove cluster col if numeric
            num_cols_wo_cluster = [c for c in num_cols if c != cluster_col]
            if len(num_cols_wo_cluster) >= 2:
                xcol, ycol = num_cols_wo_cluster[0], num_cols_wo_cluster[1]
                st.subheader(f"🗺️ 2D Scatter: {xcol} vs {ycol} (by {cluster_col})")
                try:
                    df_sc = final_clustered[[xcol, ycol, cluster_col]].copy()
                    df_sc[xcol] = pd.to_numeric(df_sc[xcol], errors="coerce")
                    df_sc[ycol] = pd.to_numeric(df_sc[ycol], errors="coerce")
                    df_sc = df_sc.dropna(subset=[xcol, ycol, cluster_col])

                    fig, ax = plt.subplots(figsize=(6.5, 4.5))
                    for cl, group in df_sc.groupby(cluster_col):
                        ax.scatter(group[xcol], group[ycol], alpha=0.6, label=str(cl))
                    ax.set_xlabel(xcol)
                    ax.set_ylabel(ycol)
                    ax.set_title("Clusters in 2D")
                    ax.legend(title=str(cluster_col), bbox_to_anchor=(1.02, 1), loc="upper left")
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not render 2D scatter: {e}")

    st.caption(f" Loaded from: {CLU_DIR}")

#  Final Insights
elif menu == "Final Insights":
    st.header("Final Business Insights & Recommendations")
    st.write("Below is a simple summary of what the data and models are telling us, and what to do with it.")

    # EDA Insights
    st.subheader("1) What we saw in the data (EDA)")
    st.markdown("""
- *Total Cost* is skewed: most jobs are inexpensive; a small number are very expensive.
- *Service Duration Hours* shows a long tail: a few jobs take much longer than the rest.
- *Vehicle Type*: cars and bikes make up most visits; trucks/buses are fewer but usually cost more.
- *Towing* jobs tend to end up expensive.
- *Mileage at Service* goes hand-in-hand with higher cost—older, high-mileage vehicles need more work.
- *Cost Difference (Total − Estimated)* is larger on high-cost jobs → we tend to *under-estimate* complex work.
- Correlations are consistent with common sense: Total Cost rises with *Estimated Cost, **Duration, and **Mileage*.
- Daily trends show predictable *busy days* that are useful for staffing and parts planning.
    """)

    #  Regression
    st.subheader("2) What drives cost (Regression + EDA)")
    st.markdown("""
- *Estimated Cost* explains most of the final bill—our estimate is the best signal we have.
- *Duration* and *Mileage* also push the cost up.
- Takeaway: getting the estimate right matters. So does spotting long jobs and high-mileage vehicles early.
    """)

    #  Clusters
    st.subheader("3) The two main service groups (Clustering)")
    st.markdown("""
*Cluster 0 – Routine, low-cost work*
- Short duration, lower mileage, estimates close to the final bill.
- Good candidates for fast-track lanes and standard packages.

*Cluster 1 – Complex, high-cost work*
- Long duration, high mileage, estimates often too low.
- Needs senior technicians, better parts readiness, and clearer customer communication up front.
    """)

    # Classification
    st.subheader("4) Insurance (Classification)")
    st.markdown("""
- Using insurance doesn’t meaningfully change the final cost.
- Offering claim assistance is useful and doesn’t hurt margins.
    """)

    # Technicians & Workflow
    st.subheader("5) Technicians & workflow")
    st.markdown("""
- Top-rated techs tend to handle complex jobs faster. Put them on *Cluster 1* work.
- Use busy-day patterns to set schedules and keep parts on hand.
    """)
    
    # Model Performance
    st.subheader("6) Model Performance")
    st.markdown("""
- *Random Forest outperformed Linear Regression for cost prediction (better R² and lower RMSE).
- *CatBoost outperformed Logistic Regression for classifying insurance claim usage.
- *K-Means achieved better performance (higher Silhouette, lower DBI) — producing clearer, well-separated service clusters.
    """)

    # What to do next
    st.subheader("7) What to do next")
    st.markdown("""
- *Tighten estimates* for long jobs and high-mileage vehicles; add a diagnostic step if needed.
- *Segment the workflow*: fast-track for routine jobs, expert lane for complex jobs.
- *Plan staffing and inventory* around the busy days and the high-cost cluster.
- Don’t force ML on *Urgency*—use rules and customer input instead.
- Keep the *cost prediction model* in the loop for quotes, scheduling, and parts planning.
    """)