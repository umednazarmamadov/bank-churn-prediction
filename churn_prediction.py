# -*- coding: utf-8 -*-
"""
============================================================
Project: Bank Customer Churn Prediction

Dataset: Churn_Modelling.csv (Kaggle)
https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling

Goal: Build a model that predicts whether a bank customer will
leave the bank (Exited = 1) or stay (Exited = 0).

Steps:
    1. Load data
    2. Exploratory Data Analysis (EDA)
    3. Data preprocessing
    4. Train models (Logistic Regression, Random Forest, XGBoost)
    5. Compare models by metrics
    6. Feature importance analysis
    7. Conclusions

Run:
    python churn_prediction.py
============================================================
"""

import os
import warnings

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # save plots to files without opening windows
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Global settings
# ------------------------------------------------------------
RANDOM_STATE = 42
DATA_PATH = "data/Churn_Modelling.csv"
PLOTS_DIR = "plots"
RESULTS_DIR = "results"

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_plot(filename: str) -> None:
    """Save the current matplotlib figure to the plots folder and close it."""
    path = os.path.join(PLOTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"   [plot saved] {path}")


# ============================================================
# 1. LOAD DATA
# ============================================================
def load_data(path: str) -> pd.DataFrame:
    """Load the dataset and print basic information about it."""
    print("=" * 60)
    print("1. LOAD DATA")
    print("=" * 60)

    df = pd.read_csv(path)

    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    print("First 5 rows:")
    print(df.head().to_string(), "\n")

    info = pd.DataFrame({
        "Type": df.dtypes.astype(str),
        "Nulls": df.isna().sum(),
        "Unique": df.nunique(),
    })
    print("Data types, nulls and unique values:")
    print(info.to_string(), "\n")

    print("Descriptive statistics (numeric features):")
    print(df.describe().round(2).to_string(), "\n")

    return df


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
def run_eda(df: pd.DataFrame) -> None:
    """Plot class balance, feature distributions, churn rates by category,
    and the correlation matrix."""
    print("=" * 60)
    print("2. EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    # --- 2.1. Class balance --------------------------------------------
    counts = df["Exited"].value_counts().sort_index()
    churn_rate = df["Exited"].mean() * 100
    print(f"Stayed  (0): {counts[0]} customers")
    print(f"Churned (1): {counts[1]} customers")
    print(f"Churn rate: {churn_rate:.1f}% -> class imbalance present\n")

    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Stayed (0)", "Churned (1)"], counts.values,
                   color=["#2ecc71", "#e74c3c"])
    for bar, cnt in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{cnt} ({cnt / len(df) * 100:.1f}%)",
                 ha="center", va="bottom")
    plt.title("Class Balance: Stayed vs Churned")
    plt.ylabel("Number of customers")
    save_plot("01_class_balance.png")

    # --- 2.2. Numeric feature distributions ----------------------------
    numeric_cols = ["CreditScore", "Age", "Tenure",
                    "Balance", "NumOfProducts", "EstimatedSalary"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flatten(), numeric_cols):
        sns.histplot(df[col], bins=30, ax=ax, color="#3498db")
        ax.set_title(f"Distribution of {col}")
    save_plot("02_numeric_distributions.png")

    # --- 2.3. Churn rate by categorical feature ------------------------
    cat_cols = ["Geography", "Gender", "HasCrCard", "IsActiveMember"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flatten(), cat_cols):
        rate = df.groupby(col)["Exited"].mean().mul(100)
        rate.plot(kind="bar", ax=ax, color="#e67e22", rot=0)
        ax.set_title(f"Churn rate by {col}")
        ax.set_ylabel("Churn rate, %")
        for i, v in enumerate(rate.values):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
    save_plot("03_churn_rate_by_category.png")

    print("Churn rate by country, %:")
    print((df.groupby("Geography")["Exited"].mean() * 100).round(1).to_string(), "\n")
    print("Churn rate by number of products, %:")
    print((df.groupby("NumOfProducts")["Exited"].mean() * 100).round(1).to_string(), "\n")

    # --- 2.4. Age and balance vs churn ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(x="Exited", y="Age", data=df, ax=axes[0],
                palette=["#2ecc71", "#e74c3c"])
    axes[0].set_title("Age vs Churn")
    sns.boxplot(x="Exited", y="Balance", data=df, ax=axes[1],
                palette=["#2ecc71", "#e74c3c"])
    axes[1].set_title("Balance vs Churn")
    save_plot("04_age_balance_vs_churn.png")

    # --- 2.5. Correlation matrix ---------------------------------------
    corr_cols = numeric_cols + ["HasCrCard", "IsActiveMember", "Exited"]
    corr = df[corr_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title("Correlation Matrix")
    save_plot("05_correlation_matrix.png")

    print("Feature correlations with target variable 'Exited':")
    print(corr["Exited"].drop("Exited").sort_values(ascending=False)
          .round(3).to_string(), "\n")


# ============================================================
# 3. DATA PREPROCESSING
# ============================================================
def preprocess(df: pd.DataFrame):
    """Drop irrelevant columns, encode categorical features,
    split into train/test, and scale numeric features."""
    print("=" * 60)
    print("3. DATA PREPROCESSING")
    print("=" * 60)

    data = df.copy()

    # Drop columns with no predictive value:
    # RowNumber, CustomerId, Surname are identifiers, not features.
    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    data = data.drop(columns=drop_cols)
    print(f"Dropped columns: {drop_cols}")

    # Encode categorical features:
    # Gender is binary -> map to 0/1
    data["Gender"] = data["Gender"].map({"Female": 0, "Male": 1})

    # Geography has 3 categories -> one-hot encoding.
    # drop_first=True drops France to avoid multicollinearity.
    data = pd.get_dummies(data, columns=["Geography"], drop_first=True, dtype=int)
    print(f"Features after encoding: {list(data.columns)}\n")

    X = data.drop(columns=["Exited"])
    y = data["Exited"]

    # Stratified 80/20 split — preserves class ratio in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples")

    # Scale numeric features using StandardScaler.
    # Fit ONLY on training data to prevent data leakage.
    # Tree-based models don't require scaling, but it doesn't hurt them.
    numeric_cols = ["CreditScore", "Age", "Tenure",
                    "Balance", "NumOfProducts", "EstimatedSalary"]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    print("Numeric features scaled (StandardScaler)\n")

    return X_train_scaled, X_test_scaled, y_train, y_test


# ============================================================
# 4. TRAIN MODELS
# ============================================================
def train_models(X_train, y_train) -> dict:
    """Train three models: Logistic Regression, Random Forest, XGBoost.

    Classes are imbalanced (~20% churn), so we use:
      - class_weight='balanced' for LogisticRegression and RandomForest;
      - scale_pos_weight for XGBoost.
    This penalizes errors on the minority class (churned customers),
    improving recall — catching potential churners matters more than
    maximizing raw accuracy.
    """
    print("=" * 60)
    print("4. TRAIN MODELS")
    print("=" * 60)

    # Class ratio for XGBoost: (count of 0s) / (count of 1s)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight for XGBoost: {scale_pos_weight:.2f}\n")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,             # limit depth to reduce overfitting
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,            # fraction of samples per tree
            colsample_bytree=0.9,     # fraction of features per tree
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Trained: {name}")
    print()

    return models


# ============================================================
# 5. EVALUATE MODELS
# ============================================================
def evaluate_models(models: dict, X_test, y_test):
    """Compute metrics for each model and plot:
    metric comparison bar chart, ROC curves, and confusion matrices."""
    print("=" * 60)
    print("5. EVALUATE MODELS")
    print("=" * 60)

    rows = []
    predictions = {}
    roc_data = {}
    cm_data = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        })
        predictions[name] = y_pred
        roc_data[name] = roc_curve(y_test, y_proba)
        cm_data[name] = confusion_matrix(y_test, y_pred)

    results = pd.DataFrame(rows).set_index("Model").round(3)
    print("Metrics summary (test set):")
    print(results.to_string(), "\n")

    results.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))
    print(f"   [table saved] {RESULTS_DIR}/model_comparison.csv")

    # --- Metric comparison bar chart ------------------------------------
    ax = results.plot(kind="bar", figsize=(11, 6), rot=0, width=0.8)
    ax.set_title("Model Comparison by Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    save_plot("06_metrics_comparison.png")

    # --- ROC curves ----------------------------------------------------
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, _) in roc_data.items():
        auc = results.loc[name, "ROC-AUC"]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    save_plot("07_roc_curves.png")

    # --- Confusion matrices --------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, cm) in zip(axes, cm_data.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Stayed", "Churned"],
                    yticklabels=["Stayed", "Churned"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    save_plot("08_confusion_matrices.png")

    return results, predictions


# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================
def analyze_feature_importance(models: dict, feature_names) -> pd.Series:
    """Plot feature importance for Random Forest and XGBoost,
    and print Logistic Regression coefficients."""
    print("=" * 60)
    print("6. FEATURE IMPORTANCE")
    print("=" * 60)

    rf_imp = pd.Series(models["Random Forest"].feature_importances_,
                       index=feature_names).sort_values()
    xgb_imp = pd.Series(models["XGBoost"].feature_importances_,
                        index=feature_names).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rf_imp.plot(kind="barh", ax=axes[0], color="#27ae60")
    axes[0].set_title("Feature Importance — Random Forest")
    xgb_imp.plot(kind="barh", ax=axes[1], color="#8e44ad")
    axes[1].set_title("Feature Importance — XGBoost")
    save_plot("09_feature_importance.png")

    print("Top-5 features (Random Forest):")
    print(rf_imp.sort_values(ascending=False).head(5).round(3).to_string(), "\n")
    print("Top-5 features (XGBoost):")
    print(xgb_imp.sort_values(ascending=False).head(5).round(3).to_string(), "\n")

    # Logistic Regression coefficients.
    # Data was scaled, so coefficients are comparable in magnitude.
    # Sign indicates direction: "+" increases churn probability, "-" decreases it.
    lr = models["Logistic Regression"]
    coef = pd.Series(lr.coef_[0], index=feature_names)
    coef = coef.reindex(coef.abs().sort_values(ascending=False).index)
    print("Logistic Regression coefficients (sign = direction of effect):")
    print(coef.round(3).to_string(), "\n")

    return rf_imp.sort_values(ascending=False)


# ============================================================
# 7. CONCLUSIONS
# ============================================================
def print_conclusions(results: pd.DataFrame, top_features: pd.Series,
                      predictions: dict, y_test) -> None:
    """Print final conclusions: best model and key churn drivers."""
    print("=" * 60)
    print("7. CONCLUSIONS")
    print("=" * 60)

    best_f1 = results["F1"].idxmax()
    best_auc = results["ROC-AUC"].idxmax()

    print(f"Best model by F1:      {best_f1} "
          f"(F1 = {results.loc[best_f1, 'F1']:.3f})")
    print(f"Best model by ROC-AUC: {best_auc} "
          f"(AUC = {results.loc[best_auc, 'ROC-AUC']:.3f})\n")

    print(f"Detailed report for '{best_f1}':")
    print(classification_report(y_test, predictions[best_f1],
                                target_names=["Stayed", "Churned"]))

    top5 = ", ".join(top_features.head(5).index)
    print(f"Top churn drivers: {top5}\n")

    print("Key findings:")
    print(" • Age — strongest factor: churned customers average ~45 years old")
    print("   vs ~37 for those who stayed.")
    print(" • NumOfProducts — customers with 2 products churn least (~8%);")
    print("   customers with 3-4 products churn almost always (83-100%).")
    print(" • IsActiveMember — inactive customers churn nearly twice as often.")
    print(" • Geography — churn in Germany (~32%) is 2x higher than in")
    print("   France or Spain (~16%).")
    print(" • Balance — churned customers hold higher balances on average;")
    print("   the bank is losing its wealthiest clients.\n")

    print("Note on accuracy: with an 80/20 class split, a dummy model that")
    print("always predicts 'stays' already gets ~80% accuracy. That is why")
    print("model selection here relies on F1, Recall and ROC-AUC.\n")

    print("Business recommendation: retention campaigns should target older,")
    print("inactive, high-balance customers (especially in Germany), and the")
    print("bank should investigate why customers with 3-4 products are dissatisfied.")


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    df = load_data(DATA_PATH)
    run_eda(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    models = train_models(X_train, y_train)
    results, predictions = evaluate_models(models, X_test, y_test)
    top_features = analyze_feature_importance(models, X_train.columns)
    print_conclusions(results, top_features, predictions, y_test)

    print("\nDone! Plots saved to 'plots/', metrics table saved to 'results/model_comparison.csv'.")


if __name__ == "__main__":
    main()
    