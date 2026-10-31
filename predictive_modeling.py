import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    mean_squared_error,
    r2_score,
)

# Ensure output directory exists
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def run_classification_pipeline():
    """Trains Decision Tree, Random Forest, and thresholded Linear Regression for classification."""
    print("=" * 60)
    print("1. SUPERVISED CLASSIFICATION & MODEL EVALUATION")
    print("=" * 60)

    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    models = {
        "Linear Regression (Threshold 0.5)": LinearRegression(),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = []
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

    for idx, (name, model) in enumerate(models.items()):
        # Fit model
        model.fit(X_train_scaled, y_train)

        # Predict probabilities/scores and class labels
        if name.startswith("Linear Regression"):
            y_scores = np.clip(model.predict(X_test_scaled), 0, 1)
            y_pred = (y_scores >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_test_scaled)
            y_scores = model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_scores)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
        })

        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=data.target_names,
            yticklabels=data.target_names,
            ax=axes_cm[idx],
        )
        axes_cm[idx].set_title(f"{name}\nAccuracy: {acc:.2%}")
        axes_cm[idx].set_xlabel("Predicted Label")
        axes_cm[idx].set_ylabel("True Label")

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    # Save ROC Curves
    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess")
    ax_roc.set_xlabel("False Positive Rate (FPR)")
    ax_roc.set_ylabel("True Positive Rate (TPR)")
    ax_roc.set_title("ROC Curves Comparison")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=300)
    plt.close(fig_roc)

    # Save Confusion Matrices
    fig_cm.suptitle("Model Confusion Matrices", fontsize=16, y=1.02)
    fig_cm.tight_layout()
    fig_cm.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_cm)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "classification_metrics.csv"), index=False)
    print(results_df.to_string(index=False))
    print(f"\nSaved 'confusion_matrices.png', 'roc_curves.png', and 'classification_metrics.csv' to ./{OUTPUT_DIR}/")


def run_regression_pipeline():
    """Demonstrates continuous target prediction using Linear Regression, Decision Tree, and Random Forest."""
    print("\n" + "=" * 60)
    print("2. SUPERVISED REGRESSION EVALUATION")
    print("=" * 60)

    # Generate synthetic continuous dataset for reproducible execution
    np.random.seed(42)
    X = np.random.rand(1000, 4) * 10
    y = 3.5 * X[:, 0] - 2.0 * X[:, 1] + 1.5 * (X[:, 2] ** 2) + np.random.normal(0, 2, 1000)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    reg_results = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, model) in enumerate(reg_models.items()):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        reg_results.append({"Model": name, "RMSE": round(rmse, 4), "R2 Score": round(r2, 4)})

        axes[idx].scatter(y_test, y_pred, alpha=0.5, color="teal")
        axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
        axes[idx].set_title(f"{name}\nR² = {r2:.3f} | RMSE = {rmse:.2f}")
        axes[idx].set_xlabel("Actual Values")
        axes[idx].set_ylabel("Predicted Values")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "regression_predictions.png"), dpi=300)
    plt.close(fig)

    reg_df = pd.DataFrame(reg_results)
    reg_df.to_csv(os.path.join(OUTPUT_DIR, "regression_metrics.csv"), index=False)
    print(reg_df.to_string(index=False))
    print(f"\nSaved 'regression_predictions.png' and 'regression_metrics.csv' to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    run_classification_pipeline()
    run_regression_pipeline()
