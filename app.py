import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

st.set_page_config(page_title="Predictive Modeling Dashboard", layout="wide")

st.title("Predictive Modeling & Model Evaluation Dashboard")
st.markdown(
    "Train and compare **Linear Regression**, **Decision Trees**, and **Random Forest** models "
    "with real-time **Accuracy**, **Confusion Matrices**, and **ROC Curves**."
)

# Sidebar Configuration
st.sidebar.header("Model & Data Settings")
test_size = st.sidebar.slider("Test Split Ratio", 0.1, 0.4, 0.2, 0.05)
max_depth = st.sidebar.slider("Decision Tree Max Depth", 2, 15, 5)
n_estimators = st.sidebar.slider("Random Forest Estimators", 10, 300, 100, 10)

# Load and split data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression (Thresholded)": LinearRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=max_depth, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=n_estimators, random_state=42),
}

metrics_list = []
roc_data = {}
cm_data = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    if "Linear Regression" in name:
        y_scores = np.clip(model.predict(X_test_scaled), 0, 1)
        y_pred = (y_scores >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test_scaled)
        y_scores = model.predict_proba(X_test_scaled)[:, 1]

    metrics_list.append({
        "Algorithm": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_scores),
    })
    cm_data[name] = confusion_matrix(y_test, y_pred)
    roc_data[name] = roc_curve(y_test, y_scores)

# Display Performance Table
st.subheader("1. Model Performance Metrics")
metrics_df = pd.DataFrame(metrics_list)
st.dataframe(metrics_df.style.format({
    "Accuracy": "{:.2%}",
    "Precision": "{:.2%}",
    "Recall": "{:.2%}",
    "F1-Score": "{:.2%}",
    "ROC-AUC": "{:.3f}",
}), use_container_width=True)

# Visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("2. Confusion Matrix")
    selected_model = st.selectbox("Select Model for Confusion Matrix", list(models.keys()), index=2)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_data[selected_model],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=data.target_names,
        yticklabels=data.target_names,
        ax=ax_cm,
    )
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title(f"{selected_model} Confusion Matrix")
    st.pyplot(fig_cm)

with col2:
    st.subheader("3. ROC Curves Comparison")
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    for name, (fpr, tpr, _) in roc_data.items():
        auc_val = metrics_df.loc[metrics_df["Algorithm"] == name, "ROC-AUC"].values[0]
        ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_val:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", label="Random Guess")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Receiver Operating Characteristic (ROC)")
    ax_roc.legend(loc="lower right", fontsize=8)
    st.pyplot(fig_roc)
