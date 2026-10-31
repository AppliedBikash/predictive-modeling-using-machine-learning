# Predictive Modeling Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Overview
This repository implements an end-to-end supervised machine learning pipeline to train, test, and evaluate predictive models. It compares **Linear Regression**, **Decision Trees**, and **Random Forests** across both classification and regression tasks, evaluating performance via accuracy metrics, confusion matrices, and Receiver Operating Characteristic (ROC) curves.

## Objectives & Learning Outcomes
- Apply supervised learning algorithms (**Linear Regression**, **Decision Trees**, and **Random Forest**).
- Partition datasets into training and testing sets with standardized feature scaling.
- Evaluate classification models using **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**, and regression models using **R²** and **RMSE**.
- Visualize model performance using **Confusion Matrices**, **ROC Curves**, and **Actual vs. Predicted Plots**.

---

## Repository Structure
```text
├── notebooks/
│   └── predictive_modeling.ipynb    # Interactive Jupyter Notebook walkthrough
├── outputs/
│   ├── classification_metrics.csv   # Exported classification evaluation metrics
│   ├── confusion_matrices.png       # Side-by-side Confusion Matrices
│   ├── regression_metrics.csv       # Exported regression evaluation metrics
│   ├── regression_predictions.png   # Actual vs. Predicted regression plots
│   └── roc_curves.png               # Multi-model ROC Curve comparison
├── src/
│   └── predictive_modeling.py       # Automated training, testing & plotting script
├── app.py                           # Interactive Streamlit web dashboard
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
├── LICENSE                          # MIT License
└── README.md                        # Project documentation
```

---

## Key Results

### 1. Classification Performance Benchmark
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Threshold 0.5)** | 95.61% | 94.67% | 98.61% | 96.60% | 0.992 |
| **Decision Tree Classifier** | 92.11% | 95.65% | 91.67% | 93.62% | 0.916 |
| **Random Forest Classifier** | 95.61% | 95.89% | 97.22% | 96.55% | 0.994 |

#### Confusion Matrices
![Confusion Matrices](outputs/confusion_matrices.png)

#### ROC Curves Comparison
![ROC Curves](outputs/roc_curves.png)

---

### 2. Continuous Regression Benchmark
| Model | RMSE | R² Score |
| :--- | :---: | :---: |
| **Linear Regression** | 11.46 | 0.939 |
| **Decision Tree Regressor** | 7.39 | 0.974 |
| **Random Forest Regressor** | 3.70 | 0.994 |

#### Actual vs. Predicted Values
![Regression Predictions](outputs/regression_predictions.png)

---

## Installation & Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/predictive-modeling-ml.git
   cd predictive-modeling-ml
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training and evaluation pipeline:**
   ```bash
   python src/predictive_modeling.py
   ```

4. **Launch the interactive Streamlit web application:**
   ```bash
   streamlit run app.py
   ```
