Credit Card Fraud Detection Project
🚀 Project Overview

This project focuses on building a machine learning-based system to detect fraudulent credit card transactions. The model learns from historical transaction data to distinguish between legitimate and fraudulent activity — helping reduce financial loss due to fraud.

Key goals:

Understand the dataset and its characteristics

Handle class imbalance issues

Train and evaluate robust ML models

Compare performance using metrics like recall and F1-score

📌 Problem Statement

Credit card fraud happens when unauthorized transactions are made using someone else’s card credentials. Because fraud instances are rare (often <0.5% of total transactions), conventional accuracy metrics are misleading. The focus must be on identifying as many fraud cases as possible while minimizing false positives.

📁 Dataset

We use the popular Credit Card Fraud Detection dataset from Kaggle, which contains transactions made by European cardholders with anonymized feature columns (V1 … V28) — output labeled as:

Class = 0: legitimate

Class = 1: fraud

Total transactions: ~284,807
Fraud cases: ~492 (≈ 0.17% — highly imbalanced)

🧠 Methodology
🛠️ 1. Data Preprocessing

Load the dataset

Check for missing values & duplicates

Normalize/scale Amount and Time

Split into train and test sets

🧠 2. Exploratory Data Analysis

Visualize class imbalance

Correlation analysis

Distribution plots

🧠 3. Handling Imbalance

Because fraud cases are rare, apply techniques like:

SMOTE oversampling

Random undersampling

Class weights in models

🤖 4. Machine Learning Models

Train and compare:

Logistic Regression

Random Forest

XGBoost

Support Vector Machine

Neural Networks

Evaluate using metrics:
✅ Recall (important for fraud detection)
✅ Precision
✅ F1-score
✅ ROC-AUC

Focus is on recall — detecting fraud is more important than overall accuracy.

🏁 Getting Started
💻 Requirements

Install dependencies:

pip install -r requirements.txt

Typical packages:

pandas, numpy

scikit-learn, imblearn (for SMOTE)

matplotlib, seaborn

🧪 Usage

Clone the repository:

git clone https://github.com/your-username/credit-card-fraud-detection.git

Navigate into the folder:

cd credit-card-fraud-detection

Run the Jupyter notebook or Python script:

jupyter notebook fraud_detection.ipynb
📊 Results & Evaluation

Include metrics and visualizations:

Confusion matrix

Precision-Recall curve

ROC curve

Comparison table of model performances
