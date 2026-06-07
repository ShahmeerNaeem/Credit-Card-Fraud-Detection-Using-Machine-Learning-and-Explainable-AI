# Credit Card Fraud Detection Using Machine Learning and Explainable AI
A machine learning project that analyzes credit card transactions and detects fraudulent activity using Logistic Regression, Random Forest, and XGBoost. The project includes class imbalance handling, model evaluation, ROC analysis, hyperparameter tuning, feature importance analysis, and SHAP explainability


## Project Overview

Credit card fraud is a major challenge for financial institutions because fraudulent transactions represent only a very small portion of all transactions. This imbalance makes fraud detection a difficult machine learning problem.

In this project, I analyzed a real-world credit card transaction dataset and built multiple machine learning models to identify fraudulent transactions. I compared different approaches, evaluated model performance using several metrics, and used explainability techniques to understand which features influenced the model's predictions.

## Objectives

* Analyze transaction patterns in credit card data
* Explore differences between fraudulent and legitimate transactions
* Handle class imbalance using undersampling and class weighting
* Build fraud detection models using machine learning
* Compare model performance
* Optimize model parameters
* Interpret model predictions using explainable AI

## Dataset

The dataset contains anonymized credit card transactions.

Features:

* Time
* Amount
* V1 to V28 anonymized features
* Class

Target Variable:

* Class = 0 → Legitimate Transaction
* Class = 1 → Fraudulent Transaction

## Project Workflow

### 1. Data Exploration

* Dataset inspection
* Missing value checking
* Class distribution analysis
* Statistical summaries

### 2. Data Visualization

* Fraud vs normal transaction comparison
* Transaction amount distributions
* Correlation analysis
* Boxplot analysis of important features

### 3. Handling Class Imbalance

* Random undersampling
* Class-weighted machine learning models

### 4. Machine Learning Models

#### Logistic Regression

Used as a baseline classification model.

#### Random Forest

Used to improve predictive performance and feature analysis.

#### XGBoost

Used as an advanced ensemble model for fraud detection.

### 5. Model Evaluation

The models were evaluated using:

* Confusion Matrix
* Precision
* Recall
* F1 Score
* ROC AUC Score
* Precision-Recall Curve

### 6. Model Validation

* 5-Fold Cross Validation
* Hyperparameter Tuning using GridSearchCV

### 7. Explainable AI

SHAP was used to identify which features contributed most to fraud predictions and to improve model interpretability.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* SHAP
* Pickle

## Key Learning Outcomes

Through this project, I gained hands-on experience in:

* Fraud detection systems
* Classification algorithms
* Handling imbalanced datasets
* Model optimization
* Performance evaluation
* Explainable AI
* Feature importance analysis
* Machine learning model deployment

## Future Improvements

* Deploy the model using Streamlit
* Create a real-time fraud detection dashboard
* Experiment with deep learning models
* Integrate automated alert systems

## Author

Shahmeer Naeem

Data Science Student interested in Machine Learning, Data Analytics, and Artificial Intelligence.
