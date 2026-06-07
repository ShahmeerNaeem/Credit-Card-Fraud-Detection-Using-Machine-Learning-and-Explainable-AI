# Shahmeer Naeem
# Credit Card Fraud Detection Project
# Machine learning based fraud detection with model comparison and explainability

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    precision_recall_curve,
    roc_auc_score
)
import pickle
from xgboost import XGBClassifier
import shap

# loading the dataset

data = pd.read_csv('credit card/creditcard.csv')

# basic checking

data.info()

print(data.head())

print('Missing values:', data.isna().sum().sum())

print(data['Class'].value_counts(normalize=True))

# checking class distribution

plt.figure(figsize=(6, 4))
data['Class'].value_counts().plot(kind='bar')
plt.title('Fraud vs Normal Transactions')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Normal', 'Fraud'], rotation=0)
plt.tight_layout()
plt.show()

# separating fraud and normal transactions

fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]

print(fraud.describe())
print(normal.describe())

# comparing transaction amounts

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(fraud['Amount'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Fraud Transactions Amount')
axes[0].set_xlabel('Amount')
axes[0].set_ylabel('Count')

sns.histplot(normal['Amount'], bins=30, kde=True, ax=axes[1])
axes[1].set_title('Normal Transactions Amount')
axes[1].set_xlabel('Amount')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()

# checking which features are most related with fraud

print(data.corr(numeric_only=True)['Class'].sort_values(ascending=False))

# some boxplots for comparison

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.boxplot(x='Class', y='Amount', data=data, ax=axes[0, 0])
axes[0, 0].set_title('Amount vs Class')

sns.boxplot(x='Class', y='V14', data=data, ax=axes[0, 1])
axes[0, 1].set_title('V14 vs Class')

sns.boxplot(x='Class', y='V12', data=data, ax=axes[1, 0])
axes[1, 0].set_title('V12 vs Class')

sns.boxplot(x='Class', y='V10', data=data, ax=axes[1, 1])
axes[1, 1].set_title('V10 vs Class')

plt.tight_layout()
plt.show()

# now performing undersampling on the normal data

undersampled_normal = resample(
    normal,
    n_samples=len(fraud),
    replace=False,
    random_state=42
)

# combining both classes again

balanced_data = pd.concat([fraud, undersampled_normal], axis=0).sample(frac=1, random_state=42)

# checking if both classes are equal now

print(balanced_data['Class'].value_counts())

# now doing train test split for the balanced data

X = balanced_data.drop('Class', axis=1)
y = balanced_data['Class']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# scaling the data

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# training data se mean and std seekh kar test ko transform karte hain

X_test_scaled = scaler.transform(X_test)

# logistic regression model

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# evaluation

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))

# now trying another real world approach without undersampling

X2 = data.drop('Class', axis=1)
y2 = data['Class']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2,random_state=42,test_size=0.2,stratify=y2)

scaler2 = StandardScaler()

# class_weight balanced because fraud data is very small compared to normal

model2 = LogisticRegression(class_weight='balanced', max_iter=1000)

X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)

model2.fit(X2_train_scaled, y2_train)

y2_pred = model2.predict(X2_test_scaled)
y2_prob_lr = model2.predict_proba(X2_test_scaled)[:, 1]

print(confusion_matrix(y2_test, y2_pred))
print(classification_report(y2_test, y2_pred))
print('Precision:', precision_score(y2_test, y2_pred))
print('Recall:', recall_score(y2_test, y2_pred))
print('ROC AUC:', roc_auc_score(y2_test, y2_prob_lr))

# random forest model

rf = RandomForestClassifier(class_weight='balanced',random_state=42,n_estimators=200)

rf.fit(X2_train, y2_train)

y_rf = rf.predict(X2_test)
y_rf_prob = rf.predict_proba(X2_test)[:, 1]

print(confusion_matrix(y2_test, y_rf))
print(classification_report(y2_test, y_rf))
print('Precision:', precision_score(y2_test, y_rf))
print('Recall:', recall_score(y2_test, y_rf))
print('ROC AUC:', roc_auc_score(y2_test, y_rf_prob))

# ROC curve

fpr, tpr, thresholds = roc_curve(y2_test, y_rf_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.show()

# precision recall curve

precision, recall, thresholds_pr = precision_recall_curve(y2_test, y_rf_prob)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.tight_layout()
plt.show()

# threshold optimization

custom_threshold = 0.3
custom_pred = (y_rf_prob >= custom_threshold).astype(int)

print(confusion_matrix(y2_test, custom_pred))
print(classification_report(y2_test, custom_pred))
print('Precision:', precision_score(y2_test, custom_pred))
print('Recall:', recall_score(y2_test, custom_pred))

# cross validation

cv_scores = cross_val_score(rf,X2,y2,cv=5,scoring='f1',n_jobs=-1)

print('Cross Validation F1 Scores:')
print(cv_scores)
print('Average F1 Score:')
print(cv_scores.mean())

# hyperparameter tuning

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1
)

grid.fit(X2_train, y2_train)

print('Best Parameters:')
print(grid.best_params_)
print('Best CV Score:')
print(grid.best_score_)

best_rf = grid.best_estimator_

best_pred = best_rf.predict(X2_test)
best_prob = best_rf.predict_proba(X2_test)[:, 1]

print(confusion_matrix(y2_test, best_pred))
print(classification_report(y2_test, best_pred))
print('Precision:', precision_score(y2_test, best_pred))
print('Recall:', recall_score(y2_test, best_pred))
print('ROC AUC:', roc_auc_score(y2_test, best_prob))

# XGBoost model

xgb = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)

xgb.fit(X2_train, y2_train)

xgb_pred = xgb.predict(X2_test)
xgb_prob = xgb.predict_proba(X2_test)[:, 1]

print(confusion_matrix(y2_test, xgb_pred))
print(classification_report(y2_test, xgb_pred))
print('Precision:', precision_score(y2_test, xgb_pred))
print('Recall:', recall_score(y2_test, xgb_pred))
print('ROC AUC:', roc_auc_score(y2_test, xgb_prob))

# feature importance plot

importance = pd.Series(best_rf.feature_importances_, index=X2.columns)
importance = importance.sort_values(ascending=False)

print(importance)

plt.figure(figsize=(10, 6))
importance.head(10).sort_values().plot(kind='barh')
plt.title('Top 10 Important Features')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# explainability using SHAP

explainer = shap.TreeExplainer(best_rf)
sample_data = X2_test.sample(n=min(500, len(X2_test)), random_state=42)
shap_values = explainer.shap_values(sample_data)

if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], sample_data)
else:
    shap.summary_plot(shap_values, sample_data)

# saving the model

pickle.dump(best_rf, open('fraud_model.pkl', 'wb'))