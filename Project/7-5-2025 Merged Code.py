# -*- coding: utf-8 -*-
"""
Finalized Gender Prediction Pipeline
Author: Dell Laptop
Date: 2025-04-29
"""

# === 1. IMPORT LIBRARIES ===
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    mean_squared_error, r2_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# === 2. LOAD DATA ===
file_path = "C:\\Users\\Dell Laptop\\Desktop\\Data Scientist Wizcore\\new_retail_data.csv"
ds = pd.read_csv(file_path)

# === 3. INITIAL EXPLORATION ===
print(ds.info())
print(ds.describe())
print("Missing values:\n", ds.isnull().sum())

# === 4. DROP DUPLICATES & UNUSED COLUMNS ===
ds = ds.drop_duplicates().reset_index(drop=True)
cols_to_drop = [
    "Transaction_ID", "Name", "Email", "Phone", "Address", "Customer_Segment", 
    "Date", "Time", "Feedback", "Shipping_Method", "Payment_Method", 
    "Order_Status", "Ratings", "products"
]
ds.drop(columns=cols_to_drop, errors='ignore', inplace=True)

# === 5. CLEAN GENDER COLUMN ===
ds = ds.dropna(subset=['Gender'])
ds = ds[ds['Gender'].astype(str).str.strip() != '']
ds.reset_index(drop=True, inplace=True)

# === 6. ENCODE TARGET VARIABLE ===
le_gender = LabelEncoder()
ds['Gender'] = le_gender.fit_transform(ds['Gender'])  # Female=0, Male=1
print(dict(enumerate(le_gender.classes_)))

# === 7. MANUAL ENCODING FOR SPECIFIC FEATURES ===
ds['Income'] = ds['Income'].map({'Low': 1, 'Medium': 2, 'High': 3})
ds['Month'] = ds['Month'].map({
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
    'June': 6, 'July': 7, 'August': 8, 'September': 9, 
    'October': 10, 'November': 11, 'December': 12
})

# === 8. LABEL ENCODE OBJECT COLUMNS FOR IMPUTATION ===
df_encoded = ds.copy()
label_encoders = {}
for col in df_encoded.select_dtypes(include='object'):
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

# === 9. IMPUTE MISSING VALUES USING KNN ===
imputer = KNNImputer(n_neighbors=5)
imputed_array = imputer.fit_transform(df_encoded)
df_imputed = pd.DataFrame(imputed_array, columns=df_encoded.columns)

# === 10. REVERSE ENCODING FOR OBJECT COLUMNS ===
for col, le in label_encoders.items():
    df_imputed[col] = le.inverse_transform(df_imputed[col].round().astype(int))

# === 11. FINAL LABEL ENCODING FOR MODELING ===
df_final = df_imputed.copy()
label_encoders_final = {}
for col in df_final.select_dtypes(include='object'):
    le = LabelEncoder()
    df_final[col] = le.fit_transform(df_final[col].astype(str))
    label_encoders_final[col] = le

# === 12. SAVE CLEANED DATA ===
df_final.to_csv("cleaned_data_final.csv", index=False)

# === 13. SPLIT DATA ===
X = df_final.drop(columns=['Gender'])
y = df_final['Gender']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === 14. BALANCE DATA USING SMOTE ===
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# === 15. TRAINING & EVALUATION UTIL FUNCTION ===
def evaluate_model(pipeline, X_train, y_train, X_test, y_test, model_name="Model"):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline.named_steps[list(pipeline.named_steps)[-1]], 'predict_proba'):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"\n=== {model_name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# === 16. MODELING ===

# Logistic Regression
logreg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])
evaluate_model(logreg_pipeline, X_train_resampled, y_train_resampled, X_test, y_test, "Logistic Regression")

# Random Forest
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])
evaluate_model(rf_pipeline, X_train_resampled, y_train_resampled, X_test, y_test, "Random Forest")

# XGBoost
xgb_model = XGBClassifier(
    use_label_encoder=False, eval_metric='logloss',
    scale_pos_weight=1.5, random_state=42
)
xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', xgb_model)
])
evaluate_model(xgb_pipeline, X_train_resampled, y_train_resampled, X_test, y_test, "XGBoost")

# KNN
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
evaluate_model(knn_pipeline, X_train_resampled, y_train_resampled, X_test, y_test, "KNN")

# === 17. CLEANUP ===
gc.collect()

     #precision    recall  f1-score   support
#0       0.39      0.51      0.44     22819
#1       0.63      0.52      0.57     37519
