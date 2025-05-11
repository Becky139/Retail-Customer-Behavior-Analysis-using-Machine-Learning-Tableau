# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:22:49 2025

@author: Dell Laptop
"""
# === 1. IMPORT LIBRARIES ===
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import calendar
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# === 2. LOAD DATA ===
file_path = "C:/Users/rebec/Downloads/DS/Project/data/data.csv"
ds = pd.read_csv(file_path)
print(ds)

# === 3. INITIAL EXPLORATION ===
print(ds.head())
print(ds.info())
print(ds.describe())
print(f"Total rows: {len(ds)}")
print("Missing values per column:\n", ds.isnull().sum())
print("Duplicated rows:", ds.duplicated().sum())

# === 4. DROP DUPLICATES ===
ds = ds.drop_duplicates().reset_index(drop=True)

# === 5. DROP UNUSED COLUMNS ===
cols_to_drop = [
    "Transaction_ID", "Name", "Email", "Phone", "Address", "Customer_Segment", 
    "Date", "Time", "Feedback", "Shipping_Method", "Payment_Method", 
    "Order_Status", "Ratings", "products"
]
ds = ds.drop(columns=cols_to_drop, errors='ignore')

# === 6. CHECKING FOR OBJECT COLUMNS
object_cols = list(ds.select_dtypes(include=['category','object']))
count = len(object_cols)
print(object_cols)
print(count)

# === 7. MANUAL ENCODING ===
income_map = {'Low': 1, 'Medium': 2, 'High': 3}
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
    'June': 6, 'July': 7, 'August': 8, 'September': 9, 
    'October': 10, 'November': 11, 'December': 12
}

ds['Income'] = ds['Income'].map(income_map)
ds['Month'] = ds['Month'].map(month_map)

# === 8. TEMPORARY LABEL ENCODING FOR IMPUTATION ===
df_encoded = ds.copy()
label_encoders = {}
categorical_cols = df_encoded.select_dtypes(include='object').columns

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = df_encoded[col].fillna("Missing")
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# === 9. APPLY KNN IMPUTER ===
imputer = KNNImputer(n_neighbors=5)
df_imputed_array = imputer.fit_transform(df_encoded)
df_imputed = pd.DataFrame(df_imputed_array, columns=df_encoded.columns)

print(df_imputed)
print(df_imputed.isnull().sum())


# === 10. DECODE BACK TO ORIGINAL CATEGORIES ===
for col in categorical_cols:
    le = label_encoders[col]
    df_imputed[col] = df_imputed[col].round().astype(int)
    df_imputed[col] = le.inverse_transform(df_imputed[col])
    
    
for col in ['City', 'State', 'Country','Gender', 'Product_Category', 'Product_Brand']:
    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0])

# Final check
print(df_imputed['Gender'].value_counts(dropna=False))
print("NaNs left in column:", df_imputed.isna().sum())  

print(df_imputed)

df_final = df_imputed.copy()
df_final.info()
df_final.isnull().sum()

# === 11. Graphs ===

#Bar Plot to check male and female customers
plt.figure(figsize=(6, 4))
sns.countplot(data=df_final, x='Gender', palette='Set2')
plt.title("Gender Distribution")
plt.show()

#Average Total Amount Spent by Gender
plt.subplot(1, 2, 2)
sns.boxplot(data=df_final, x='Gender', y='Total_Amount', palette='Set2')
plt.title("Total Amount Spent by Gender")
plt.tight_layout()
plt.show()

#Shows certain product categories by different genders
sns.countplot(data=df_final, x='Product_Category', hue='Gender')

#Age gap by Gender
sns.kdeplot(data=df_final, x='Age', hue='Gender', fill=True)

#Total_Purchases by Gender
sns.lineplot(data=df_final, x='Total_Purchases', y='Total_Amount', hue='Gender')

#Correlation Heatmap (numeric columns)
plt.figure(figsize=(12, 8))
numeric_cols = df_final.select_dtypes(include='number')
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

#Monthly spending trend by Gender
sns.countplot(data=df_final, x='Month', hue='Gender')

# === 12. FINAL LABEL ENCODING FOR MODELING ===

label_encoders_final = {}

for col in df_final.select_dtypes(include='object'):
    le = LabelEncoder()
    df_final[col] = df_final[col].astype(str)
    df_final[col] = le.fit_transform(df_final[col])
    label_encoders_final[col] = le

# === 13. SAVE FINAL CLEANED DATA ===
df_final.to_csv("cleaned_data_final.csv", index=False)

# === 14. OPTIONAL: CLEANUP MEMORY ===
gc.collect()


# === 15. FINAL CHECK ===
print("Final cleaned dataset info:")
print(df_final.info())
print(df_final.head())

df_final.info()
df_final.isnull().sum()


# === 16. DEFINE X AND y ===
# Replace 'Target_Column' with your actual target column name
target_column = 'Gender'
X = df_final.drop(columns=[target_column])
y = df_final[target_column]

# === 16. TRAIN-TEST SPLIT (80-20) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) <= 10 else None
)

print("Train set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# === 17. SCALE FEATURES ===
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 18. TRAIN LOGISTIC REGRESSION ===
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_scaled, y_train)

# === 19. PREDICT AND EVALUATE ===
y_pred = clf.predict(X_test_scaled)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n✅ Accuracy Score:", accuracy_score(y_test, y_pred))

# Match coefficients to feature names
feature_importance = pd.Series(clf.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)

print("\n📈 Top Influential Features for Gender Prediction:")
print(feature_importance.head(10))

