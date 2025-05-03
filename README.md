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

# === 2. LOAD DATA ===
file_path = "C:\\Users\\Dell Laptop\\Desktop\\Data Scientist Wizcore\\new_retail_data.csv"
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

for col in df_encoded.select_dtypes(include='object'):
    le = LabelEncoder()
    df_encoded[col] = df_encoded[col].astype(str)  # Treat NaNs as 'nan'
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# === 9. APPLY KNN IMPUTER ===
imputer = KNNImputer(n_neighbors=5)
imputed_array = imputer.fit_transform(df_encoded)
df_imputed = pd.DataFrame(imputed_array, columns=df_encoded.columns)

# === 10. DECODE BACK TO ORIGINAL CATEGORIES ===
for col, le in label_encoders.items():
    df_imputed[col] = df_imputed[col].round().astype(int)
    df_imputed[col] = le.inverse_transform(df_imputed[col])

# === 11. FINAL LABEL ENCODING FOR MODELING ===
df_final = df_imputed.copy()
label_encoders_final = {}

for col in df_final.select_dtypes(include='object'):
    le = LabelEncoder()
    df_final[col] = df_final[col].astype(str)
    df_final[col] = le.fit_transform(df_final[col])
    label_encoders_final[col] = le

# === 12. SAVE FINAL CLEANED DATA ===
df_final.to_csv("cleaned_data_final.csv", index=False)

# === 13. OPTIONAL: CLEANUP MEMORY ===
gc.collect()

# === 14. FINAL CHECK ===
print("Final cleaned dataset info:")
print(df_final.info())
print(df_final.head())

df_final.info()
df_final.isnull().sum()
