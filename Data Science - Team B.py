# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:22:49 2025

@author: Dell Laptop
"""
#importing all the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gc
import calendar
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
#reading csv data into spyder IDE
ds= pd.read_csv("D:\DS\Project\data\data.csv")

print(ds.head())
print(ds.info())
print(len(ds))
print(ds.describe())
ds.isnull().sum()
print(ds.tail())
print(ds.sample(4))
print(ds.isnull().sum())

print(ds.duplicated().sum())

ds1 = ds.drop_duplicates()
print(ds1.duplicated().sum())
ds1.isnull().sum()
list(ds1.columns.values)

ds1.Age.mean()
ds1.Age.min()
ds1.Age.max()

ds1.isnull().sum()
#dropping columns
cols_to_drop =["Transaction_ID", "Name", "Email", "Phone", "Address", 
               "Customer_Segment", "Date", "Time", "Feedback", 
                   "Shipping_Method", "Payment_Method", "Order_Status", 
                   "Ratings", "products"]
ds2 = ds1.drop (cols_to_drop, axis=1)

list(ds1.columns.values)
list(ds2.columns.values)

ds2.to_csv('ds2.csv')

ds.info()
ds2.info()
print(ds2.head(20))
object_cols = list(ds2.select_dtypes(include=['category','object']))
print(object_cols)


# FILL MISSING VALUES
#----------------------------

#Remap Income and Month
income_map = {'Low': 1, 'Medium': 2, 'High': 3}
ds['Income'] = ds['Income'].map(income_map)

month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 
             'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 
             'November': 11, 'December': 12}
ds['Month'] = ds['Month'].map(month_map)

ds.head()

# Encode object columns temporarily
df_encoded = ds.copy()
label_encoders = {}

for col in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[col] = df_encoded[col].astype(str)  # Treat NaNs as 'nan'
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le
    
    
# Apply KNN Imputer
imputer = KNNImputer(n_neighbors=5)
df_imputed_array = imputer.fit_transform(df_encoded)
df_imputed = pd.DataFrame(df_imputed_array, columns=ds.columns)

# Decode label-encoded columns back to original values
for col in label_encoders:
    le = label_encoders[col]
    df_imputed[col] = df_imputed[col].round().astype(int)
    df_imputed[col] = le.inverse_transform(df_imputed[col])
    
df_imputed.info()
df_imputed.head()
    
    


























