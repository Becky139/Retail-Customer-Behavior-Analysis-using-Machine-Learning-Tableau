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
#reading csv data into spyder IDE
ds= pd.read_csv("C:\\Users\\Dell Laptop\\Desktop\\Data Scientist Wizcore\\new_retail_data.csv")

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
cols_to_drop =["Transaction_ID", "Name", "Email", "Phone", "Address", "Customer_Segment", "Date", "Time", "Feedback", 
                   "Shipping_Method", "Payment_Method", "Order_Status", "Ratings", "products"]
ds2 = ds1.drop (cols_to_drop, axis=1)

list(ds1.columns.values)
list(ds2.columns.values)

ds2.to_csv('ds2.csv')

ds.info()
ds2.info()

object_cols = list(ds2.select_dtypes(include=['category','object']))
print(object_cols)




