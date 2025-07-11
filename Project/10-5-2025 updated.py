@@ -1,598 +1,160 @@
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:22:49 2025

@author: Dell Laptop
Finalized Gender Prediction Pipeline
Author: Dell Laptop
Date: 2025-04-29
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score
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
print(ds)

# === 3. INITIAL EXPLORATION ===
print(ds.head())
print(ds.info())
print(ds.describe())
print(f"Total rows: {len(ds)}")
print("Missing values per column:\n", ds.isnull().sum())
print("Duplicated rows:", ds.duplicated().sum())
print("Missing values:\n", ds.isnull().sum())

# === 4. DROP DUPLICATES ===
# === 4. DROP DUPLICATES & UNUSED COLUMNS ===
ds = ds.drop_duplicates().reset_index(drop=True)

# === 5. DROP UNUSED COLUMNS ===
cols_to_drop = [
    "Transaction_ID", "Name", "Email", "Phone", "Address", "Customer_Segment", 
    "Date", "Time", "Feedback", "Shipping_Method", "Payment_Method", 
    "Order_Status", "Ratings", "products"
]
ds = ds.drop(columns=cols_to_drop, errors='ignore')


# === 6. CHECKING HOW MANY GENGER VALUES = BLANK OR NAN

print("Unique values in 'Gender':")
print(ds['Gender'].unique())

print("\nValue counts:")
print(ds['Gender'].value_counts(dropna=False))

#Value counts:
#Gender
#Male      187596
#Female    114093
#NaN          317
#Name: count, dtype: int64

blank_string_count = (ds['Gender'].astype(str).str.strip() == '').sum()
print(f"Blank string values in Gender: {blank_string_count}")


# # === 7. DROPPING ROWS WHERE GENDER (TARGET VARIABLE) = BLANK OR NAN
ds = ds.dropna(subset=['Gender'])  # Remove NaNs
ds = ds[ds['Gender'].astype(str).str.strip() != '']  # Remove empty strings


# CHECK HOW MANY ROWS WERE DELETED
deleted_rows = ds.shape[0] - (
    ds.dropna(subset=['Gender'])
      .loc[ds['Gender'].astype(str).str.strip() != '']
      .shape[0]
)
print(f"Rows deleted: {deleted_rows}") #317 row deleted


# Reset index after deletion
ds = ds.reset_index(drop=True)

ds.drop(columns=cols_to_drop, errors='ignore', inplace=True)

# IDENTIFY THE GENDER CLASSES
le = LabelEncoder()
y = le.fit_transform(ds['Gender'])
# === 5. CLEAN GENDER COLUMN ===
ds = ds.dropna(subset=['Gender'])
ds = ds[ds['Gender'].astype(str).str.strip() != '']
ds.reset_index(drop=True, inplace=True)

# To see what 0 and 1 mean:
print(dict(enumerate(le.classes_))) #{0: 'Female', 1: 'Male'}
# === 6. ENCODE TARGET VARIABLE ===
le_gender = LabelEncoder()
ds['Gender'] = le_gender.fit_transform(ds['Gender'])  # Female=0, Male=1
print(dict(enumerate(le_gender.classes_)))


# === 6. CHECKING FOR OBJECT COLUMNS
object_cols = list(ds.select_dtypes(include=['category','object']))
count = len(object_cols)
print(object_cols)
print(count)



# === 7. MANUAL ENCODING ===
income_map = {'Low': 1, 'Medium': 2, 'High': 3}
month_map = {
# === 7. MANUAL ENCODING FOR SPECIFIC FEATURES ===
ds['Income'] = ds['Income'].map({'Low': 1, 'Medium': 2, 'High': 3})
ds['Month'] = ds['Month'].map({
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
    'June': 6, 'July': 7, 'August': 8, 'September': 9, 
    'October': 10, 'November': 11, 'December': 12
}

ds['Income'] = ds['Income'].map(income_map)
ds['Month'] = ds['Month'].map(month_map)

})



# === 8. TEMPORARY LABEL ENCODING FOR IMPUTATION ===
# === 8. LABEL ENCODE OBJECT COLUMNS FOR IMPUTATION ===
df_encoded = ds.copy()
label_encoders = {}

for col in df_encoded.select_dtypes(include='object'):
    le = LabelEncoder()
    df_encoded[col] = df_encoded[col].astype(str)  # Treat NaNs as 'nan'
    df_encoded[col] = le.fit_transform(df_encoded[col])
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

# === 9. APPLY KNN IMPUTER ===
# === 9. IMPUTE MISSING VALUES USING KNN ===
imputer = KNNImputer(n_neighbors=5)
imputed_array = imputer.fit_transform(df_encoded)
df_imputed = pd.DataFrame(imputed_array, columns=df_encoded.columns)

# === 10. DECODE BACK TO ORIGINAL CATEGORIES ===
# === 10. REVERSE ENCODING FOR OBJECT COLUMNS ===
for col, le in label_encoders.items():
    df_imputed[col] = df_imputed[col].round().astype(int)
    df_imputed[col] = le.inverse_transform(df_imputed[col])
    df_imputed[col] = le.inverse_transform(df_imputed[col].round().astype(int))

# === 11. FINAL LABEL ENCODING FOR MODELING ===
df_final = df_imputed.copy()
label_encoders_final = {}

for col in df_final.select_dtypes(include='object'):
    le = LabelEncoder()
    df_final[col] = df_final[col].astype(str)
    df_final[col] = le.fit_transform(df_final[col])
    label_encoders_final[col] = le

# === 12. SAVE FINAL CLEANED DATA ===
df_final.to_csv("cleaned_data_final3.csv", index=False)

# === 13. OPTIONAL: CLEANUP MEMORY ===
gc.collect()

# === 14. FINAL CHECK ===
print("Final cleaned dataset info:")
print(df_final.info())

print(df_final.head())

df_final.info()
df_final.isnull().sum()


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

#Count of Product Type by Gender
sns.countplot(data=df_final, x='Product_Type', hue='Gender', palette='Set2')
plt.title("Count of Product Types by Gender")
plt.xlabel("Product Type")
plt.ylabel("Count")
plt.legend(title="Gender")
plt.show()

#Proportion of Genders per Product Type
cross_tab = pd.crosstab(df_final['Product_Type'], df_final['Gender'], normalize='index')
cross_tab.plot(kind='bar', stacked=True, colormap='Set2')
plt.title("Proportion of Genders per Product Type")
plt.xlabel("Product Type")
plt.ylabel("Proportion")
plt.legend(title="Gender")
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
    df_final[col] = le.fit_transform(df_final[col].astype(str))
    label_encoders_final[col] = le

# === 13. SAVE FINAL CLEANED DATA ===
# === 12. SAVE CLEANED DATA ===
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
# === 13. SPLIT DATA ===
X = df_final.drop(columns=['Gender'])
y = df_final['Gender']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) <= 10 else None
    X, y, test_size=0.2, stratify=y, random_state=42
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



# === SUPERVISED LEARNING ===
# === TARGET VARIABLE = GENDER - CATEGORICAL VLAUES - BINARY

# === MODEL 1: LINEAR REGRESSION ===

# === 1. Create pipeline ===
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# === 2. Train the pipeline ===
pipeline.fit(X_train, y_train)

# === 3. Predict and Evaluate ===
y_pred = pipeline.predict(X_test)

model = pipeline.named_steps['regressor']


# === PRINT RESULTS ===
print("Intercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(X_train.columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred)) #lower = better performance - 0.234
print("R² Score:", r2_score(y_test, y_pred)) #0.00622 - not good fit

# === 6. PLOT ACTUAL VS PREDICTED VALUES ===
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === MODEL 2: LOGISTICAL REGRESSION ===


# === 1. Create pipeline ===
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])

# === 2. Train the pipeline ===
pipeline.fit(X_train, y_train)

# === 3. Predict and Evaluate ===
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred)) #62% of predictions are correct
print("\nClassification Report:\n", classification_report(y_test, y_pred))
  
    # precision    recall  f1-score   support

#0       1.00      0.00      0.00     22819
#1       0.62      1.00      0.77     37519
    #class 0 
    #Precision = 100% → When the model predicts class 0, it's always right.
    #Recall = 0% → But it never actually predicts class 0 when it should.
    #F1 Score = 0% → Because recall is 0, the overall performance on class 0 is poor.
    
    #class 1
    #Precision = 62% → Of all class 1 predictions, 62% were correct.
    #Recall = 100% → It correctly found all actual class 1 cases.
    #F1 Score = 77% → Balanced performance, but still not great.
    
    
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve
)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
#Shows where the model is getting predictions right or wrong.

print(y_train.value_counts())


# === IMBALANCE DETECTED - imbalanced dataset, your model is likely biased toward 
#predicting the majority class (Gender = 1)

# === modify pipeline to handle class imbalance, using class_weight='balanced'

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(class_weight='balanced', max_iter=1000))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred)) #52% of predictions are correct
print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # precision    recall  f1-score   support

#0       0.40      0.53      0.45     22819
#1       0.64      0.52      0.57     37519
    
    #model now predicts both classes, not just the majority class — that's progress!
    #However, performance is still modest, with only ~52% accuracy and low precision/recall for both classes.
    #Recall for class 0 improved from 0% to 53%, showing class balancing helped.


# === implement SMOTE (Synthetic Minority Over-sampling Technique) - to improve performance 

!pip install imbalanced-learn

# === 1. Apply SMOTE to balance the training data ===
# === 14. BALANCE DATA USING SMOTE ===
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# === 2. Create pipeline ===
pipeline = Pipeline([
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
    ('logreg', LogisticRegression(max_iter=1000))
    ('logreg', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])
evaluate_model(logreg_pipeline, X_train_resampled, y_train_resampled, X_test, y_test, "Logistic Regression")

# === 3. Train pipeline on resampled data ===
pipeline.fit(X_train_resampled, y_train_resampled)

# === 4. Predict and Evaluate ===
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # precision    recall  f1-score   support
#0       0.40      0.49      0.44     22819
#1       0.64      0.55      0.59     37519

    #SMOTE helped slightly, but overall performance is still modest.
    #The model now captures more class 0 cases (recall ↑ from 0.0 to 0.49), but struggles with precision.
    #Class 1 remains dominant in prediction, with slightly better metrics.

# === Logistic Regression may not be powerful enough 
# === Try a Random Forest Classifier

# === MODEL 3: RANDOM FOREST ===

from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
# Random Forest
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])
evaluate_model(rf_pipeline, X_train_resampled, y_train_resampled, X_test, y_test, "Random Forest")

pipeline.fit(X_train_resampled, y_train_resampled)

y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred)) #59% of predictions are correct
print("\nClassification Report:\n", classification_report(y_test, y_pred))
    #precision    recall  f1-score   support
#0       0.40      0.15      0.22     22819
#1       0.63      0.86      0.73     37519

# === 5. Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Male (0)', 'Female (1)'])

plt.figure(figsize=(6, 4))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - Random Forest")
plt.grid(False)
plt.show()

#The model predicts "Female" (1) very frequently.
#Many male samples are misclassified as female → 19,334 false positives.
#True positives (32,393) are high, showing it performs better on females.
#True negatives (3,485) are quite low, indicating difficulty identifying males.



# === MODEL 4: XGBoost ===

!pip install xgboost

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# 1. Create XGBoost model with imbalance handling
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=1.5,  # Adjust this based on class imbalance ratio
    random_state=42
# XGBoost
xgb_model = XGBClassifier(
    use_label_encoder=False, eval_metric='logloss',
    scale_pos_weight=1.5, random_state=42
)

# 2. Create pipeline
pipeline = Pipeline([
xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', model)
    ('xgb', xgb_model)
])
evaluate_model(xgb_pipeline, X_train_resampled, y_train_resampled, X_test, y_test, "XGBoost")

# 3. Fit model
pipeline.fit(X_train_resampled, y_train_resampled)

# 4. Predict
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]  # For ROC-AUC

# 5. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred)) #56% of predictions are correct
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba)) #54%

     #precision    recall  f1-score   support
#0       0.39      0.01      0.01     22819
#1       0.62      0.99      0.77     37519


# === MODEL 5: KNN

from sklearn.neighbors import KNeighborsClassifier


# 1. Create pipeline with scaler and KNN
pipeline = Pipeline([
# KNN
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))  # You can tune k later
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
evaluate_model(knn_pipeline, X_train_resampled, y_train_resampled, X_test, y_test, "KNN")

# 2. Train the model
pipeline.fit(X_train, y_train)

# 3. Predict on test set
y_pred = pipeline.predict(X_test)

# 4. Evaluate the results
print("Accuracy:", accuracy_score(y_test, y_pred)) #56% of predictions are correct
print("\nClassification Report:\n", classification_report(y_test, y_pred))

    #precision    recall  f1-score   support
#0       0.39      0.30      0.34     22819
#1       0.63      0.72      0.67     37519

# Elbow Method (using Within-Cluster-Sum-of-Squares) - OPTIMAL NUMBER OF CLUSTERS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []  # Within-cluster sum of squares
K = range(1, 11)  # Try 1 to 10 clusters

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the elbow
plt.plot(K, wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method to Find Optimal k')
plt.grid(True)
plt.show()

# Lets try Optimal k = 3

# 1. Create pipeline with scaler and KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=3))  # You can tune k later
])

# 2. Train the model
pipeline.fit(X_train, y_train)

# 3. Predict on test set
y_pred = pipeline.predict(X_test)

# 4. Evaluate the results
print("Accuracy:", accuracy_score(y_test, y_pred)) #55% of predictions are correct
print("\nClassification Report:\n", classification_report(y_test, y_pred))

     #precision    recall  f1-score   support
#0       0.39      0.34      0.36     22819
#1       0.63      0.68      0.65     37519


# Lets try Optimal k = 4

# 1. Create pipeline with scaler and KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=4))  # You can tune k later
])

# 2. Train the model
pipeline.fit(X_train, y_train)

# 3. Predict on test set
y_pred = pipeline.predict(X_test)

# 4. Evaluate the results
print("Accuracy:", accuracy_score(y_test, y_pred)) #52% of predictions are correct
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# === 17. CLEANUP ===
gc.collect()

     #precision    recall  f1-score   support
#0       0.39      0.51      0.44     22819
