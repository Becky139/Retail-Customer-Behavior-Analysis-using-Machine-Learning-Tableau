# 🛍️ Retail Customer Behavior Analysis using Machine Learning & Tableau

A data science project following the **CRISP-DM** methodology to analyze retail customer behavior using machine learning and visualization tools.

## 🧠 CRISP-DM Overview

This project follows the 6-phase **CRISP-DM (Cross-Industry Standard Process for Data Mining)** lifecycle:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

---

## 1. 🧩 Business Understanding

**Team Members:** Rebecca Marriott, Nouman Mehar, Suneeta Vota  
**Goal:** Use machine learning and Tableau to analyze purchasing behavior and personalize ERP systems.

### 🎯 Objectives
- Predict customer gender using behavioral and transactional data.
- Enable better customer segmentation and targeted marketing.

---

## 2. 📊 Data Understanding

### 📦 Dataset Source
- **Platform:** Kaggle  
- **Dataset:** [Retail Transactions Dataset](https://www.kaggle.com/datasets/prasad22/retail-transactions-dataset/data)

### 🧾 Dataset Overview
- **Records:** 302,010
- **Features:** 30
- **Dimensions:**
  - Customer Info: Age, Gender, Income, Segment
  - Transaction History: Date, Amount, Feedback
  - Product Info: Category, Brand
  - Geography: Country (USA, UK, Canada, Australia, Germany)

### ⚠️ Data Quality Issues
- Missing values in `Promotion`
- Inconsistent `Customer Names`
- No duplicates or outliers

---

## 3. 🛠️ Data Preparation

### 🔧 Cleaning & Wrangling
- **Missing Values:** Imputed `Promotion` with `'None'`
- **Standardization:** Cleaned inconsistent customer names
- **Encoding:**
  - Gender → Binary
  - Segment, Category → One-Hot
  - Country → Label/One-Hot

### 🧪 Feature Engineering
- Recency (days since last purchase)
- Total Purchases per customer
- Customer Lifetime Value
- Time features: Year, Month, Day

### ⚖️ Scaling
- Applied Min-Max Scaling for numerical columns

---

## 4. 🤖 Modelling

**Target Variable:** `Gender`  
**Challenge:** Significant class imbalance (Males: 187,596 | Females: 114,093)

### 📈 Models Tested

| Model                 | Accuracy | F1-Male | F1-Female | Notes |
|----------------------|----------|---------|-----------|-------|
| Logistic (Raw)       | 62%      | 0.00    | 0.77      | Strong bias towards Female |
| Logistic (Balanced)  | 52%      | 0.45    | 0.57      | Improved parity |
| SMOTE + Logistic     | 53%      | 0.44    | 0.59      | Slight improvement |
| Random Forest        | 59%      | 0.22    | 0.73      | Better accuracy, weak recall for Males |
| XGBoost              | 56%      | 0.01    | 0.77      | Worst for Males |
| KNN (k=3)            | 55%      | 0.36    | 0.65      | Balanced but moderate performance |

### 📉 Evaluation Metrics
- **Precision-Recall AUC:** ~0.69
- **ROC AUC:** ~0.54 (low, only slightly better than random)
- **Confusion Matrix:** Males frequently misclassified as Females

---

## 5. 📋 Evaluation

### ✅ Key Takeaways
- Logistic Regression with class weighting helps parity
- Random Forest performs better overall, but biased toward Females
- Gender imbalance affects fairness and recall

### 🔍 Recommendations
- Drop `Customer_ID` to reduce data leakage
- Improve feature selection & engineering
- Try deeper or ensemble models

---

## 6. 🚀 Deployment

### 📦 Plan
- Integrate recommendation engine with retail platform
- Tailor marketing using gender & segmentation insights
- Use model predictions to optimize inventory planning

### 🧩 Monitoring & Maintenance
- Retrain models regularly with new data
- Use feedback loops to improve recommendations

### 📊 Dashboard
Created Tableau visualizations to highlight:
- Gender and income segmentation
- Spending over time and by city
- Age vs. income analysis

---

## 📊 Tableau Dashboard Highlights

- **Gender Distribution:** Imbalance affects marketing strategies
- **Spending Over Time:** Seasonal spikes noted
- **City-Level Spending:** London and major cities lead
- **Product Preferences:** Similar patterns across genders
- **Income vs Age:** High spenders mostly aged 30–35

---

## 📁 Repository Structure (Suggested)

```bash
.
├── data/
│   └── retail_transactions.csv
├── notebooks/
│   ├── data_preparation.ipynb
│   ├── modelling.ipynb
│   └── evaluation.ipynb
├── visuals/
│   └── dashboard_screenshots/
├── scripts/
│   ├── preprocessing.py
│   └── train_models.py
├── README.md
└── requirements.txt
