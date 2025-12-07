# 🛒 E-Commerce Product Return Prediction  
### **Sprint 1 – Data Understanding, Cleaning & Initial EDA**  
**Team Name:** Vector 3.0  
**Hackathon:** HCLTech – Data Science Track  

---

## 📌 **1. Problem Statement**

E-commerce companies face high operational costs due to product returns.  
The objective of this project is to **predict whether an order will be returned** using customer, product, and order-level features.  

Predicting returns helps businesses:  
- Reduce logistics cost  
- Improve product quality  
- Prevent misuse or fraudulent returns  
- Enhance customer satisfaction  

---

## 📂 **2. Dataset Overview**

We selected a dataset that closely represents a real-world e-commerce scenario.  
The dataset contains:

- 5000 rows
- 17 features
- Binary target variable: `returned` (0 = Not Returned, 1 = Returned)

### Key Columns
| Column | Description |
|--------|-------------|
| order_id | Unique order identifier |
| product_id | Product identifier |
| category | Product category (Clothing, Beauty, etc.) |
| product_price | Original product price |
| discount_percent | Discount applied |
| final_price | Selling price after discount |
| order_channel | App / Website / Mobile Web |
| payment_method | UPI / COD / Wallet / Credit Card |
| customer_tenure_days | Days since customer joined |
| num_prior_orders | Number of past orders |
| num_prior_returns | Past return count |
| returned | Target variable |

---

## 🎯 3. Why We Selected This Dataset

We chose this dataset because:

- It matches the business problem directly  
- It contains rich behavioural data (price, discount, tenure, channel, returns history)  
- Enough rows for modelling (5000)  
- Clean, structured, and realistic e-commerce format  
- Suitable for EDA, feature engineering, and machine learning  

---

## 1. Data Loading and Initial Exploration

The dataset `retail_dataset.csv` was loaded into a pandas DataFrame. Basic information about the DataFrame, including its shape, data types, and descriptive statistics, was inspected.

```python
import pandas as pd

df = pd.read_csv('/content/sample_data/retail_dataset.csv')
# Initial display of DataFrame
display(df.head())
# Get information about the DataFrame
df.info()
# Get descriptive statistics
display(df.describe())
```

## 2. Handling Missing Values

Missing values were identified using `df.isnull().sum()`. Columns with missing values included `product_price`, `discount_percent`, `order_channel`, and `region`. These rows were dropped from the DataFrame.

```python
# Check for missing values
df.isnull().sum()
# Drop rows with any missing values
df.dropna(inplace=True)
# Verify no missing values remain
df.isnull().sum()
```

## 3. Handling Duplicate Rows

Duplicate rows were identified and removed from the DataFrame to ensure data integrity.

```python
# Check for duplicate rows
print(f"Number of duplicate rows: {df.duplicated().sum()}")
display(df[df.duplicated()].head())
# Drop duplicate rows
df.drop_duplicates(inplace=True)
print(f"Number of duplicate rows after removal: {df.duplicated().sum()}")
```

## 4. Outlier Detection and Removal

Outliers were detected and removed for numerical columns, specifically `product_price` and `delivery_days`, using the Interquartile Range (IQR) method. Box plots were used to visualize the distribution before and after outlier removal.

### Product Price Outliers

```python
Q1 = df['product_price'].quantile(0.25)
Q3 = df['product_price'].quantile(0.75)
IQR = Q3 - Q1

outlier_threshold_upper = Q3 + 1.5 * IQR
outlier_threshold_lower = Q1 - 1.5 * IQR

outliers_product_price = df[(df['product_price'] < outlier_threshold_lower) | (df['product_price'] > outlier_threshold_upper)]

print(f"Number of outliers in 'product_price': {len(outliers_product_price)}")
display(outliers_product_price.head())

df = df[(df['product_price'] >= outlier_threshold_lower) & (df['product_price'] <= outlier_threshold_upper)]
print(f"Number of rows after removing 'product_price' outliers: {len(df)}")

import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(data=df, x='returned', y='product_price')
plt.title('Boxplot of Product Price by Returned Status (Outliers Removed)')
plt.show()
```

### Delivery Days Outliers

```python
Q1_delivery = df['delivery_days'].quantile(0.25)
Q3_delivery = df['delivery_days'].quantile(0.75)
IQR_delivery = Q3_delivery - Q1_delivery

outlier_threshold_upper_delivery = Q3_delivery + 1.5 * IQR_delivery
outlier_threshold_lower_delivery = Q1_delivery - 1.5 * IQR_delivery

outliers_delivery_days = df[(df['delivery_days'] < outlier_threshold_lower_delivery) | (df['delivery_days'] > outlier_threshold_upper_delivery)]

print(f"Number of outliers in 'delivery_days': {len(outliers_delivery_days)}")
display(outliers_delivery_days.head())

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['delivery_days'])
plt.title('Boxplot of Delivery Days (Before Outlier Removal)')
plt.xlabel('Delivery Days')
plt.show()

df = df[(df['delivery_days'] >= outlier_threshold_lower_delivery) & (df['delivery_days'] <= outlier_threshold_upper_delivery)]
print(f"Number of rows after removing 'delivery_days' outliers: {len(df)}")

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['delivery_days'])
plt.title('Boxplot of Delivery Days (After Outlier Removal)')
plt.xlabel('Delivery Days')
plt.show()
```

## 5. Feature Engineering and Dropping Irrelevant Columns

- The `customer_tenure_days` column was dropped as it was found to have a very low correlation with the `returned` target variable.
- The `final_price` column was dropped to avoid multicollinearity, as it is derived from `product_price` and `discount_percent`.
- A new feature `return_ratio` was created by dividing `num_prior_returns` by `(num_prior_orders + 1)` to provide a normalized measure of return behavior. The column order was then adjusted to place `return_ratio` next to `returned`.

```python
df.drop(columns=['customer_tenure_days'], inplace=True)
print(f"DataFrame shape after dropping 'customer_tenure_days': {df.shape}")

df.drop(columns=['final_price'], inplace=True)
print(f"DataFrame shape after dropping 'final_price': {df.shape}")

df['return_ratio'] = df['num_prior_returns'] / (df['num_prior_orders'] + 1)

columns = df.columns.tolist()
returned_index = columns.index('returned')
return_ratio_index = columns.index('return_ratio')
columns[returned_index], columns[return_ratio_index] = columns[return_ratio_index], columns[returned_index]
df = df[columns]
print("Columns after swapping 'returned' and 'return_ratio':")
display(df.head())
```

## 6. Categorical Data Cleaning

- Corrected a typo in the `category` column (`clOThing` was replaced with `Clothing`).
- Investigated and removed rows where `payment_method` was marked as `??`.

```python
df['category'] = df['category'].replace('clOThing', 'Clothing')
print("Unique values in 'category' after correction:")
display(df['category'].unique())

unknown_payment_method_rows = df[df['payment_method'] == '??']
print(f"Number of rows with '??' in payment_method: {len(unknown_payment_method_rows)}")
display(unknown_payment_method_rows.head())

df = df[df['payment_method'] != '??']
print(f"Number of rows after removing '??' from payment_method: {len(df)}")

print("Value counts for 'payment_method' after cleaning:")
df['payment_method'].value_counts()
```

## 7. Correlation Analysis

A correlation matrix and heatmap of numerical features were generated to understand the relationships between different variables, especially with the `returned` target variable.

```python
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numerical_cols].corr()
print("Correlation Matrix:")
display(correlation_matrix)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
```

## ✔ Model Selection Summary

Since our problem is binary classification (returned: 0/1),
we evaluated four models:

- Logistic Regression 
- Decision Tree
- Random Forest 
- XGBoost

1. Logistic Regression
Accuracy: 0.7697
Precision: 0.7663
Recall: 0.7683
F1 Score: 0.7673
2. Random Forest Classifier
Accuracy: 0.7993
3.Decision Tree
Accuracy: 0.7599
4. XGBoost Classifier
Accuracy: 0.8276


XG Boost outperformed other models because it handles
complex patterns, categorical variables, and non-linear relationships.

