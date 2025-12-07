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

## ✔ Model Selection Summary

Since our problem is binary classification (returned: 0/1),
we evaluated four models:

- Logistic Regression (baseline)
- Decision Tree
- Random Forest (best performing)
- XGBoost (advanced)

Random Forest outperformed other models because it handles
complex patterns, categorical variables, and non-linear relationships.

