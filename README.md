# Customer Personality Analysis using Machine Learning

## Project Overview
This project applies both **Supervised** and **Unsupervised Machine Learning** algorithms to the Kaggle *Customer Personality Analysis* dataset. The goal is to perform predictive modeling (Regression & Classification) and customer segmentation (Clustering) to help a retail company optimize its marketing strategies.

---

## Repository Structure
```text
├── data/
│   └── marketing_campaign.csv       # The original dataset (Tab-separated)
├── Customer_Personality_Analysis.ipynb # Main Jupyter Notebook with all sections
├── requirements.txt                 # Required Python libraries
└── README.md                        # Project documentation (This file)

## Key Steps & Findings

### 1. Data Preprocessing & Feature Engineering
* **Missing Values:** Imputed the missing entries in the `Income` column using the **Median** to avoid outlier distortion.
* **Feature Engineering:** Formed three strategic features: `Age` (derived from 2026), `TotalSpending`, and `TotalChildren`.
* **Outlier Filtering:** Removed unrealistic ages (<18 or >100) and zero-income profiles.
* **Encoding:** Applied *Ordinal Label Encoding* to `Education` (Basic=0, Graduation=1, Master=2, PhD=3) and *One-Hot Encoding* to the top 4 categories of `Marital_Status` (grouping the rest as "Other").

### 2. Supervised Learning - Regression (Predicting `TotalSpending`)
We compared three regressors on an 80/20 train/test split. Feature scaling via `StandardScaler` was applied properly to the training set to prevent data leakage.
* **Decision Tree Regressor (max_depth=5)** achieved the best performance with an **$R^2$ Score of 0.8461** and the lowest **RMSE (241.82)**, outperforming Linear and Ridge Regressions.

### 3. Supervised Learning - Classification (Predicting Campaign `Response`)
* **Class Imbalance:** Dealt with a heavily skewed target variable (1,903 negatives vs. 334 positives) by utilizing class balancing techniques.
* **Model Insights:** **Logistic Regression** (configured with `class_weight='balanced'`) yielded the highest **Recall for Class 1 (72%)**, making it the most business-critical model for capturing potential buyers without missing them.

### 4. Unsupervised Learning - Clustering (Customer Segmentation)
* **Optimal Clusters ($k$):** Identified using the **Elbow Method** at $k = 3$.
* **Segmentation Profiles:** 1. *Cluster 0:* Mature, established families with average income and high child counts (Target with family bundles).
  2. *Cluster 1:* High-Income, high-spending VIP couples/singles with no kids (Target with luxury products).
  3. *Cluster 2:* Younger, budget-conscious individuals with lower income (Target with discounts/clearance sales).
