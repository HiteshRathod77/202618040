# Lab Assignment - 3
## Scikit-learn: Data Preprocessing and Model Performance Evaluation

**Name:** Hiteshsinh Manharsinh Rathod
**Student ID:** 202618040
**Course:** DS605 - Fundamentals of Machine Learning
**Date:** August 2026

---

## Dataset
- **Source:** Kaggle Hotel Booking Demand
- **File:** hotel_bookings.csv
- **Target Variable:** is_canceled (0 = Not Canceled, 1 = Canceled)

---

## Project Overview
This project builds and compares Scikit-learn preprocessing pipelines and evaluates two classification models (Logistic Regression and Decision Tree) to predict hotel booking cancellations.

---

## Preprocessing Choices

### Dropped Columns
1. **company** - Dropped due to 94.2% missing values
2. **reservation_status** - Removed to prevent target leakage
3. **reservation_status_date** - Removed to prevent target leakage

### Missing Value Handling
- **Numerical:** KNNImputer (n_neighbors=5)
- **Categorical:** SimpleImputer (strategy="most_frequent")

### Feature Encoding
- **Categorical:** OneHotEncoder (handle_unknown="ignore")

### Feature Scaling
- **Pipeline A:** StandardScaler
- **Pipeline B:** MinMaxScaler

### Outlier Treatment
- Used IQR method (1.5 × IQR)
- Removed extreme outliers

---

## Models Evaluated

| # | Model | Preprocessing |
|---|-------|---------------|
| 1 | Logistic Regression | Pipeline A (StandardScaler) |
| 2 | Logistic Regression | Pipeline B (MinMaxScaler) |
| 3 | Decision Tree | Pipeline A (StandardScaler) |
| 4 | Decision Tree | Pipeline B (MinMaxScaler) |

---

## Results Summary

| Model | Test Accuracy | Test F1-Score |
|-------|--------------|---------------|
| Logistic Regression + StandardScaler | 0.8201 | 0.8089 |
| Logistic Regression + MinMaxScaler | 0.8178 | 0.8056 |
| Decision Tree + StandardScaler | 0.8156 | 0.8023 |
| Decision Tree + MinMaxScaler | 0.8142 | 0.8009 |

---

## Final Observations

1. **Best Model:** Logistic Regression with StandardScaler achieved the highest test F1-score (0.8089) and showed the best generalization.

2. **Scaling Impact on Logistic Regression:** StandardScaler performed slightly better than MinMaxScaler, suggesting standardization (zero mean, unit variance) is more suitable for logistic regression's decision boundary.

3. **Scaling Impact on Decision Tree:** Both scaling methods produced similar results, confirming that tree-based models are scale-invariant by nature.

4. **Overfitting:** Decision Tree models showed significant overfitting (train-test accuracy gap > 5%), while Logistic Regression models demonstrated minimal overfitting (gap < 1%).

5. **Practical Insight:** The Logistic Regression model with StandardScaler provides the most reliable and interpretable predictions for hotel booking cancellations.

---

## Repository Contents

| File | Description |
|------|-------------|
| `main.ipynb` | Complete Jupyter Notebook with all analysis |
| `hotel_bookings.csv` | Original dataset |
| `cleaned_hotel_bookings_features.csv` | Cleaned features after preprocessing |
| `cleaned_hotel_bookings_target.csv` | Cleaned target variable |
| `model_comparison_results.csv` | Final comparison table |
| `confusion_matrices_best_models.png` | Confusion matrices for best models |
| `overfitting_analysis.png` | Train-test performance comparison |
| `README.md` | This file |

---

