# Hotel Booking Demand - Classification Analysis

## Lab Assignment 3: Scikit-learn Data Preprocessing and Model Performance Evaluation

### Student Information
- **Name:** [Your Name]
- **Student ID:** [Your Student ID]
- **Course:** Machine Learning Lab
- **Date:** August 2026

### Dataset
- **Source:** Kaggle Hotel Booking Demand
- **File:** hotel_bookings.csv
- **Target Variable:** is_canceled

### Project Overview
This project implements data preprocessing pipelines and evaluates two classification models (Logistic Regression and Decision Tree) on hotel booking data to predict cancellations.

### Repository Contents
- `main.ipynb` - Complete Jupyter notebook with all analysis
- `hotel_bookings.csv` - Original dataset
- `cleaned_hotel_bookings_features.csv` - Cleaned features after preprocessing
- `cleaned_hotel_bookings_target.csv` - Cleaned target variable
- `README.md` - This file

### Preprocessing Choices
1. **Dropped Columns with High Missingness:**
   - `company` (94.2% missing values)

2. **Removed Target Leakage Columns:**
   - `reservation_status`
   - `reservation_status_date`

3. **Missing Value Imputation:**
   - Numerical: KNNImputer (n_neighbors=5)
   - Categorical: SimpleImputer (strategy="most_frequent")

4. **Outlier Treatment:**
   - Used IQR method (1.5 * IQR)
   - Removed extreme outliers

5. **Feature Scaling:**
   - Pipeline A: StandardScaler
   - Pipeline B: MinMaxScaler

### Models Evaluated
1. Logistic Regression + StandardScaler
2. Logistic Regression + MinMaxScaler
3. Decision Tree + StandardScaler
4. Decision Tree + MinMaxScaler

### Performance Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrices

### Key Findings
- Logistic Regression with StandardScaler showed best overall performance
- Decision Tree models exhibited signs of overfitting
- Scaling choices impact Logistic Regression more than Decision Trees

### How to Run
```bash
# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn

# Run Jupyter Notebook
jupyter notebook main.ipynb
