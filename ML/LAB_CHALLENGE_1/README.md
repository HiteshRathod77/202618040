# Online Shoppers Purchasing Intention Challenge

## Results
- Kaggle Public Score: 87.01%
- Validation Accuracy: 85.61%
- Cross-Validation Accuracy: 83.80%

## Model
- Logistic Regression from scikit-learn
- Parameters: random_state=42, max_iter=1000, class_weight=balanced

## Data Processing
- Numerical features: Median imputation, StandardScaler
- Categorical features: Mode imputation, OneHotEncoder with drop=first
- Row retention: 100% (9864 rows preserved out of 9864)

## Files
- 202618040_challenge.ipynb: Complete notebook
- submission.csv: Final predictions
- train.csv: Training data
- test.csv: Test data

## Requirements Met
- Logistic Regression only
- Row retention above 90%
- All missing values handled
- Correct submission format

## Top Features
- PageValues: 1.7657
- ExitRates: -0.2749
- VisitorType_Returning_Visitor: -0.2274
- Month_Sep: 0.2128
- Month_Oct: 0.1831

## Author
- Student ID: 202618040
- Date: August 27, 2026
