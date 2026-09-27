# DS605 Lab 5 — Productivity Prediction of Garment Employees

This lab compares a scikit-learn implementation of linear and logistic regression against a from-scratch NumPy/Pandas implementation, and then optimizes the manual version. The dataset is the UCI Garment Employee Productivity dataset.

## Files

- `garments_worker_productivity.csv` — the raw dataset (1197 rows, 15 columns).
- `part_a_sklearn.ipynb` — scikit-learn implementation. Uses sklearn's Pipeline, SimpleImputer, OneHotEncoder, StandardScaler, LinearRegression, LogisticRegression, and metrics.
- `part_b_scratch.ipynb` — from-scratch implementation using only NumPy and Pandas. Includes manual imputation, one-hot encoding, standardization, closed-form linear regression, and gradient descent logistic regression.
- `part_c_optimization.ipynb` — optimized manual implementation. Replaces gradient descent with Newton-Raphson for logistic regression and uses np.linalg.solve for the closed-form linear regression.
- `comparison.md` — side-by-side tables of all three implementations.

## Setup

Only numpy, pandas, and scikit-learn are needed. On Anaconda these are already installed.

## How to run

Run the notebooks in order: `part_a_sklearn.ipynb`, then `part_b_scratch.ipynb`, then `part_c_optimization.ipynb`. Each notebook has cells that print their own results.

## Data

The dataset contains production records from a garment factory. Each row is a team-day observation with features such as targeted productivity, standard minute value, overtime, incentives, work in progress, and number of workers. The regression target is `actual_productivity`. The classification target is `MeetsTarget`, defined as 1 when `actual_productivity >= targeted_productivity` and 0 otherwise.

Two targets, so the work is split:

- Regression predicts `actual_productivity`.
- Classification predicts `MeetsTarget`. `actual_productivity` is not used as an input for classification, since that would leak the target.

## Preprocessing

The raw CSV has a few issues that had to be handled first.

- `wip` has 506 missing values (about 42% of rows). We impute with the median from the training set. Mean imputation was avoided because `wip` has large outliers.
- `department` has inconsistent labels — `sweing` (typo) and `finishing ` (trailing space). Both are cleaned before encoding.
- `date` is dropped. `quarter` and `day` already carry the relevant temporal information.
- `quarter`, `department`, `day`, and `team` are treated as categorical and one-hot encoded.
- Numeric columns are standardized. The scaler is fit only on the training set.

The train/test split is 80/20 with `random_state=42`, done once and reused across all three notebooks. This keeps the comparison fair.

## Results

Linear regression: manual and optimized versions match scikit-learn exactly on MAE, RMSE, and R2. Both solve the same least-squares problem, so identical results are expected.

Logistic regression: manual gradient descent lands close to scikit-learn, and the optimized Newton-Raphson version drops training time by about 24x while staying within 0.4% accuracy of scikit-learn. The small difference in accuracy comes from the two solvers reaching slightly different near-optimal weights, which on a 239-row test set flips one or two predictions.

Full tables are in `comparison.md`.

## Observations

- The regression R2 is around 0.35. The model explains roughly a third of the variance in productivity. This dataset is known to be difficult for regression on the raw features; the relationship between the inputs and `actual_productivity` is not strongly linear.
- For classification, accuracy sits around 0.76 with recall around 0.91. The model leans toward predicting class 1, which makes sense given that about 73% of rows meet the target. Precision and recall both land in a reasonable range; F1 is around 0.84.
- Linear regression and logistic regression both produce essentially the same solution whether they are implemented via sklearn, from scratch, or via an optimized manual version. The choice of library matters less than the choice of solver.
- The main cost in the from-scratch logistic regression was the training loop. Two thousand iterations of gradient descent are slow in pure Python. Newton-Raphson needs only a handful of iterations and is dramatically faster, which is why Part C uses it.
