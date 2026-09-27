# Comparison of Implementations

All three parts use the same train/test split (1197 rows total, 80/20 with random_state=42, 958 train / 239 test), the same preprocessing (median impute on wip, one-hot on quarter/department/day/team, standard scaling on numeric columns), and the same evaluation metrics.

## Linear Regression

| Metric        | sklearn  | manual   | optimized |
|---------------|----------|----------|-----------|
| train time (s)| 0.003864 | 0.005915 | 0.001459  |
| predict (s)   | 0.001017 | 0.000064 | 0.000038  |
| MAE           | 0.101798 | 0.101798 | 0.101798  |
| RMSE          | 0.141897 | 0.141897 | 0.141897  |
| R2            | 0.348388 | 0.348388 | 0.348388  |

Manual and optimized versions match sklearn exactly on all three metrics. This is expected: all three solve the same ordinary least squares problem. The manual version uses the closed-form normal equations via np.linalg.lstsq; the optimized version uses np.linalg.solve with a tiny L2 penalty (1e-6), which is small enough not to change the fit but lets us solve directly instead of via least squares.

## Logistic Regression

| Metric        | sklearn  | manual   | optimized |
|---------------|----------|----------|-----------|
| train time (s)| 0.009557 | 0.048376 | 0.001983  |
| predict (s)   | 0.000140 | 0.000076 | 0.000049  |
| accuracy      | 0.761506 | 0.757322 | 0.753138  |
| precision     | 0.787879 | 0.783920 | 0.785714  |
| recall        | 0.912281 | 0.912281 | 0.900585  |
| f1            | 0.845528 | 0.843243 | 0.839237  |

The manual version uses batch gradient descent (lr=0.1, 2000 iterations). It matches sklearn closely but not exactly, because sklearn's default solver (LBFGS) optimizes a different objective (with L2 regularization by default) using a quasi-Newton method.

The optimized version replaces gradient descent with Newton-Raphson (IRLS). This drops training time from 0.048 s to 0.002 s — about 24x faster — because Newton converges in roughly 6-8 iterations instead of 2000. Accuracy is essentially unchanged (within 0.4%, which on a 239-row test set is one or two flipped predictions). Precision is marginally higher, recall marginally lower.

## Summary

- Linear regression: manual matches sklearn exactly; the optimized version is 4x faster than the manual version on training.
- Logistic regression: manual is close to sklearn; the optimized version is 24x faster than the manual version and stays within noise of sklearn's metrics.
- On this dataset, sklearn's fitted models and our from-scratch models land on essentially the same solution. The only meaningful difference is training time, which Part C addresses by switching to a second-order solver.
