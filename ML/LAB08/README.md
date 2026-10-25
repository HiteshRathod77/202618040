# Lab 08: Decision Trees and Support Vector Machines

## Overview
This repository contains the implementation for Lab Assignment 8. The assignment involves:
1. Implementing a Decision Tree and a Linear SVM from scratch using NumPy.
2. Comparing the manual implementations against Scikit-learn's optimized models.
3. Visualizing 2D decision boundaries and SVM margins.
4. Exploring nonlinear SVM kernels (Polynomial and RBF) on a synthetic dataset.
5. Evaluating all models on the UCI Banknote Authentication dataset using all four features.

## Dataset
The models are trained on the UCI Banknote Authentication dataset. The dataset file (data_banknote_authentication.txt) is included in the data/ folder. It contains four continuous features (variance, skewness, curtosis, and entropy) and a binary class label.

## Repository Structure
- 202618040_Lab8.ipynb : The main Jupyter Notebook containing all code, plots, and analysis.
- data/ : Folder containing the Banknote Authentication dataset.
- README.md : This file.

## Implementation Details

### Decision Tree (From Scratch)
The custom Decision Tree is implemented using NumPy. It calculates Gini impurity to determine the best splits at each node. A recursive function builds the tree up to a specified maximum depth, and a separate function traverses the tree to make predictions on new data.

### Linear SVM (From Scratch)
The custom Linear SVM uses Hinge Loss and Gradient Descent. The algorithm iterates through the dataset, updating the weights and bias. It penalizes misclassifications and points that fall within the margin, aiming to maximize the distance between the two classes.

### SVM Kernels
For the synthetic dataset, we explore the limitation of linear boundaries. We implement and apply the Polynomial Kernel and the Radial Basis Function (RBF) kernel to transform the data, allowing the SVM to find a non-linear decision boundary.

## How to Run
1. Clone the repository:
   git clone https://github.com/HiteshRathod77/202618040.git

2. Navigate to the Lab 08 directory:
   cd 202618040/ML/LAB08

3. Ensure the required Python libraries are installed (numpy, pandas, matplotlib, scikit-learn).

4. Launch the Jupyter Notebook:
   jupyter notebook 202618040_Lab8.ipynb

## Results Summary
The manual implementations were evaluated against Scikit-learn's built-in models using the Banknote Authentication dataset.

- The Decision Tree from scratch achieved an accuracy comparable to the Scikit-learn implementation, demonstrating the correctness of the Gini impurity split logic.
- The Linear SVM from scratch converged successfully, though the Scikit-learn implementation was significantly faster due to optimized C-backend loops.
- Visualizations of the decision boundaries confirmed that both models successfully separated the classes.
