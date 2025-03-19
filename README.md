# Diabetic-Prediction
## Project Overview

This project aims to classify individuals as diabetic or non-diabetic using the Diabetes Health Indicators Dataset. The dataset was originally imbalanced, so we applied Random Undersampling to balance the class distribution. A Logistic Regression model was then trained on the balanced dataset.

## Dataset

> Source: CDC Diabetes Health Indicators Dataset

> Target Variable: Diabetes_binary (0 = Non-Diabetic, 1 = Diabetic)

> Features: 20 health-related indicators

> Original Class Distribution:

  1. Non-Diabetic (0): 218,334 instances

  2. Diabetic (1): 35,346 instances

> Balanced Class Distribution (After undersampling):

  1. Non-Diabetic (0): 35,346 instances

  2. Diabetic (1): 35,346 instances

## Implementation Steps

1) Load the dataset: Read the CSV file into a Pandas DataFrame.

2) Define features and target: Extract feature variables (X) and the target column (Diabetes_binary).

3)Handle class imbalance: Apply Random Undersampling to equalize both classes.

4) Split dataset: Use an 80-20 train-test split.

5) Train a Logistic Regression model: Fit the model on the training data.

6) Evaluate performance:

   > Accuracy score

   > Classification report (Precision, Recall, F1-score)

   > Confusion matrix visualization

## Results

> Accuracy: Approximately 74%

> Confusion Matrix:

  . Displays true positives, true negatives, false positives, and false negatives.

  . Helps in understanding model performance on diabetic vs. non-diabetic predictions.

## Dependencies

. Python

. pandas

. numpy

. scikit-learn

. imbalanced-learn

. matplotlib

. seaborn

## Future Improvements

1) Experiment with feature engineering to improve predictive performance.

2) Try different sampling techniques such as SMOTE (Synthetic Minority Over-sampling Technique).

3) Use hyperparameter tuning for better model performance.

4) Implement other classifiers like Random Forest, SVM, or Neural Networks for comparison.
