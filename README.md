# Diabetic-Prediction
## Project Overview

This project aims to classify individuals as diabetic or non-diabetic using the Diabetes Health Indicators Dataset. The dataset was originally imbalanced, so we applied Random Undersampling to balance the class distribution. A Logistic Regression model was then trained on the balanced dataset.

Dataset

Source: CDC Diabetes Health Indicators Dataset

Target Variable: Diabetes_binary (0 = Non-Diabetic, 1 = Diabetic)

Features: 20 health-related indicators

Original Class Distribution:

Non-Diabetic (0): 218,334 instances

Diabetic (1): 35,346 instances

Balanced Class Distribution (After undersampling):

Non-Diabetic (0): 35,346 instances

Diabetic (1): 35,346 instances

Implementation Steps

Load the dataset: Read the CSV file into a Pandas DataFrame.

Define features and target: Extract feature variables (X) and the target column (Diabetes_binary).

Handle class imbalance: Apply Random Undersampling to equalize both classes.

Split dataset: Use an 80-20 train-test split.

Train a Logistic Regression model: Fit the model on the training data.

Evaluate performance:

Accuracy score

Classification report (Precision, Recall, F1-score)

Confusion matrix visualization

Results

Accuracy: Approximately 74%

Confusion Matrix:

Displays true positives, true negatives, false positives, and false negatives.

Helps in understanding model performance on diabetic vs. non-diabetic predictions.

Dependencies

Python

pandas

numpy

scikit-learn

imbalanced-learn

matplotlib

seaborn

Future Improvements

Experiment with feature engineering to improve predictive performance.

Try different sampling techniques such as SMOTE (Synthetic Minority Over-sampling Technique).

Use hyperparameter tuning for better model performance.

Implement other classifiers like Random Forest, SVM, or Neural Networks for comparison.
