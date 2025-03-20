# Diabetic-Prediction
## Project Overview

This project aims to classify individuals as diabetic or non-diabetic using the Diabetes Health Indicators Dataset. The dataset was originally imbalanced, so we applied Random Undersampling to balance the class distribution. A Logistic Regression model was then trained on the balanced dataset.

## Dataset

> Source: CDC Diabetes Health Indicators Dataset

> Target Variable: Diabetes_binary (0 = Non-Diabetic, 1 = Diabetic)

> Features: 20 health-related indicators

> Size: 253,680 instances

> Missing Values: None

## Data Preprocessing
1) Removed duplicates

2) Converted categorical columns to categorical datatype

3) Encoded categorical features using integer encoding

4) Standardized numeric features (BMI, MentHlth, PhysHlth)

5) Addressed class imbalance using Random Under-Sampling

## Models Implemented

The following machine learning models were trained and evaluated:

1) Logistic Regression

2) Decision Tree Classifier

3) Random Forest Classifier

4) Gradient Boosting Classifier

5) Support Vector Machine (SVM)

6) K-Nearest Neighbors (KNN)

## Model Evaluation

The models were evaluated using the following metrics:

1) Accuracy Score

2) Classification Report (Precision, Recall, F1-Score)

3) Confusion Matrix

4) Model Comparison Chart (Accuracy Scores for all models)

## Results

1) The best-performing model is determined based on accuracy.

2) A bar chart visualizes the accuracy of all models.

3) Sample predictions are displayed for comparison.

## Dependencies

. Python

. pandas

. numpy

. scikit-learn

. imbalanced-learn

. matplotlib

. seaborn

## Installation & Execution

. Install required dependencies:

. Run the script

## Future Improvements

1) Feature selection to remove less important variables.

2) Use SMOTE for synthetic minority oversampling.

3) Implement deep learning models (Neural Networks).

4) Hyperparameter tuning for improved accuracy.


