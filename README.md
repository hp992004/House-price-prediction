# House Price Prediction

## Introduction

This project aims to predict house prices using a machine learning model based on the provided dataset ('HousePricePrediction.xlsx - Sheet1.csv'). The workflow includes data loading, preprocessing, exploratory data analysis (EDA), feature engineering, model building, and hyperparameter tuning.

## Steps

### Step 1: Data Loading and Exploration

- Import essential libraries: NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn.
- Load the dataset using Pandas from 'HousePricePrediction.xlsx - Sheet1.csv'.
- Display the first few rows to understand the dataset's structure.

### Step 2: Data Cleaning

- Remove rows with missing values using the dropna() method.

### Step 3: Exploratory Data Analysis (EDA)

- Visualize categorical features using barplots to understand their distributions.

### Step 4: Feature Engineering

- One-hot encode categorical variables to prepare the data for machine learning.

### Step 5: Correlation Analysis

- Calculate and visualize a correlation matrix using a heatmap to understand relationships between variables.

### Step 6: Data Splitting

- Split the dataset into features (X) and the target variable (y).
- Perform a train-test split using train_test_split() from scikit-learn.

### Step 7: Model Building

- Initialize a RandomForestRegressor model with default hyperparameters and fit it to the training data.
- Make predictions on the test set.

### Step 8: Hyperparameter Tuning with GridSearchCV

- Define a parameter grid for hyperparameter tuning.
- Employ GridSearchCV to search for the best hyperparameter combination for the RandomForestRegressor model.
- Extract the best hyperparameters for building a tuned model.

### Step 9: Model Evaluation

- Make predictions using the tuned model on the test set.
- Calculate Mean Squared Error (MSE) to evaluate the tuned model's performance.

## Results and Conclusion

The initial RandomForestRegressor model provided predictions, but hyperparameter tuning further improved its performance. The tuned model demonstrated a reduction in Mean Squared Error, indicating enhanced predictive accuracy.
