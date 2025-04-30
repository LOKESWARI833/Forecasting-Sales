Sales Prediction Model

This project implements a machine learning model to predict product units sold based on historical sales data. The model uses features such as ad spend, unit price, item ID, and various date-related features to forecast future sales volumes.
Key Features

Data Preprocessing: Handles missing values, converts date strings to datetime objects, and engineers temporal features
Exploratory Data Analysis: Includes distribution analysis, time series visualization, and correlation analysis
Feature Engineering: Extracts year, month, day, and day-of-week from dates
Model Implementation: Uses Random Forest Regression with hyperparameter tuning via GridSearchCV
Evaluation: Measures model performance using Mean Squared Error on both training and validation sets

The final model achieves good predictive performance and can be used to forecast product sales, enabling better inventory management and marketing budget allocation.
