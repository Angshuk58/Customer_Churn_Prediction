Customer Churn Prediction
This repository contains the Jupyter notebook for predicting customer churn in a bank using a dataset that includes various customer attributes.

Table of Contents
Introduction
Data Loading and Exploration
Libraries Used
Dataset
Data Cleaning and Exploration
Model Training
Results
Conclusion
Introduction
This notebook focuses on predicting customer churn in a bank using the dataset containing various customer attributes. Customer churn, or the rate at which customers leave a company, is a critical metric for businesses.

Data Loading and Exploration
The notebook begins by loading the dataset into a pandas DataFrame and examining its structure. This includes understanding the column names, data types, and summary statistics to gain initial insights into the data.

Libraries Used
The following libraries are used in this notebook:

pandas
numpy
seaborn
matplotlib
scikit-learn
tensorflow
pickle
Dataset
The dataset contains details of bank customers, including their credit score, geography, gender, age, tenure, balance, number of products held, credit card ownership, activity status, estimated salary, and whether they exited the bank.

Data Cleaning and Exploration
The notebook performs data cleaning and exploratory data analysis to understand the distribution of various features and their relationship with the target variable (customer churn).

Model Training
Using Tensorflow and Keras I created an ANN to predict customer churn, including data preprocessing steps such as label encoding and standard scaling.

Results
The results of the model training are evaluated to determine the performance of the predictive models.

Conclusion
The notebook concludes with insights gained from the analysis and model predictions, along with potential areas for further improvement.
