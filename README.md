# Customer Churn Prediction

![image alt]([https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.cleartouch.in%2Fblog%2Fwhat-is-customer-churn-and-how-do-you-prevent-it%2F&psig=AOvVaw1eJYB5QgZeW-ww8cAr8-yX&ust=1736075777340000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCNC1r7T424oDFQAAAAAdAAAAABAE](https://miro.medium.com/v2/resize:fit:1400/0*d58iZ6esNNcfntQ7))

This repository contains the Jupyter notebook for predicting customer churn in a bank using a dataset that includes various customer attributes.

## Table of Contents
- Introduction
- Data Loading and Exploration
- Data Cleaning and Preprocessing
- ANN Implementation using Tensorflow and keras
- Model Training
- Model Evaluation
- Conclusion

## Introduction
This notebook focuses on predicting customer churn in a bank using the dataset containing various customer attributes. Customer churn, or the rate at which customers leave a company or stop using its services, is a critical metric for businesses, as it directly impacts revenue and customer retention.

The dataset contains details of bank customers, including their credit score, geography, gender, age, tenure, balance, number of products held, credit card ownership, activity status, estimated salary, and other relevant variables. The goal is to explore the dataset, and create an ANN Model using the Tensorflow and Keras library to predict churn

## Data Loading and Exploration
The notebook begins by loading the dataset into a pandas DataFrame and examining its structure. This includes understanding the column names, data types, and summary statistics to gain initial insights into the data.

## Libraries Used
The following libraries are used in this notebook:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- tensorflow

## Dataset
The dataset used for training and testing the model contains information about customer demographics, account details, and usage patterns. Each row represents a customer, with features as described below.

### Dataset Columns

- RowNumber: Corresponds to the record (row) number and has no effect on the output.

- CustomerId: Contains random values and has no effect on customer leaving the bank.

- Surname: The surname of a customer has no impact on their decision to leave the bank.

- CreditScore: Can have an effect on customer churn, since a customer with a higher credit score is less likely to leave the bank.

- Geography: A customer’s location can affect their decision to leave the bank. One-hot encoding is applied to this column.

- Gender: It’s interesting to explore whether gender plays a role in a customer leaving the bank. Label encoding is applied to this column.

- Age: This is certainly relevant, since older customers are less likely to leave their bank than younger ones.

- Tenure: Refers to the number of years that the customer has been a client of the bank. Normally, older clients are more loyal and less likely to leave a bank.

- Balance: Also a very good indicator of customer churn, as people with a higher balance in their accounts are less likely to leave the bank compared to those with lower balances.

- NumOfProducts: Refers to the number of products that a customer has purchased through the bank.

- HasCrCard: Denotes whether or not a customer has a credit card. This column is also relevant, since people with a credit card are less likely to leave the bank.

- IsActiveMember: Active customers are less likely to leave the bank.

- EstimatedSalary: As with balance, people with lower salaries are more likely to leave the bank compared to those with higher salaries.

- Exited: Whether or not the customer left the bank.

- Complain: Indicates whether the customer has a complaint or not.

- Satisfaction Score: Score provided by the customer for their complaint resolution.

- Card Type: Type of card held by the customer. One-hot encoding is applied to this column.

## Data Cleaning and Exploration
The notebook performs data cleaning and exploratory data analysis to understand the distribution of various features and their relationship with the target variable (customer churn).
### Label Encoding the Gender Column
### Onehot Encoding the Geography and Card Type Column
### Standard Scaling the numerical features

## Model Training
ANN model was build using Keras and Tensorflow to train and predict customer churn, including data preprocessing steps such as label encoding, onehot encoding and standard scaling.

The ANN model consists of the following layers:

- Input Layer: Accepts preprocessed features from the dataset.

- Hidden Layers: Three fully connected layers with ReLU activation.

- Output Layer: A single neuron with a sigmoid activation function to output the probability of churn.

  Optimizer: Adam and Loss Function: Binary Crossentropy

## Results
The results of the model training are evaluated to determine the performance of the predictive models.

-Accuracy: The model achieves an accuracy of 99.5% in predicting customer churn.

- Confusion Matrix: Provides insights into true positives, false positives, true negatives, and false negatives.

- Loss and Accuracy Curves: Visualize training and validation performance.

- F1 score: The model achieves an F1 score of 99.64%



## Conclusion
In this notebook, we analyzed and predicted customer churn in a bank using a dataset containing various customer attributes. Through the data analysis process, we gained valuable insights into the factors that contribute to customer churn and built an ANN model to predict churn.

By leveraging our ANN model, specifically using a Tensorflow library, we were able to predict customer churn with an Accuracy level of 99.85%. The model incorporated features such as credit score, age, balance, tenure, and other relevant customer attributes. The evaluation metrics, including accuracy, precision, recall, and F1 score, provided a comprehensive assessment of the model's performance.

## Usage
To run this notebook, ensure you have the required libraries installed and the dataset available in the specified path. Execute the cells in the notebook sequentially to reproduce the results.
