# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 17:17:21 2019

@author: Vivek
"""
# Predict the salaries of various employees based on their experience using Simple-Linear Regression
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test set results
Y_pred = regressor.predict(X_test)

# Visualising the training set results
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Experience in years')
plt.ylabel('Salary')
plt.show()

# Visualising the test set results
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, Y_pred, color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Experience in years')
plt.ylabel('Salary')
plt.show()