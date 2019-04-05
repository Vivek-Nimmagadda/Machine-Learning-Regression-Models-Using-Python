# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 09:51:22 2019

@author: Vivek
"""

#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing Dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:].values

#Fitting Random Forest Regressor to the Dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X, Y)

#Predicting the Salary of Employee with a Position Level of 6.5
Y_pred = regressor.predict([[6.5]])

#Visualising the Random Forest Regression Results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.0001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='green')
plt.xlabel("Position Level of an Employee")
plt.ylabel("Salary")
plt.title("Truth or Bluff(Random Forest)")
plt.show() 