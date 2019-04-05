# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:41:34 2019

@author: Vivek
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Position_salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Fitting Decision Tree to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, Y)

#Predicting a new result
Y_pred = regressor.predict([[6.5]])

# Visualising the Decision Tree Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.0001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.xlabel('Position Level of an Employee')
plt.ylabel('Salary')
plt.title('Truth or Bluff (Decision Tree)')
plt.show()