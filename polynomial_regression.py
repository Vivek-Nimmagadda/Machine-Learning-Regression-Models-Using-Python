# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:41:34 2019

@author: Vivek
"""

# Polynomial Regression
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# Visualising the Polynomial Regression and Linear Regression together
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.plot(X, lin_reg.predict(X), color = 'violet')
plt.xlabel('Position Level of an Employee')
plt.ylabel('Salary')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.show()