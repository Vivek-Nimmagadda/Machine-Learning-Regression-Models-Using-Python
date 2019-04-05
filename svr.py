# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:32:55 2019

@author: Vivek
"""
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X, Y)

#Predicting a new result
Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the SVR
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.xlabel('Position Level of an Employee')
plt.ylabel('Salary')
plt.title('Truth or Bluff (SVR)')
plt.show()