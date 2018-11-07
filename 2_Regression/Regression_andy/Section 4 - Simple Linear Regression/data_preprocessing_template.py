# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3., random_state = 0)


# Feature Scaling: 
#   in simple linear regression the library does it for us already
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Make predictions of Y, u/ Regression model
y_pred = regressor.predict(X_test)

#
# Plot results
#

# Training set
plt.scatter(X_train, y_train, color="r")
# plt.scatter(X_test, y_pred, color="g")  # plot the predicted Y for test set
plt.plot(X_train, regressor.predict(X_train), color="b")
plt.title("Experience Vs Salary (training set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
#plt.close()

# Test set
plt.scatter(X_test, y_test, color="r")
plt.plot(X_train, regressor.predict(X_train), color="b") # we want to see how\
# ..the trained regression line looks in contrast to the test set data 
plt.title("Experience vs. Salary (test set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()