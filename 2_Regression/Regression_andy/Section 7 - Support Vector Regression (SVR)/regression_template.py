# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#
# Import the dataset
#

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#
# Regression model
#

# linear
from sklearn.linear_model import LinearRegression
reg_sim = LinearRegression()
reg_sim.fit(X, y)

# Polynomial
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

pol = PolynomialFeatures(degree = 4)
X_pol = pol.fit_transform(X)

reg_pol = LinearRegression()
reg_pol.fit(X_pol, y)


#
# Making a prediction
#

# Linear
reg_sim.predict(6.5)

# Polynomial
reg_pol.predict(pol.fit_transform(6.5))


#
# Plotting
#

# smoothing the predictions

X_smooth = np.arange(min(X), max(X), step=0.01)

plt.scatter(X, y, color="r")
plt.plot(X, reg_sim.predict(X), color="b")
plt.plot(X, reg_pol.predict(X_pol), color="g")
plt.title("Simple vs. Polynomial Regression Models")
plt.xlabel("Rank")
plt.ylabel("Salary")

