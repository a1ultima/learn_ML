# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
#X = dataset.iloc[:, 1].values # this will work, but then it is a vector, not..
#..a matrix. In general, it is good practice to have the y as vec and x as mat
X = dataset.iloc[:, 1:2].values # upper bound 2 ignored, but makes X a matrix
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


##
## Comparing simple vs. polynomial regressors
##


#
# Linear regression model
#

from sklearn.linear_model import LinearRegression

linreg_sim = LinearRegression()
linreg_sim.fit(X, y)

#
# Polynomial regression models
#

from sklearn.preprocessing import PolynomialFeatures

# Preprocess the feature vector to include higher order terms b0+b0*X+b0*X^2+...

poly = PolynomialFeatures(degree=4) # degree 4, i.e. parabolic relationship
X_poly = poly.fit_transform(X) # note: i=0 is b0 intercept, i=2 is x^2 order term

linreg_pol = LinearRegression()
linreg_pol.fit(X_poly, y)

##
## Visualise: comparison of Linear vs. Polynomial regression
##

plt.scatter(X, y, color="red")
plt.plot(X, linreg_sim.predict(X), color="blue")
plt.plot(X, linreg_pol.predict(X_poly), color="green")

plt.title("Comparing polynomial vs. simple regression models")

plt.ylabel("Salary (USD)")
plt.xlabel("Company Rank")

plt.show()




