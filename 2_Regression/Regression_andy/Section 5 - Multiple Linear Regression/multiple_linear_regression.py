# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical variables

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform( X[:, 3])
onehotencoder = OneHotEncoder( categorical_features = [3] )
X = onehotencoder.fit_transform( X ).toarray()

# Dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


# Multiple linear regression
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor = regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# Feature selection by Backwards elimination

import statsmodels.formula.api as sm

# incorporate the intercept (B_0) column
X = np.append(arr=np.ones((len(X),1)).astype(int), values=X, axis=1) #..
#.. axis = 0: add a row, axis = 1: add a column

features = [0,1,2,3,4,5]

def backwards_eliminator(features, X):
    """ Recursively generates a multiple linear regression model, each time..
    ..eliminating a feature w/ P-value > 0.05, until a regression model with..
    ..all features having P-value < 0.05 is made.
    
    Args:
        - features (list): list of indices, that are columns of X to be..
            ..recursively removed by backwards elimination. 
        - X (np.ndarray): feature data
    Returns:
        - features (list): features who when included in multiple regression..
            ..all have P-value < 0.05
            
        - regressor_OLS (statsmodels.regression.linear_model.RegressionResultsWrapper):
            ..regression model object with backwards eliminated  features.
    """
    X_opt = X[:, features]
    regressor_OLS = sm.OLS( endog=y, exog=X_opt).fit()
    p_values = regressor_OLS.pvalues
    for i,p_value in enumerate(p_values):
        print("before",i,p_value,features,regressor_OLS.pvalues)
        if p_value>0.05:
            features.remove(features[i])
            print("after",i,p_value,features,regressor_OLS.pvalues)
            return backwards_eliminator(features, X)
    return features, regressor_OLS
        
backwards_eliminator(features)

#
#X_opt = X[:, [0, 3]]
#regressor_OLS = sm.OLS( endog=y, exog=X_opt).fit()
#regressor_OLS.pvalues


