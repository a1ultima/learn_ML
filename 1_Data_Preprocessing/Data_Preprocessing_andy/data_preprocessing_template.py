# -*- coding: utf-8 -*-
"""
Udemy course module on Data_Processing




"""

# IMPORTS 
import numpy as np

#np.set_printoptions(threshold='NaN') # this allows to print entire array

import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import Imputer



# DATA INPUT
dataset = pd.read_csv("./Data.csv")

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values # upperbound is ommited, i.e. takes only col 1, 2

# Preprocessing: missing values imputed

# alternative 1 {{
#
# One step 
#
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
X[:, 1:3] = imputer.fit_transform(X[:, 1:3]) # i.e. col 2 and 3
#
# }} 1 alternative 2 {{ 
#
# Two steps
#        
# imputer = imputer.fit(X[:, 1:3])
#X[:, 1:3] = imputer.transform(X[:, 1:3]) # X.tolist() will print whole array        
## }} alternative 2
#

X_printable = X.tolist()  # so one can view the full array
import pprint
pprint.pprint(X_printable)

#
# Encode categorical variables
#
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # strings to numbers,'France'=0

# there are >2 categories, but encoding them as 0, 1, 2 is wrong, it's better
#..to encode each category as binary in three separate columns, thus:
onehotencoder = OneHotEncoder(categorical_features = [0]) 
X = onehotencoder.fit_transform(X).toarray()

# unlike X, Y is already binary, no need for onehotencoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#
# Split data into training and test set
#
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,\
                                                    random_state=0)

#
# Scale the features: most libraries do this for u
#

from sklearn.preprocessing import StandardScaler

# we scale the training and test set, but not the output class since binary

sc_X = StandardScaler()  # subtracts mean then divide by std

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)



