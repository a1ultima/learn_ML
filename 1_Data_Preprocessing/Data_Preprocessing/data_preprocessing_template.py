# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Split into training vs. test set
from sklearn.model_selection import train_test_split
X_test, X_train, y_test, y_train = train_test_split(X, y, random_state = 0)

# Apply feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test  = sc.fit_transform(X_test)

# Train the logistic regression classifier
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit_transform(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)

# Visualise 
from matplotlib.colors import ListedColormap

colors = ListedColormap(["red","green"])

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid( np.arange(min(X_set[:, 0]), max(X_set[:, 0]), step=0.01), 
                      np.arange(min(X_set[:, 1]), max(X_set[:, 1]), step=0.01))

for i,j in enumerate(y_set):
    
    plt.scatter(X_train[i, 0], X_train[i, 1], c=colors(j), label=j)

plt.legend()

