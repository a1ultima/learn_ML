# Logistic Regression

#
# Importing the libraries
#
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#
# Import the Dataset
#
dataset = pd.read_csv("Social_Network_Ads.csv")

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

#
# Split into training vs. test sets
#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.25, 
                                                     random_state = 0 )
#
# Feature scaling
#
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#
# Train Logistic Regression
#
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#
# Predict results
#
y_pred = classifier.predict(X_test)

#
# Performance Statistics
#
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#
# Visualisation of Performance
#
from matplotlib.colors import ListedColormap

colors = ListedColormap(["red","green"])
labels = ["Purchased'", "Purchased"]

X_set, y_set = X_train, y_train

X_age, X_salary = np.meshgrid( 
        np.arange(min(X_set[:, 0])-1, max(X_set[:, 0])+1, step = 0.01), 
        np.arange(min(X_set[:, 1])-1, max(X_set[:, 1])+1, step = 0.01) ) #..
        #..obtain two coordinate matrices, a pair of elements points to a region..
        #..in the age-salary feature space

X_plane = np.array([X_age.ravel(), X_salary.ravel()]).T # flatten the coordinate..
#..matrices, .ravel(), and transpose, .T, to obtain pairs of coordinates in..
#..feature space, so we can..:

y_plane = classifier.predict(X_plane) # ..predict Y-values corresponding to X features

# Draw decision boundary
plt.contourf(X_age, X_salary, 
             y_plane.reshape(X_age.shape), # "un-ravel" the y_plane
             alpha=0.5, 
             cmap=colors)

# Plot data points in feature space, colour according to Y-class (red, green)
for j in np.unique(y_set):
    plt.scatter( X_train[y_set==j, 0], X_train[y_set==j, 1], # all rows where y={0,1}
                 c=colors(j),      
                 label=labels[j])  # label for making a legend
plt.legend()
plt.title("Logistic Regression-based classification of training data")
plt.xlabel("Age")
plt.ylabel("Salary")

