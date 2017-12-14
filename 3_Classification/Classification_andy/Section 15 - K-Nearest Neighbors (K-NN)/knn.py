# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#
# Preprocessing
#

# Importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# training and test set split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, 
                                                    test_size = 0.25, 
                                                    random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


#
# Classifier 
#

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit( X_train, y_train)

#
# Predictions & performance
#

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# 
# Visualising results 
#

def two_feature_decision_map( X1, X2, y, classifier, X12_labels, y_labels, y_colors, title): 

    """ Plots a 2D feature space showing the decision boundary of a trained 
    binary classifier, along with datapoints coloured according to class (y).
    
    Args:
        X1 - numpy array of values for one feature used to train a binary classifier.       
        
            E.g. 
            
                import pandas as pd
                dataset = pd.read_csv("Social_Network_Ads.csv")
                X = dataset.iloc[:, [2, 3]].values
                X1 = X[:, 0]
        
        X2 - numpy array of values for a second feature used to train the classifier.
        
            E.g. 
            
                import pandas as pd
                dataset = pd.read_csv("Social_Network_Ads.csv")
                X = dataset.iloc[:, [2, 3]].values
                X1 = X[:, 1]
        
        y - list of values representing known class outputs for X1 and X2 data.
        classifier - binary classifier model object trained on X1, X2, y. 
            
            E.g.
                
                from sklearn.neighbors import KNeighborsClassifier
                classifier_obj = KNeighborsClassifier()
                classifier = KNeighborsClassifier.fit_transform([X1,X2], y)
        
        X12_labels - list of two strings, 1st to label the Y-axis describing X1, 
                     and the 2nd to label the X-axis describing X2.
        
            E.g. ["Salary", "Age"]
        
        y_labels - list of two strings, 1st describes one of the two classes in 
                    y, and 2nd describes the other class.
                    
            E.g. ["Not purchased","Purchased"]
        
        y_colors - list of colours corresponding to the output classes in y_labels.
        
            E.g. ["Red", "Green"]
            
        title - the plot title.
    """
    X1_grid, X2_grid = np.meshgrid( np.arange(min(X1)-1, max(X1)+1, step=0.01),
                                    np.arange(min(X2)-1, max(X2)+1,step=0.01))

    ygrid_pred = classifier.predict(np.array([X1_grid.ravel(),X2_grid.ravel()]).T)    
    
    from matplotlib.colors import ListedColormap
    
    colors = ListedColormap(y_colors)
    labels = y_labels
    
    for i in np.unique(y):
        plt.scatter( X1[y == i], X2[y == i], c=colors(i), label=labels[i])
        
    plt.contourf( X1_grid, X2_grid, ygrid_pred.reshape(X1_grid.shape), 
                 cmap=colors,
                 alpha=0.35)
        
    plt.legend()
    plt.title(title)
    plt.xlabel(X12_labels[0])
    plt.ylabel(X12_labels[1])

# Run the function to plot data from training set 

two_feature_decision_map( 
        X_test[:, 0], 
        X_test[:, 1], 
        y_test,
        classifier,
        X12_labels = ["Salary", "Age"],
        y_labels = ["Not purchased","Purchased"],
        y_colors = ["red", "green"],
        title = "KNN Classifier (training set)"
        ) 

