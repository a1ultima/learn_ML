# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, 3].values

# Using the elbow method to find optimal number of clusters

from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):
    kmeans = KMeans( n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)  
plt.title("Within Cluster Sum of Squares (WCSS) vs. K clusters")
plt.ylabel("WCSS")
plt.xlabel("K number of Clusters")
plt.close()

# Plot using the chosen k clusters

k=5 # the k clusters decided from the elbow analysis

kmeans = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=0)

ymeans = kmeans.fit_predict(X)

cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0, 1, k))

labels = ["Cluster "+str(i) for i in range(0,k+1)]

for i in np.unique(ymeans):
    
    plt.scatter(X[ymeans==i, 0], X[ymeans==i, 1], s=100, c=colors[i], label=labels[i], edgecolors="black")

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c="yellow", label="centroids", edgecolors="black")    

plt.title("K-means clustering of customers (k=5)")
plt.ylabel("Customer Score")
plt.xlabel("Salary")
plt.legend()