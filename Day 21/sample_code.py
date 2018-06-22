# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 05:12:24 2018

@author: deepe
"""
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')

features = dataset.iloc[:,[-2,-1]].values

from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(features)

plt.scatter(features[y_kmeans == 0,0],features[y_kmeans == 0,1], s = 100, c = 'red', label = 'cluster1')
plt.scatter(features[y_kmeans == 1,0],features[y_kmeans == 1,1], s = 100, c = 'blue', label = 'cluster2')
plt.scatter(features[y_kmeans == 2,0],features[y_kmeans == 2,1], s = 100, c = 'pink', label = 'cluster3')
plt.scatter(features[y_kmeans == 3,0],features[y_kmeans == 3,1], s = 100, c = 'cyan', label = 'cluster4')
plt.scatter(features[y_kmeans == 4,0],features[y_kmeans == 4,1], s = 100, c = 'green', label = 'cluster5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 200, c = 'magenta', label = 'Centroid')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()