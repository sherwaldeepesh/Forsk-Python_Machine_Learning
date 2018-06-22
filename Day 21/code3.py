# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:02:53 2018

@author: deepe
"""

import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('tshirts.csv')

features = dataset.iloc[:,[1,2]].values

from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = 'k-means++', random_state =0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_means = kmeans.fit_predict(features)

plt.scatter(features[y_means == 0,0], features[y_means == 0,1], s = 100, c= 'red', label = 'Medium')
plt.scatter(features[y_means == 1,0], features[y_means == 1,1], s = 100, c= 'blue', label = 'Large')
plt.scatter(features[y_means == 2,0], features[y_means == 2,1], s = 100, c= 'pink', label = 'Small')


plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s = 300,c = 'yellow',label = 'Centroids')
plt.title('Cluster of Size')
plt.xlabel('Height(Inches)')
plt.ylabel('Weight(Pounds)')
plt.legend()
plt.show()