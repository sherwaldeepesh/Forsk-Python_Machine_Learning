# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 12:10:40 2018

@author: deepe
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris()
iris=iris.data

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit_transform(iris)

explained_variance = pca.explained_variance_ratio_

from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(iris)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(iris)

plt.scatter(iris[y_kmeans == 0,0],iris[y_kmeans == 0,1], s = 100, label = 'Iris_setosa')
plt.scatter(iris[y_kmeans == 1,0],iris[y_kmeans == 1,1], s = 100, label = 'Iris_versicolor')
plt.scatter(iris[y_kmeans == 2,0],iris[y_kmeans == 2,1], s = 100, label = 'Iris_virginica')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 200, label = 'Centroid')
plt.title('Variance in Species')

plt.legend(loc = 1)
plt.show()

