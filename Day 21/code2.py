# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:31:56 2018

@author: deepe
"""
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('deliveryfleet.csv')

dataset = dataset[dataset['Speeding_Feature']>5]

dataset = dataset.ix[dataset['Speeding_Feature'] > 5]


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

kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 0)
y_means = kmeans.fit_predict(features)

plt.scatter(features[y_means == 0,0], features[y_means == 0,1], s = 100, c= 'red', label = 'Cluster 1')
plt.scatter(features[y_means == 1,0], features[y_means == 1,1], s = 100, c= 'blue', label = 'Cluster 2')


plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s = 300,c = 'yellow',label = 'Centroids')
plt.title('Cluster of Drivers')
plt.xlabel('Distance_feature')
plt.ylabel('Speeding_Feature')
plt.legend()
plt.show()


kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0)
y_means = kmeans.fit_predict(features)

plt.scatter(features[y_means == 0,0], features[y_means == 0,1], s = 100, c= 'red', label = 'Over Speed Urban')
plt.scatter(features[y_means == 1,0], features[y_means == 1,1], s = 100, c= 'blue', label = 'In speed Rural')

plt.scatter(features[y_means == 2,0], features[y_means == 2,1], s = 100, c= 'pink', label = 'In Speed Urban')
plt.scatter(features[y_means == 3,0], features[y_means == 3,1], s = 100, c= 'green', label = 'Over Speed Rural')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s = 300,c = 'yellow',label = 'Centroids')
plt.title('Cluster of Drivers')
plt.xlabel('Distance_feature')
plt.ylabel('Speeding_Feature')
plt.legend()
plt.show()
