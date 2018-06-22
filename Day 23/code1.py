# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 11:09:56 2018

@author: deepe
"""
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('crime_data.csv')
features1 = dataset.iloc[:,0:].values
features = dataset.iloc[:,[1,2,4]].values

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
features = pca.fit_transform(features)
explained_variance = pca.explained_variance_ratio_


from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(features)

plt.scatter(features[y_kmeans == 0,0],features[y_kmeans == 0,1], s = 100, label = 'Murder')
plt.scatter(features[y_kmeans == 1,0],features[y_kmeans == 1,1], s = 100, label = 'Assault')
plt.scatter(features[y_kmeans == 2,0],features[y_kmeans == 2,1], s = 100, label = 'Rape')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 200, label = 'Centroid')
plt.title('Grouped Cities')

plt.legend(loc = 1)
plt.show()


data=[]
for i in range(50):
    if y_kmeans[i] == 0:
        data.append(dataset.iloc[i,0])
dat1 = pd.DataFrame(data , columns = ['State having more Murder cases'])


data1=[]
for i in range(50):
    if y_kmeans[i] == 1:
        data1.append(dataset.iloc[i,0])
dat2 = pd.DataFrame(data1 , columns = ['State having more Assault cases'])


data2=[]
for i in range(50):
    if y_kmeans[i] == 2:
        data2.append(dataset.iloc[i,0])
dat3 = pd.DataFrame(data2 , columns = ['State having more Rape cases'])

