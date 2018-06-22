# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 13:02:20 2018

@author: deepe
"""



#Doing Clustering of Differnet data columns





import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Pokemon.csv')

dataset.isnull().any()

features = dataset.iloc[:,5:13].values
labels = dataset.iloc[:,2].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
features[:,-1] = labelencoder.fit_transform(features[:,-1])

labels = labelencoder.fit_transform(labels)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)

import statsmodels.formula.api as sm
features = np.append(arr = np.ones((800,1)).astype(int),values = features, axis = 1)

features_opt = features[:,[0,1,2,3,4,5,6,7,8]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:,[0,1,2,3,4,5,6,7]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:,[0,1,2,3,5,6,7]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:,[0,1,2,3,5,7]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:,[0,1,2,3,7]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

from sklearn.decomposition import PCA
pca = PCA(n_components = 2, random_state = 0)
features_opt = pca.fit_transform(features_opt)

plt.scatter(features_opt[:,1],labels)


from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features_opt,labels,test_size = 0.2, random_state = 0)









"""
#Multiple Regression Model
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(features_train,labels_train)

prediction = regressor1.predict(features_test)
score = regressor1.score(features_test,labels_test)
"""


"""
#Clustering of Defense and Attack Data by K-Means
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(features)

plt.scatter(features[y_kmeans == 0,0], features[y_kmeans == 0,1], s = 100, label = 'Rating 2')
plt.scatter(features[y_kmeans == 1,0], features[y_kmeans == 1,1], s = 100, label = 'Rating 4')
plt.scatter(features[y_kmeans == 2,0], features[y_kmeans == 2,1], s = 100, label = 'Rating 3')
plt.scatter(features[y_kmeans == 3,0], features[y_kmeans == 3,1], s = 100, label = 'Rating 1')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 200, label = 'Centroid')
plt.title('Ratings')
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.legend()
plt.show()


dataset.boxplot(column="Attack",by="Generation")
dataset.boxplot(column="Defense",by="Generation")

dataset.hist(column="Attack",by=y_kmeans,bins=50)
"""

