# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 09:39:55 2018

@author: deepe
"""

import pandas as pd
import numpy as np

dataset1 = pd.read_csv('Bots_fetched_data_2.csv')
dataset2 = pd.read_csv('Human_fetched_data_2.csv')

dataset1['labels'] = 1
dataset2['labels'] = 0

dataset = pd.concat([dataset1,dataset2],axis = 0)

dataset.isnull().sum()

features = dataset.iloc[:,2:13].values
labels = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
features[:,1] = labelencoder.fit_transform(features[:,1])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.2,random_state = 0)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2,random_state = 0)
features = pca.fit_transform(features)

#Naive_Bayes algorithm
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(features_train,labels_train)

labels_pred = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pred)

score = classifier.score(features_test,labels_test)

#KNN classifier algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, p = 2)
classifier.fit(features_train,labels_train)

labels_pred = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pred)

score = classifier.score(features_test,labels_test)

from sklearn.cross_validation import cross_val_score
cvs = cross_val_score(estimator = classifier,X = features_train,y = labels_train,cv = 10)
cvs.mean()
