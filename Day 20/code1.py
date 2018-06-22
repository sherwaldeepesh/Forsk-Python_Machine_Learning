# -*- coding: utf-8 -*-
"""
Created on Thu Jun 07 10:49:33 2018

@author: deepe
"""

import pandas as pd
dataset= pd.read_csv('mushrooms.csv')

dataset.isnull().any()

features = dataset.iloc[:,[5,-2,-1]].values
labels = dataset.iloc[:,0].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

features[:,0] = labelencoder.fit_transform(features[:,0])
features[:,1] = labelencoder.fit_transform(features[:,1])
features[:,2] = labelencoder.fit_transform(features[:,2])

labels = labelencoder.fit_transform(labels)

from sklearn.preprocessing import OneHotEncoder
onehot1 = OneHotEncoder(categorical_features = [0])
features = onehot1.fit_transform(features).toarray()
features = features[:,1:]

onehot2 = OneHotEncoder(categorical_features = [-2])
features = onehot2.fit_transform(features).toarray()
features = features[:,1:]

onehot3 = OneHotEncoder(categorical_features = [-1])
features = onehot3.fit_transform(features).toarray()
features = features[:,1:]

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.25,random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,p=2)

classifier.fit(features_train,labels_train)

pred = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, pred)

score = classifier.score(features_test,labels_test)

