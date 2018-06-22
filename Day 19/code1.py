# -*- coding: utf-8 -*-
"""
Created on Wed Jun 06 11:01:57 2018

@author: deepe
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('affairs.csv')
features = dataset.iloc[:,:-1].values
labels = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()

for i in range(6,8):
    features[:,i] = labelencoder.fit_transform(features[:,i])
    
onehotencoder1 = OneHotEncoder(categorical_features = [6])
features = onehotencoder1.fit_transform(features).toarray()
features = features[:,1:]

onehotencoder1 = OneHotEncoder(categorical_features = [-1])
features = onehotencoder1.fit_transform(features).toarray()
features = features[:,1:]

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.25, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(features_train,labels_train)

labels_pred = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pred)

score = classifier.score(features_test,labels_test)

new_pred = classifier.predict(np.array([1,0,0,0,0,0,0,1,0,0,3,25,3,1,4,16]).reshape(1,-1))

