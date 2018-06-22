# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:20:03 2018

@author: deepe
"""

import pandas as pd

dataset = pd.read_csv('tree_addhealth.csv')
dataset = dataset.apply(lambda x:x.fillna(x.mode()[0]))

#Code Challenge Part 1
features = dataset.iloc[:,:16]
features = features.drop(['TREG1'], axis = 1)

features = features.iloc[:,:].values

labels = dataset.iloc[:,7].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.25, random_state = 0)

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(features_train,labels_train)

pred = classifier.predict(features_test)

predScore = classifier.score(features_test,labels_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,pred)
#Code Challenge Part 2
features1 = dataset.iloc[:,[0,17]].values
labels1 = dataset.iloc[:,21].values

"""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features1 = scaler.fit_transform(features1)
"""

from sklearn.model_selection import train_test_split
features_train1,features_test1,labels_train1,labels_test1 = train_test_split(features1,labels1,test_size = 0.2, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
classifier1 = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)

classifier1.fit(features_train1,labels_train1)

pred1 = classifier1.predict(features_test1)

predScore1 = classifier1.score(features_test1,labels_test1)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(labels_test1,pred1)

#Code Challange Part 3

features2 = dataset.iloc[:,1:5].values
labels2 = dataset.iloc[:,7].values

from sklearn.model_selection import train_test_split
features_train2,features_test2,labels_train2,labels_test2 = train_test_split(features2,labels2,test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators =10, criterion = 'entropy',random_state = 0)

classifier2.fit(features_train2,labels_train2)

pred2 = classifier2.predict(features_test2)

predScore2 = classifier2.score(features_test2,labels_test2)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(labels_test2,pred2)
