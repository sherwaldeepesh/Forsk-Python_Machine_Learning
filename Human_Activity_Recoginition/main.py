# -*- coding: utf-8 -*-
"""
Created on Tue Sep 04 18:21:27 2018

@author: deepe
"""

import pandas as pd
dataset = pd.read_csv('train.csv')

features = dataset.iloc[:,:-2].values
labels = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

labels = labelencoder.fit_transform(labels)

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.2,random_state = 0)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)

classifier.fit(features_train,labels_train)

labels_pred = classifier.predict(features_test)

score_logistic = classifier.score(features_test,labels_test)

#KNN Approach
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5,p=2)
classifier_knn.fit(features_train,labels_train)

labels_pred_knn = classifier_knn.predict(features_test)

score_knn = classifier_knn.score(features_test,labels_test)

#Tree Approach Classifier
from sklearn.tree import DecisionTreeClassifier
classifier_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_tree.fit(features_train,labels_train)

lables_pred_tree = classifier_tree.predict(features_test)

score_tree = classifier_tree.score(features_test,labels_test)

#RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier_random_forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_random_forest.fit(features_train,labels_train)

labels_pred_random_forest = classifier_random_forest.predict(features_test)

score_random_forest = classifier_random_forest.score(features_test,labels_test)