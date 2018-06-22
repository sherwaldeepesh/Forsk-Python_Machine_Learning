# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 12:40:09 2018

@author: deepe
"""

import pandas as pd
dataset = pd.read_csv('banknotes.csv')

features = dataset.iloc[:,1:6].values
labels = dataset.iloc[:,-1].values
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2,random_state = 0)
features = pca.fit_transform(features)

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.2, random_state = 0)


"""
#Using LogisiticRegression Algorithm
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(features_train,labels_train)

prediction = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,prediction)

score = classifier.score(features_test,labels_test)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = features_train, y = labels_train, cv = 10)
print ("Mean Accuracy is",accuracies.mean())
print (accuracies.std())
"""

"""
# Using KNeighborsClassifier Algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(features_train,labels_train)

prediction = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,prediction)

score = classifier.score(features_test,labels_test)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = features_train, y = labels_train, cv = 10)
print ("Mean Accuracy is",accuracies.mean())
print (accuracies.std())
"""

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = features_train, labels_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN (Training set)')

plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = features_test, labels_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN (Test set)')

plt.legend()
plt.show()