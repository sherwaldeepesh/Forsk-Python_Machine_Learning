# -*- coding: utf-8 -*-
"""
Created on Mon Jun 04 11:18:52 2018

@author: deepe
"""
import numpy as np
import pandas as pd
dataset = pd.read_csv('bluegills.csv')

features = dataset.iloc[:,0:1].values
labels = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split

features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.1,random_state = 0)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(features_train,labels_train)

pred1 = lin_reg.predict(features_test)

import matplotlib.pyplot as plt

plt.scatter(features,labels)
plt.plot(features_train,lin_reg.predict(features_train))
plt.title('Age vs length of Fish')
plt.xlabel('Age of Fish')
plt.ylabel('Length of the Fish')
plt.show()


from sklearn.preprocessing import PolynomialFeatures

poln_object = PolynomialFeatures(degree = 155)
features_poln = poln_object.fit_transform(features)

features_train1,features_test1,labels_train1,labels_test1 = train_test_split(features_poln,labels,test_size = 0.1,random_state = 0)


lin_reg_1 = LinearRegression()
lin_reg_1.fit(features_train1,labels_train1)

pred1 = lin_reg_1.predict(features_test1)

plt.scatter(features,labels)
plt.plot(features_train,lin_reg_1.predict(features_train1))
plt.title('Age vs length of Fish')
plt.xlabel('Age of Fish')
plt.ylabel('Length of the Fish')
plt.show()

features_grid = np.arange(min(features),max(features),0.1)
features_grid = features_grid.reshape((-1,1))
plt.scatter(features,labels)
plt.plot(features_grid,lin_reg_1.predict(poln_object.fit_transform(features_grid)))
plt.title('Age vs length of Fish')
plt.xlabel('Age of Fish')
plt.ylabel('Length of the Fish')
plt.show()


