# -*- coding: utf-8 -*-
"""
Created on Mon Jun 04 12:11:24 2018

@author: deepe
"""

import numpy as np
import pandas as pd
dataset = pd.read_csv('bluegills.csv')

features = dataset.iloc[:,0:1].values
labels = dataset.iloc[:,1].values

from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt

lin_reg = LinearRegression()
lin_reg.fit(features,labels)

pred1 = lin_reg.predict(features)

plt.scatter(features,labels)
plt.plot(features,lin_reg.predict(features))
plt.title('Age vs length of Fish')
plt.xlabel('Age of Fish')
plt.ylabel('Length of the Fish')
plt.show()

from sklearn.preprocessing import PolynomialFeatures

poln_object = PolynomialFeatures(degree = 155)
features_poln = poln_object.fit_transform(features)

lin_reg_1 = LinearRegression()
lin_reg_1.fit(features_poln,labels)

pred1 = lin_reg_1.predict(poln_object.fit_transform(3.5))

plt.scatter(features,labels)
plt.plot(features,lin_reg_1.predict(features_poln))
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

