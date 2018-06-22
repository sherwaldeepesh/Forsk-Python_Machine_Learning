# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:41:10 2018

@author: deepe
"""
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Bahubali2_vs_Dangal.csv')

features = df.iloc[:,:1].values
labels = df.iloc[:,1:].values

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.2,random_state =0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(features_train,labels_train)

predict_data = regressor.predict(features_test)

plt.scatter(features_train,labels_train[:,0],color = 'green',label = 'Bahubali 2')
plt.scatter(features_train,labels_train[:,1],color = 'orange',label = 'Dangal')
plt.plot(features_train,regressor.predict(features_train)[:,0],color = 'green')
plt.plot(features_train,regressor.predict(features_train)[:,1],color = 'orange')
plt.legend()
plt.title("Day vs Movies (Training Set)")
plt.xlabel("Day")
plt.ylabel('Movies')
plt.show()

prediction_score = regressor.score(features_test,labels_test)

new_prediction = regressor.predict(10)

if new_prediction[0][0] > new_prediction[0][1]:
    print '\n\n\n\tBahubali_2'
else:
    print '\n\n\n\tDangal'