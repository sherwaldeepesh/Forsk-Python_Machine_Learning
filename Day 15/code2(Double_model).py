# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:41:10 2018

@author: deepe
"""
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Bahubali2_vs_Dangal.csv')

features = df.iloc[:,:1].values
labels1 = df.iloc[:,1:2].values
labels2 = df.iloc[:,2:3].values

from sklearn.model_selection import train_test_split
features_train1,features_test1,labels_train1,labels_test1 = train_test_split(features,labels1,test_size = 0.2,random_state =0)

features_train2,features_test2,labels_train2,labels_test2 = train_test_split(features,labels2,test_size = 0.2,random_state =0)

from sklearn.linear_model import LinearRegression

regressor1 = LinearRegression()
regressor2 = LinearRegression()

regressor1.fit(features_train1,labels_train1)
predict_data1 = regressor1.predict(features_test1)

prediction_score1 = regressor1.score(features_test1,labels_test1)

regressor2.fit(features_train2,labels_train2)
predict_data2 = regressor2.predict(features_test2)

prediction_score2 = regressor2.score(features_test2,labels_test2)

plt.plot(features,labels1,color = 'red')
plt.scatter(features_train1,labels_train1,color = 'red',label = 'Bahubali 2')
plt.plot(features_train1,regressor1.predict(features_train1),color = 'red')
plt.plot(features,labels2,color = 'magenta')
plt.scatter(features_train2,labels_train2,color = 'magenta', label = 'Dangal')
plt.plot(features_train2,regressor2.predict(features_train2),color = 'magenta')
plt.legend()
plt.show()

new_prediction1 = regressor1.predict(10)
new_prediction2 = regressor2.predict(10)

if new_prediction1[0][0]>new_prediction2[0][0]:
    print "\n\n\n\tBahubali 2"
else:
    print "\n\n\n\tDangal"
