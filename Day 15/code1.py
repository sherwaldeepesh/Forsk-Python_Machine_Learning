# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:06:42 2018

@author: deepe
"""
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Foodtruck.csv")

df.isnull().any()

features = df.iloc[:,:-1].values
labels = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split

features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.1, random_state = 0)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(features_train,labels_train)

predict_data = regressor.predict(features_test)

new_prediction = regressor.predict(3.073)

#to visualize Train Set Data
plt.scatter(features_train,labels_train,color = 'red')
plt.plot(features_train,regressor.predict(features_train),color = 'blue')
plt.title("Population vs Profit (Training Set)")
plt.xlabel("Population")
plt.ylabel('Profit')
plt.show()

#to Visualize test set data
plt.scatter(features_test,labels_test,color = 'red')
plt.plot(features_train,regressor.predict(features_train),color = 'blue')
plt.plot("Population vs Profit (Test Set)")
plt.xlabel("Population")
plt.ylabel('Profit')
plt.show()


predict_score = regressor.score(features_test,labels_test)

