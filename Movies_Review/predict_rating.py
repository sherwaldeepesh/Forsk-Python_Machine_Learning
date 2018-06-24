# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 14:15:03 2018

@author: deepe
"""

import pandas as pd

ratings = pd.read_table('ratings.dat',sep='::',names=['user_id','movie_id','rating','time'])
users = pd.read_table('users.dat',sep='::',names=['user_id','gender','age','occupation','zip'])
movies = pd.read_table('movies.dat',sep='::',names=['movie_id','title','genre'])

movielens_dataset = pd.merge(pd.merge(ratings,users),movies)

"""
age_rating = movielens_dataset.groupby('age',as_index = False)['rating'].mean()

users_rating = movielens_dataset.groupby(['title',movielens_dataset.groupby('user_id', as_index=False)['rating'].mean()])

movies_rating = movielens_dataset.groupby(['genre','movie_id'], as_index = False)['rating'].mean()
"""

ran = movielens_dataset.groupby(['age','occupation','gender','user_id'],as_index = False)['rating'].mean()

ran = ran[['user_id','gender','age','occupation','rating']]

typegr = movielens_dataset.groupby('genre',as_index=False)['rating'].mean()

typeuser = movielens_dataset.groupby(['user_id','gender','age','occupation','zip','genre'],as_index=False)['rating'].mean()

movielens_dataset.dtypes









"""
Convert Timestamp to datetime
pd.to_datetime(df['mydates']).apply(lambda x: x.datetime())

"""

features = movielens_dataset.iloc[:,[0,1,4,5,6,7,9]].values
labels = movielens_dataset.iloc[:,2].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

features[:,2] = labelencoder.fit_transform(features[:,2])
features[:,-1]  = labelencoder.fit_transform(features[:,-1])
features[:,-2] = labelencoder.fit_transform(features[:,-2])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

features = scaler.fit_transform(features)



from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.2,random_state = 0)



"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train,labels_train)
pred = regressor.predict(features_test)
score = regressor.score(features_test,labels_test)

from sklearn.tree import DecisionTreeRegressor

regressor1 = DecisionTreeRegressor(random_state = 0)

regressor1.fit(features_train,labels_train)

regressor1.predict(features_test)

regressor1.score(features_test,labels_test)
"""

"""
from sklearn.tree import DecisionTreeRegressor
regressor5 = DecisionTreeRegressor(random_state = 0)
regressor5.fit(features_train,labels_train)

score = regressor5.score(features_test,labels_test)
"""

from sklearn.ensemble import RandomForestRegressor
ran_for_regressor = RandomForestRegressor(n_estimators = 10,random_state = 0)
ran_for_regressor.fit(features_train,labels_train)
score = ran_for_regressor.score(features_test,labels_test)