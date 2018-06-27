# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 20:12:40 2018

@author: deepe
"""

import pandas as pd
dataset1 = pd.read_csv('movie_metadata.csv')

dataset.isnull().sum()

dataset = dataset1.drop(['color','aspect_ratio','budget','gross','plot_keywords','content_rating','director_name','director_facebook_likes'],axis = 1)

dataset = dataset.fillna(dataset.mean())

dataset = dataset.dropna()


genres_unique = pd.DataFrame(dataset.genres.str.split('|').tolist()).stack().unique()
genres_unique = pd.DataFrame(genres_unique, columns=['genres']) # Format into DataFrame to store later
dataset = dataset.join(dataset.genres.str.get_dummies().astype(bool))
dataset.drop('genres', inplace=True, axis=1)

dataset2 = dataset.drop(['imdb_score','movie_title','movie_imdb_link','language','country','movie_facebook_likes'],axis = 1)

features = dataset2.iloc[:,0:].values
labels = dataset.iloc[:,16].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

features[:,3] = labelencoder.fit_transform(features[:,3])
features[:,5] = labelencoder.fit_transform(features[:,5])
features[:,8] = labelencoder.fit_transform(features[:,8])

for i in range(13,37):
    features[:,i] = labelencoder.fit_transform(features[:,i])
    
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(features_train,labels_train)
regressor.score(features_test,labels_test)


from sklearn.ensemble import RandomForestRegressor
ran_for_regressor = RandomForestRegressor(n_estimators = 10,random_state = 0)
ran_for_regressor.fit(features_train,labels_train)
pred1 = ran_for_regressor.predict(features_test)
score = ran_for_regressor.score(features_test,labels_test)    