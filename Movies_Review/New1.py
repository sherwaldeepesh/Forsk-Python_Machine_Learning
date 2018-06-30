# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:05:18 2018

@author: deepe
"""

import numpy as np
import pandas as pd 
movie=pd.read_csv('movie_metadata.csv')
movie=movie.drop(['color','duration','language','country','plot_keywords','aspect_ratio','budget','facenumber_in_poster'],1)
movie.isnull().values.any(0)
movie.isnull().sum()
movie['director_name']=movie['director_name'].fillna(movie['director_name'].mode()[0])
movie = movie.dropna(subset =['actor_3_facebook_likes','content_rating','num_critic_for_reviews','gross','actor_1_facebook_likes','actor_2_name','actor_1_facebook_likes','actor_1_name','actor_3_name','num_user_for_reviews','title_year','actor_2_facebook_likes'])

genres_unique = pd.DataFrame(movie.genres.str.split('|').tolist()).stack().unique()
genres_unique = pd.DataFrame(genres_unique, columns=['genres']) # Format into DataFrame to store later
movie = movie.join(movie.genres.str.get_dummies().astype(int))
movie.drop('genres', inplace=True, axis=1)


avs = movie.drop('imdb_score',axis = 1)

features = avs.iloc[:,0:]
labels = movie.iloc[:,17]

movielens_dataset = pd.concat([features,labels],axis = 1)


movielens_dataset.drop(movielens_dataset.columns[[1,2,3,5,6,8,9,10,12,13,14,15,16,17]],axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

for i in movielens_dataset:
    if movielens_dataset[i].dtypes == object:
        movielens_dataset[i] = labelencoder.fit_transform(movielens_dataset[i])


from sklearn.metrics.pairwise import cosine_similarity

#Movies_avg_Rating
movies_rating = movielens_dataset
    
movies_rating1 = movies_rating.iloc[20:500,:-1]    

#New Movie data inserts


data1 = movies_rating.iloc[20:500,:-1]    
  
data2 = movies_rating.iloc[20:500,-1]    

   
nk = []        
for i in range(len(data1)):
    
    #append index matches with new data   
    cbv = []
    vcb = []
    for item in range(len(movies_rating1)):
        if cosine_similarity(movies_rating1.iloc[item].values.reshape(1,-1),data1.iloc[i].values.reshape(1,-1))>0.5:
            cbv.append(item)
            vcb.append(cosine_similarity(movies_rating1.iloc[item].values.reshape(1,-1),data1.iloc[i].values.reshape(1,-1))[0][0])
    
    acbv = pd.DataFrame(np.column_stack([cbv,vcb]),columns = ['Index_same_values','Cosine_values'])
    
    #Rating of rows matches with new_data
    identified_rows = movies_rating.iloc[acbv['Index_same_values']]
    
    identified_rows =  identified_rows.reset_index()
    
    identified_r = pd.DataFrame(identified_rows['imdb_score']*acbv['Cosine_values'],columns = ['mean_rating'])
    
    nk.append(identified_r['mean_rating'].mean())


from sklearn.metrics import r2_score
r2_score(data2,nk)














"""
features = movielens_dataset.iloc[:,:-1].values
labels = movielens_dataset.iloc[:,-1].values


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

features = scaler.fit_transform(features)


from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.2,random_state = 0)




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

from sklearn.ensemble import RandomForestRegressor
ran_for_regressor = RandomForestRegressor(n_estimators = 10,random_state = 0)
ran_for_regressor.fit(features_train,labels_train)
pred1 = ran_for_regressor.predict(features_test)
score = ran_for_regressor.score(features_test,labels_test)

from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator = ran_for_regressor, X = features_train, y = labels_train, cv = 10)
print ("mean accuracy is",accuracies.mean())
print (accuracies.std())

"""
