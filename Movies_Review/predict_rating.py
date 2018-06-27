# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 14:15:03 2018

@author: deepe
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ratings = pd.read_table('ratings.dat',sep='::',names=['user_id','movie_id','rating','time'])
users = pd.read_table('users.dat',sep='::',names=['user_id','gender','age','occupation','zip'])
movies = pd.read_table('movies.dat',sep='::',names=['movie_id','title','genre'])


"""
movies = movies.drop('title', axis=1)
movies.head()
"""

movies['year'] = movies.title.str.extract("\((\d{4})\)", expand=True)
movies.year = pd.to_datetime(movies.year, format='%Y')
movies.year = movies.year.dt.year # As there are some NaN years, resulting type will be float (decimals)
movies.title = movies.title.str[:-7]



genres_unique = pd.DataFrame(movies.genre.str.split('|').tolist()).stack().unique()
genres_unique = pd.DataFrame(genres_unique, columns=['genre']) # Format into DataFrame to store later
movies = movies.join(movies.genre.str.get_dummies().astype(int))
movies.drop('genre', inplace=True, axis=1)


movielens_dataset = pd.merge(pd.merge(ratings,users),movies)

#Most rated Movies
movielens_dataset.title.value_counts()[:25]


#Most highly rated
movie_stats = movielens_dataset.groupby('title').agg({'rating': [np.size, np.mean]})
movie_stats.head()

# sort by rating average
movie_stats.sort_values([('rating', 'mean')], ascending=False).head(10)

#Movies rated more than 100 times
atleast_100 = movie_stats['rating']['size'] >= 100
movie_stats[atleast_100].sort_values([('rating', 'mean')], ascending=False)[:15].plot(kind = 'barh',legend = False)

#Age plot of Users
movielens_dataset.age.plot.hist(bins=30)
plt.title("Distribution of users' ages")
plt.ylabel('count of users')
plt.xlabel('age')

#Yearly plotting of release count Movies
movielens_dataset.year.plot.hist(bins = 30)
plt.title("Distribution of Movies' Yearly")
plt.xlabel("Year")

#Plotting Genres Popularity
dfe = movielens_dataset[['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary','Drama',
 'Fantasy',
 'Film-Noir',
 'Horror',
 'Musical',
 'Mystery',
 'Romance',
 'Sci-Fi',
 'Thriller',
 'War',
 'Western']]
dfe.columns
asd = []
for column in dfe:
    asd.append(dfe[column].sum())
fig, ax = plt.subplots()    
width = 0.75 # the width of the bars 
ind = [u'Action', u'Adventure', u'Animation', u'Childrens', u'Comedy',
       u'Crime', u'Documentary', u'Drama', u'Fantasy', u'Film-Noir', u'Horror',
       u'Musical', u'Mystery', u'Romance', u'Sci-Fi', u'Thriller', u'War',
       u'Western']  # the x locations for the groups
ax.barh(ind, asd, color="blue")
plt.title('Genres Popularity')
plt.xlabel('No of Movies')
    
#Low Rated Movies
lower_rated = movie_stats['rating']['mean'] < 1.5
movie_stats[lower_rated].head(10).plot(kind = 'barh',figsize = (8,5))
 
 




#Converting 18 categories of genres into three clusters

features = movielens_dataset.iloc[:,10:].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('no of clusters')
plt.show()

kmeans = KMeans(n_clusters = 9, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(features)

plt.scatter(features[y_kmeans == 0,0],features[y_kmeans == 0,1], s = 100, label = 'Cluster 1')
plt.scatter(features[y_kmeans == 1,0],features[y_kmeans == 1,1], s = 100, label = 'Cluster 2')
plt.scatter(features[y_kmeans == 2,0],features[y_kmeans == 2,1], s = 100, label = 'Cluster 3')
plt.scatter(features[y_kmeans == 3,0],features[y_kmeans == 3,1], s = 100, label = 'Cluster 4')
plt.scatter(features[y_kmeans == 4,0],features[y_kmeans == 4,1], s = 100, label = 'Cluster 5')
plt.scatter(features[y_kmeans == 5,0],features[y_kmeans == 5,1], s = 100, label = 'Cluster 6')
plt.scatter(features[y_kmeans == 6,0],features[y_kmeans == 6,1], s = 100, label = 'Cluster 7')
plt.scatter(features[y_kmeans == 7,0],features[y_kmeans == 7,1], s = 100, label = 'Cluster 8')
plt.scatter(features[y_kmeans == 8,0],features[y_kmeans == 8,1], s = 100, label = 'Cluster 9')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 300, label = 'Centroid')
plt.title('CLustering')
plt.legend()
plt.show()

#new_dataset
lens_dataset = movielens_dataset.iloc[:,:10]
a=[]
for i in range(1000209):
    a.append(y_kmeans[i])
a = pd.DataFrame(a,columns = ['Cluster'])    

merged = lens_dataset.merge(a, left_index=True, right_index=True, how='inner')

#avg_rating by movie_id

movies_rating = merged.groupby(['movie_id','Cluster'],as_index = False)['rating'].mean()
bn = merged.groupby(['movie_id','Cluster']).size().reset_index().groupby('Cluster')[[0]].mean()

merged.pivot_table(columns='Cluster',aggfunc=sum)

#avg_rating by user_id

users_rating = merged.groupby(['user_id','Cluster'],as_index = False)['rating'].mean()
bn = merged.groupby(['user_id','Cluster']).size().reset_index().groupby('Cluster')[[0]].max()

#avg_rating based on gender,age,occupation

age_rating = merged.groupby(['gender','age','occupation','Cluster'],as_index = False)['rating'].mean()


merged.groupby('Cluster',as_index = False)['rating'].mean().plot(kind = 'barh')





  


import time
ratings['time'] = ratings['time'].apply(lambda x: time.strftime('%Y', time.localtime(x)))
ratings.head()


age_rating = movielens_dataset.groupby('age',as_index = False)['rating'].mean()

users_rating = movielens_dataset.groupby('user_id', as_index=False)['rating'].mean()




#Movies_avg_Rating
movies_rating = movielens_dataset.groupby(['movie_id',
 'year', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary','Drama',
 'Fantasy',
 'Film-Noir',
 'Horror',
 'Musical',
 'Mystery',
 'Romance',
 'Sci-Fi',
 'Thriller',
 'War',
 'Western'], as_index = False)['rating'].mean()  
    
movies_rating1 = movies_rating.iloc[:,2:-1]    
#New Movie data inserts
data1 = np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]).reshape(1,-1)

#   
cbv = []
for item in range(len(movies_rating1)):
    if (movies_rating1.iloc[item].values == data1).all():
        cbv.append(item)

acbv = pd.DataFrame(cbv,columns = ['Index_same_values'])





"""
ran = movielens_dataset.groupby(['age','occupation','gender','user_id'],as_index = False)['rating'].mean()

ran = ran[['user_id','gender','age','occupation','rating']]

typegr = movielens_dataset.groupby('genre',as_index=False)['rating'].mean()

"""
typeuser = movielens_dataset.groupby(['user_id','gender','age','occupation', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary','Drama',
 'Fantasy',
 'Film-Noir',
 'Horror',
 'Musical',
 'Mystery',
 'Romance',
 'Sci-Fi',
 'Thriller',
 'War',
 'Western'],as_index=False)['rating'].mean()

typeuser.dtypes

typeuser['zip'] = typeuser['zip'].apply(lambda x: x.split('-')[0]).astype('int64')



movielens_dataset.dtypes


features = age_rating.iloc[:,:-1].values
labels = age_rating.iloc[:,-1].values



"""
Convert Timestamp to datetime
pd.to_datetime(df['mydates']).apply(lambda x: x.datetime())

"""


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

features[:,0] = labelencoder.fit_transform(features[:,0])
features[:,2] = labelencoder.fit_transform(features[:,2])


for i in range(1,19):
    features[:,i]  = labelencoder.fit_transform(features[:,i])
    
   

from sklearn.preprocessing import StandardScaler,OneHotEncoder
scaler = StandardScaler()

onehotencoder = OneHotEncoder(categorical_features = [2])
features = onehotencoder.fit_transform(features).toarray()

features = scaler.fit_transform(features)

from sklearn import preprocessing as pre
normalized_X = pre.normalize(features)


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
from sklearn.ensemble import RandomForestRegressor
ran_for_regressor = RandomForestRegressor(n_estimators = 10,random_state = 0)
ran_for_regressor.fit(features_train,labels_train)
pred1 = ran_for_regressor.predict(features_test)
score = ran_for_regressor.score(features_test,labels_test)

from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator = ran_for_regressor, X = features_train, y = labels_train, cv = 10)
print ("mean accuracy is",accuracies.mean())
print (accuracies.std())



