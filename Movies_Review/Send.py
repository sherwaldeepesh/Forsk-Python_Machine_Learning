# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 14:10:06 2018

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
 