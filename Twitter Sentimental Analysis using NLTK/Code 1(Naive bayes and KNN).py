# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 22:52:45 2018

@author: deepe
"""

import pandas as pd

train_dataset = pd.read_csv('train_tweets.csv')

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(31962):
    review = re.sub('[^a-zA-Z]', ' ', train_dataset['tweet'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 500)
features = cv.fit_transform(corpus).toarray()
labels = train_dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.3, random_state = 0)

"""
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(features_train,labels_train)

labels_pred = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pred)

score = classifier.score(features_test,labels_test)
"""

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, p = 2)
classifier.fit(features_train,labels_train)

labels_pred = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pred)

score = classifier.score(features_test,labels_test)
