# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 11:22:44 2018

@author: deepe
"""

import pandas as pd
dataset = pd.read_csv('movie.csv')

dataset.describe()

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(2000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
features = cv.fit_transform(corpus).toarray()

labels = dataset.iloc[:,0].values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labels = labelencoder.fit_transform(labels)

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.25,random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(features_train,labels_train)

pred = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(labels_test,pred)

score = classifier.score(features_test,labels_test)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = features_train, y = labels_train, cv = 10)
print ("mean accuracy is",accuracies.mean())
print (accuracies.std())


