# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 22:02:48 2018

@author: deepe
"""

import urllib2
url = "https://friendorfollow.com/twitter/most-followers/"
page = urllib2.urlopen(url)

from bs4 import BeautifulSoup
soup = BeautifulSoup(page)

all_tables = soup.find_all(class_ = 'tUser')

A = []

for row in all_tables:
    A.append(row.text.strip())
    
import pandas as pd
dataset_most_followers = pd.DataFrame(A,columns = ['screen_name'])    