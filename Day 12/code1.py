# -*- coding: utf-8 -*-
"""
Created on Mon May 28 11:40:27 2018

@author: deepe
"""

import urllib2
url = "https://en.wikipedia.org/wiki/List_of_states_and_union_territories_of_India_by_area"

scrap = urllib2.urlopen(url)

from bs4 import BeautifulSoup

soup = BeautifulSoup(scrap)

all_tables = soup.find_all('table')
tr_table = all_tables[1].find_all('tr')
A=[]
B=[]
C=[]
D=[]
E=[]
F=[]

for row in tr_table:
    cells = row.find_all('td')
    if len(cells) == 7:
        A.append(cells[0].text.strip())
        B.append(cells[1].text.strip())
        C.append(cells[2].text.strip())
        D.append(cells[3].text.strip())
        E.append(cells[4].text.strip())
        F.append(cells[5].text.strip())
        
import pandas as pd
df = pd.DataFrame()
df['Rank']=A
df['State']=B
df['Area']=C
df['Region']=D
df['National Share']=E
df['Country of Comaprable size']=F
pn = df.values[:6]

import matplotlib.pyplot as plt
x = [pn[i][1] for i in range(len(pn))]
y = [float(pn[i][-2]) for i in range(len(pn))]
explode = (0.1,0,0,0,0,0)
plt.pie(y,explode = explode,labels = x,autopct = '%1.2f%%')
plt.pie()
plt.title('State with largest national Share')
plt.show()
