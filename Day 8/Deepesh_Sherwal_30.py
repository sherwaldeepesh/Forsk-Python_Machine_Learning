# -*- coding: utf-8 -*-
"""
Created on Tue May 22 12:36:14 2018

@author: deepe
"""

import urllib2
iccRanking = "https://www.icc-cricket.com/rankings/mens/team-rankings/odi"
rankingTable=urllib2.urlopen(iccRanking)

from bs4 import BeautifulSoup

soup = BeautifulSoup(rankingTable)

all_tables = soup.find_all('tbody')

tr_table = all_tables[0].find_all('tr')
A = []
B = []
C = []
D = []
E = []

for row in tr_table:
    cells = row.find_all('td')
    A.append(cells[0].text.strip())
    B.append(cells[1].text.strip())
    C.append(cells[2].text.strip())
    D.append(cells[3].text.strip())
    E.append(cells[4].text.strip())
    
import pandas as pd
df=pd.DataFrame()
df["Position"] = A
df["Team"] = B
df["Matches"] = C
df["Points"] = D
df["Rating"] = E



df