# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 23:37:38 2018

@author: deepe
"""

import urllib2
from bs4 import BeautifulSoup

#Ghaziabad
url = "http://www.intellicast.com/Local/Observation.aspx?location=INXX0051"
page = urllib2.urlopen(url)
soup = BeautifulSoup(page)
all_tables = soup.find_all('table')
tr_table = soup.find('table', class_ = 'Container')

#Faridabad
url1 = "http://www.intellicast.com/Local/Observation.aspx?location=INXX0046"
page1 = urllib2.urlopen(url1)
soup1 = BeautifulSoup(page1)
all_tables1 = soup.find_all('table')
tr_table1 = soup.find('table', class_ = 'Container')

#Noida
url2 = "http://www.intellicast.com/Local/Observation.aspx?location=INXX0384"
page2 = urllib2.urlopen(url2)
soup2 = BeautifulSoup(page2)
all_tables2 = soup.find_all('table')
tr_table2 = soup.find('table', class_ = 'Container')

#New Delhi
url3 = "http://www.intellicast.com/Local/Observation.aspx?location=INXX0096"
page3 = urllib2.urlopen(url3)
soup3 = BeautifulSoup(page3)
all_tables3 = soup.find_all('table')
tr_table3 = soup.find('table', class_ = 'Container')


#generating List

A = []
B = []
C = []
D = []
E = []
F = []


B1 = []
C1 = []
D1 = []
E1 = []
F1 = []


B2 = []
C2 = []
D2 = []
E2 = []
F2 = []


B3 = []
C3 = []
D3 = []
E3 = []
F3 = []



for row in tr_table('tr'):
    cells = row.findAll('td')
    if len(cells) == 10:
        A.append(cells[0].text.strip())
        B.append(cells[1].text.strip().encode('ascii', 'ignore'))
        C.append(cells[2].text.strip().encode('ascii', 'ignore'))
        D.append(cells[3].text.strip().split('%')[0])
        E.append(cells[4].text.strip().split('in')[0])
        F.append(cells[5].text.strip().split('mi')[0])
        
for row in tr_table1('tr'):
    cells1 = row.findAll('td')
    if len(cells1) == 10:
        #A1.append(cells1[0].text.strip())
        B1.append(cells1[1].text.strip().encode('ascii', 'ignore'))
        C1.append(cells1[2].text.strip().encode('ascii', 'ignore'))
        D1.append(cells1[3].text.strip().split('%')[0])
        E1.append(cells1[4].text.strip().split('in')[0])
        F1.append(cells1[5].text.strip().split('mi')[0])
        

for row in tr_table2('tr'):
    cells = row.findAll('td')
    if len(cells) == 10:
        #A2.append(cells[0].text.strip())
        B2.append(cells[1].text.strip().encode('ascii', 'ignore'))
        C2.append(cells[2].text.strip().encode('ascii', 'ignore'))
        D2.append(cells[3].text.strip().split('%')[0])
        E2.append(cells[4].text.strip().split('in')[0])
        F2.append(cells[5].text.strip().split('mi')[0])

for row in tr_table3('tr'):
    cells = row.findAll('td')
    if len(cells) == 10:
        #A3.append(cells[0].text.strip())
        B3.append(cells[1].text.strip().encode('ascii', 'ignore'))
        C3.append(cells[2].text.strip().encode('ascii', 'ignore'))
        D3.append(cells[3].text.strip().split('%')[0])
        E3.append(cells[4].text.strip().split('in')[0])
        F3.append(cells[5].text.strip().split('mi')[0])

        
        
import pandas as pd
dataset = pd.DataFrame()
dataset[A[0]] = A[1:]
dataset[" ".join([B[0],'Gha'])] = B[1:]
dataset[" ".join([C[0],'Gha'])] = C[1:]
dataset[" ".join([D[0],'Gha'])] = D[1:]
dataset[" ".join([E[0],'Gha'])] = E[1:]
dataset[" ".join([F[0],'Gha'])] = F[1:]

dataset[" ".join([B1[0],'Far'])] = B1[1:]
dataset[" ".join([C1[0],'Far'])] = C1[1:]
dataset[" ".join([D1[0],'Far'])] = D1[1:]
dataset[" ".join([E1[0],'Far'])] = E1[1:]
dataset[" ".join([F1[0],'Far'])] = F1[1:]


dataset[" ".join([B2[0],'Noi'])] = B2[1:]
dataset[" ".join([C2[0],'Noi'])] = C2[1:]
dataset[" ".join([D2[0],'Noi'])] = D2[1:]
dataset[" ".join([E2[0],'Noi'])] = E2[1:]
dataset[" ".join([F2[0],'Noi'])] = F2[1:]

dataset[" ".join([B3[0],'NDEL'])] = B3[1:]
dataset[" ".join([C3[0],'NDEL'])] = C3[1:]
dataset[" ".join([D3[0],'NDEL'])] = D3[1:]
dataset[" ".join([E3[0],'NDEL'])] = E3[1:]
dataset[" ".join([F3[0],'NDEL'])] = F3[1:]

     
dataset.to_csv('Train_set.csv', sep=',')

dataset.dtypes       

