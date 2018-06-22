# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 19:24:30 2018

@author: deepe
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 11:19:53 2018

@author: deepe
"""

import pandas as pd

dataset = pd.read_csv('election_data.csv')

dataset1 = dataset[dataset.groupby('Name_of_Candidate').Name_of_Candidate.transform(len) <= 1]
dataset2 = dataset[dataset.groupby('Name_of_Candidate').Name_of_Candidate.transform(len) > 1]
jk = []
for i in dataset1['Name_of_Candidate']:
    jk.append(i)
jk = list(set(jk))

mk = []
for j in dataset2['Name_of_Candidate']:
    mk.append(j)
mk = list(set(mk))


