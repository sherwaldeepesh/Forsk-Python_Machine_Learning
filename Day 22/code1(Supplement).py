# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 11:19:53 2018

@author: deepe
"""

import pandas as pd

dataset = pd.read_csv('election_data.csv')

dataset = dataset[dataset.groupby('Name_of_Candidate').Name_of_Candidate.transform(len) > 1]

jk = []
for i in dataset['Name_of_Candidate']:
    jk.append(i)
jk = list(set(jk))

ass_cand_name = []
for i in jk:
    ass_no = list(set(dataset["Assembly_no"][dataset["Name_of_Candidate"]==i]))
    if len(ass_no) > 1:
        ass_cand_name.append(i)

ppl_lst, freq = [], []

for i in ass_cand_name:
    elec_lst = list(set(dataset["Constituency_no"][dataset["Name_of_Candidate"]==i]))
    if len(elec_lst) > 1:
        freq.append(elec_lst)
        ppl_lst.append(i)


total =[]
for c,f in zip(ppl_lst,freq):
    votes = []
    for const in f:
        votes.append(dataset["Votes"][(dataset["Constituency_no"]==const) & (dataset["Name_of_Candidate"]==c)].mean())
    total.append(votes)

# check for vote increase or decrese
performance = []
for t in total:
    max_no = max(t)
    ind = t.index(max_no)
    if ind == 0:
        performance.append(False)
    else:
        performance.append(True)

dec = performance.count(False)
inc = performance.count(True)

candi_data = [dec,inc]

import matplotlib.pyplot as plt
labels = ["No-Benifit","Benifit"]
plt.pie(candi_data,labels=labels,autopct="%1.2f%%")
plt.title("Candidate Performance after Changing Constitution_no")
plt.show()

