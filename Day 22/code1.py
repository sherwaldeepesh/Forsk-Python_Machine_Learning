# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:44:01 2018

@author: deepe
"""


import pandas as pd

data = pd.read_csv("election_data.csv")

cand_names = data["Name_of_Candidate"].drop_duplicates()

ass_cand_names = []

# check for candidates who have contested from more than 1 assembly election
for i in cand_names:
    ass_no = list(set(data["Assembly_no"][data["Name_of_Candidate"]==i]))
    if len(ass_no) > 1:
        ass_cand_names.append(i)

# check for change in constituency
ppl_lst, freq = [], []

for i in ass_cand_names:
    elec_lst = list(set(data["Constituency_no"][data["Name_of_Candidate"]==i]))
    if len(elec_lst) > 1:
        freq.append(elec_lst)
        ppl_lst.append(i)

# check for votes for each constiteuncy
total = []
for c,f in zip(ppl_lst,freq):
    votes = []
    for const in f:
        votes.append(data["Votes"][(data["Constituency_no"]==const) & (data["Name_of_Candidate"]==c)].mean())
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

