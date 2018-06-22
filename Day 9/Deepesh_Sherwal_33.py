# -*- coding: utf-8 -*-
"""
Created on Wed May 23 12:48:27 2018

@author: deepe
"""

from pymongo import MongoClient
from datetime import datetime

ref = MongoClient('localhost',27017)
mydb1 = ref.db_university

def add_detail(Student_Name,Student_Age,Student_Roll_no,Student_Branch):
    unique_detail = mydb1.university_details.find_one({"Student Roll no":Student_Roll_no}, {"_id":0})
    if unique_detail:
        return "Details already exist"
    else:
        mydb1.university_details.insert(
                {"Student Name":Student_Name,
                 "Student Age":Student_Age,
                 "Student Roll no":Student_Roll_no,
                 "Student Branch":Student_Branch,
                 "Date-Time" : datetime.now()
                        })
        return "Details added Successfully"

name=raw_input("Student Name ")
age=raw_input("Student Age ")
rollno=raw_input("Student Roll no ")
branch=raw_input("Student Branch ")

print add_detail(name,age,rollno,branch)