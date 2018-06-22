# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:17:50 2018

@author: deepe
"""

import requests
url="https://api.mlab.com/api/1/databases/my-db/collections/university_details?apiKey=zKM7h9Xa6icPwVfpBDCPZDR3dVu4KFWY"
data =  [{
	"Student Name": "Deepesh Sherwal",
	"Student Age": "20",
	"Student Roll no": "6548",
	"Student Branch": "CSE"
}, {
	"Student Name": "Abhishek Sharma",
	"Student Age": "22",
	"Student Roll no": "6549",
	"Student Branch": "CSE"
}, {
	"Student Name": "Ajay raj",
	"Student Age": "21",
	"Student Roll no": "6550",
	"Student Branch": "CSE"
}, {
	"Student Name": "Anish Pratap",
	"Student Age": "22",
	"Student Roll no": "6551",
	"Student Branch": "CSE"
}, {
	"Student Name": "Akhilendra Singh",
	"Student Age": "23",
	"Student Roll no": "6552",
	"Student Branch": "CSE"
}, {
	"Student Name": "Ankit Sharma",
	"Student Age": "21",
	"Student Roll no": "6553",
	"Student Branch": "CSE"
}, {
	"Student Name": "Arjun Sharma",
	"Student Age": "21",
	"Student Roll no": "6554",
	"Student Branch": "CSE"
}, {
	"Student Name": "Ashutosh Goyal",
	"Student Age": "22",
	"Student Roll no": "6555",
	"Student Branch": "CSE"
}, {
	"Student Name": "Astha Jain",
	"Student Age": "21",
	"Student Roll no": "6556",
	"Student Branch": "CSE"
}, {
	"Student Name": "Bindu Sharma",
	"Student Age": "20",
	"Student Roll no": "6557",
	"Student Branch": "CSE"
}]
connection = requests.post(url = url, json = data)
