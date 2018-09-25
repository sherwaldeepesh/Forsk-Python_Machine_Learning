# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 11:12:35 2018

@author: Deepesh
"""

import pandas as pd
import numpy as np
dataset = pd.read_csv("hackathon_rentomter_nobroker.csv")


import json
internet=[]
ac=[]
club=[]
intercom=[]
pool=[]
cpa=[]
fs=[]
servant=[]
security=[]
sc=[]
gp=[]
park=[]
rwh=[]
stp=[]
hk=[]
pb=[]
vp=[]
for i in dataset['amenities']:
    obj=json.loads(i)
    internet.append(obj['INTERNET'])
    ac.append(obj['AC'])
    if 'CLUB' in obj.keys():
        club.append(obj['CLUB'])
    else:
        club.append(False)
    intercom.append(obj['INTERCOM'])
    if 'FS' in obj.keys():
        fs.append(obj['FS'])
    else:
        fs.append(False)
    if 'CPA' in obj.keys():
        cpa.append(obj['CPA'])
    else:
        cpa.append(False)
    if 'SERVANT' in obj.keys():
        servant.append(obj['SERVANT'])
    else:
        servant.append(False)
    security.append(obj['SECURITY'])
    sc.append(obj['SC'])
    if 'GP' in obj.keys():
        gp.append(obj['GP'])
    else:
        gp.append(False)
    park.append(obj['PARK'])
    if 'RWH' in obj.keys():
        rwh.append(obj['RWH'])
    else:
        rwh.append(False)
   # rwh.append(obj['RWH'])
    if 'STP' in obj.keys():
        stp.append(obj['STP'])
    else:
        stp.append(False)
  #  stp.append(obj['STP'])
    hk.append(obj['HK'])
    pb.append(obj['PB'])
    if 'VP' in obj.keys():
        vp.append(obj['VP'])
    else:
        vp.append(False) 
        
        
        
d1=pd.DataFrame({'ac':ac,'internet':internet,'intercom':intercom,'vp':vp,'stp':stp,'rwh':rwh,'gp':gp,'servant':servant,'cpa':cpa,'fs':fs,'club':club,'fs':fs,'security':security,'sc':sc,'park':park,'hk':hk,'pb':pb})
d1=d1.astype(int)
new_data=pd.concat([dataset,d1],axis=1)

features=new_data.drop(['rent','id'],1)
labels=new_data['rent']

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
for i in features:
    if features[i].dtypes==object:
        features[i]=labelencoder.fit_transform(features[i])

features = features.drop('activation_date',axis=1)

#onehotencoder = OneHotEncoder(categorical_features = [1])

#features = onehotencoder.fit_transform(features).toarray()

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.2,random_state = 0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

features = scaler.fit_transform(features)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10,random_state = 0)

regressor.fit(features_train,labels_train)

score = regressor.score(features_test,labels_test)



from sklearn.tree import DecisionTreeRegressor
regressor_tree = DecisionTreeRegressor(random_state = 0)

regressor_tree.fit(features_train,labels_train)
regressor_tree.predict(features_test)
score_1 = regressor_tree.score(features_test,labels_test)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = features_train, y = labels_train, cv = 10)
print ("mean accuracy is",accuracies.mean())
print (accuracies.std())





"""


#Input Test Set File
dataset = pd.read_csv()      #Test set file name


import json
internet=[]
ac=[]
club=[]
intercom=[]
pool=[]
cpa=[]
fs=[]
servant=[]
security=[]
sc=[]
gp=[]
park=[]
rwh=[]
stp=[]
hk=[]
pb=[]
vp=[]
for i in dataset['amenities']:
    obj=json.loads(i)
    internet.append(obj['INTERNET'])
    ac.append(obj['AC'])
    if 'CLUB' in obj.keys():
        club.append(obj['CLUB'])
    else:
        club.append(False)
    intercom.append(obj['INTERCOM'])
    if 'FS' in obj.keys():
        fs.append(obj['FS'])
    else:
        fs.append(False)
    if 'CPA' in obj.keys():
        cpa.append(obj['CPA'])
    else:
        cpa.append(False)
    if 'SERVANT' in obj.keys():
        servant.append(obj['SERVANT'])
    else:
        servant.append(False)
    security.append(obj['SECURITY'])
    sc.append(obj['SC'])
    if 'GP' in obj.keys():
        gp.append(obj['GP'])
    else:
        gp.append(False)
    park.append(obj['PARK'])
    if 'RWH' in obj.keys():
        rwh.append(obj['RWH'])
    else:
        rwh.append(False)
   # rwh.append(obj['RWH'])
    if 'STP' in obj.keys():
        stp.append(obj['STP'])
    else:
        stp.append(False)
  #  stp.append(obj['STP'])
    hk.append(obj['HK'])
    pb.append(obj['PB'])
    if 'VP' in obj.keys():
        vp.append(obj['VP'])
    else:
        vp.append(False) 
        
        
        
d1=pd.DataFrame({'ac':ac,'internet':internet,'intercom':intercom,'vp':vp,'stp':stp,'rwh':rwh,'gp':gp,'servant':servant,'cpa':cpa,'fs':fs,'club':club,'fs':fs,'security':security,'sc':sc,'park':park,'hk':hk,'pb':pb})
d1=d1.astype(int)
new_data=pd.concat([dataset,d1],axis=1)

features=new_data.drop(['rent','deposit','id'],1)


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
for i in features:
    if features[i].dtypes==object:
        features[i]=labelencoder.fit_transform(features[i])

features = features.drop('activation_date',axis=1)

"""

















import statsmodels.formula.api as sm
features = np.append(arr = np.ones((25000,1)).astype(int), values = features, axis = 1)
features_opt = features[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,35,36,37]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,35,36,37]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,36,37]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,18,19,20,21,22,23,24,25,26,27,29,30,31,32,36,37]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,18,19,20,21,22,23,24,25,26,27,29,31,32,36,37]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,18,19,20,21,22,23,24,25,27,29,31,32,36,37]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:,[0,1,2,3,4,6,7,8,9,10,11,12,14,15,18,19,20,21,22,23,24,25,27,29,31,32,36,37]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:,[0,1,2,3,4,7,8,9,10,11,12,14,15,18,19,20,21,22,23,24,25,27,29,31,32,36,37]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:,[0,1,2,3,4,7,8,9,10,11,12,14,15,18,20,21,22,23,24,25,27,29,31,32,36,37]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()






from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features_opt,labels,test_size = 0.2,random_state = 0)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10,random_state = 0)

regressor.fit(features_train,labels_train)

score = regressor.score(features_test,labels_test)


from sklearn.tree import DecisionTreeRegressor
regressor_tree = DecisionTreeRegressor(random_state = 0)

regressor_tree.fit(features_train,labels_train)

score_1 = regressor_tree.score(features_test,labels_test)