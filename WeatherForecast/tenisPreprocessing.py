# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 18:25:32 2023

@author: Lenovo
"""

import pandas as pd
import numpy as np

data = pd.read_csv("odev_tenis.csv")

# ----- Preprocessing -----

# Missing Values
""" None """

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# outlook = data.iloc[:,0:1].values
# play = data.iloc[:,-1:].values
# windy = data.iloc[:,3:4].values

# from sklearn import preprocessing

# outlook encoding
# labelEncoder = preprocessing.LabelEncoder()
# outlook[:,0] = labelEncoder.fit_transform(data.iloc[:,0:1])

# oneHotEncoder = preprocessing.OneHotEncoder()
# outlook = oneHotEncoder.fit_transform(outlook).toarray()

# #lay encoding
# play[:,0] = labelEncoder.fit_transform(play[:,0])

# windy encoding
# windy[:,0] = labelEncoder.fit_transform(windy[:,0])

#Shortcut for all encodings
data2 = data.apply(LabelEncoder().fit_transform)

outlook = data.iloc[:,0:1].values
outlook = OneHotEncoder().fit_transform(outlook).toarray()

#The Columns that no need for encoding
remains = data.iloc[:,1:3]

outlook = pd.DataFrame(data = outlook, index=range(14), columns=["overcast","rainy","sunny"])

data = pd.concat([outlook,remains],axis = 1)

data = pd.concat([data,data2.iloc[:,-2:]], axis = 1)

humidity = data.iloc[:,-3:-2].values

left = data.iloc[:,0:4]
right = data.iloc[:,-2:]

data3 = pd.concat([left,right],axis = 1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(data3,humidity,test_size=0.33,random_state=0)

# ----- Multiple Linear Regression Model -----
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_predict = regressor.predict(x_test)
print("Prediction value before backward elimination \n",y_predict)

# ---- Backward Elimination -----
import statsmodels.api as sm

alpha = np.append(arr = np.ones((14,1)).astype(int), values=data3, axis = 1)

alpha_list = data3.iloc[:,[0,1,2,3,4,5]].values
alpha_list = np.array(alpha_list,dtype=float)
model = sm.OLS(endog=humidity, exog=alpha_list).fit()

print(model.summary())

alpha_list = data3.iloc[:,[0,1,2,3,5]].values
model = sm.OLS(endog=humidity, exog=alpha_list).fit()

print(model.summary())

x_train = pd.concat([x_train.iloc[:,0:4],x_train.iloc[:,-1:]],axis = 1)
x_test = pd.concat([x_test.iloc[:,0:4],x_test.iloc[:,-1:]],axis = 1)

regressor.fit(x_train,y_train)

y_predict = regressor.predict(x_test)

print("Prediction value after backward elimination \n", y_predict)