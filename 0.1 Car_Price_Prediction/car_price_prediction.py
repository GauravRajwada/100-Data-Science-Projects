# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:05:18 2020

@author: Gaurav
"""

from datetime import date
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor 
import numpy as np

df=pd.read_csv("E:/All Data Set/Car_Datasets.csv")

cur_year= date.today().year
df['current_year']=cur_year
df['No_of_year']=df["current_year"]-df['Year']

df.drop(['Year',"current_year","Car_Name"],axis=1,inplace=True)

df=pd.get_dummies(df,drop_first=True)

sb.pairplot(df)

"""Plotting correlation"""
corrmat=df.corr() 
top_corr_features=corrmat.index 
plt.figure(figsize=(20,20)) 
g=sb.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

x=df.iloc[:,1:]
y=df.iloc[:,0]

"""Finding Important features"""
#Feature Importance 
model=ExtraTreesRegressor() 
model.fit(x,y)

#plot graph of feature importances for better visualization 
feat_importances = pd.Series(model.feature_importances_, index=x.columns) 
feat_importances.nlargest(8).plot(kind='barh') 
plt.show()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)



#Randomized Search CV
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


rf=RandomForestRegressor()
regressor=RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = -1)


regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

sb.distplot(y_test-y_pred)

import pickle
file = open("E:/Project/Car_Dekho/car_model.pkl", 'wb')

pickle.dump(regressor,file)
