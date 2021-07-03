# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 13:17:26 2021

@author: Hassani
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.tree import DecisionTreeRegressor
import pickle


df_train=pd.read_csv('Training.csv')
df_test=pd.read_csv('Testing.csv')

df_train = df_train.iloc[:,:-1]

df_train = df_train.sample(frac = 1)






x=df_train.iloc[:,:-1]
y=df_train.iloc[:,-1:]


x_test=df_test.iloc[:,:-1]
y_test=df_test.iloc[:,-1:]


transformer = make_column_transformer((['prognosis'], OneHotEncoder()))

y = OneHotEncoder().fit_transform(y).toarray()
y_t = OneHotEncoder().fit_transform(y_test).toarray()


regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)


y_pred=regressor.predict(x_test)

filename = 'finalized_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
    
    
with open(filename,'rb') as file:
    loaded_model = pickle.load(file)
    



result = loaded_model.predict(x)
 
