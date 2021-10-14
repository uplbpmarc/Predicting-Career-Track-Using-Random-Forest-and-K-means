# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:23:13 2021

@author: Ranzivelle
"""

#Importing the pertinent packages 
import numpy as np 
import pandas as pd 
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

#Loading the data
balance_data = pd.read_csv(r"file_address/path", sep= ',')        

#Assigning the predictors and target variable 
#X=balance_data[:,n], where n is the number of predictors 
#Y=balance_data[:,n], where n is the index of the target variable
#Example is for 10 predictors and 1 target variable

balance_data=balance_data.to_numpy ()
X = balance_data[:,0:10]
Y = balance_data[:,10]

#Splitting into training/testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, Y)

##Oversampling the minority class 
##SMOTE 
#X_train, X_test, y_train, y_test = train_test_split(X, Y)
#sm = SMOTE (random_state=42, sampling_strategy=1)
#X_train, y_train= sm.fit_resample(X_train, y_train)

#
##Undersampling 
#X_train, X_test, y_train, y_test = train_test_split(X, Y)
#rus = RandomUnderSampler(random_state=0)
#X_train, y_train = rus.fit_resample(X_train, y_train)

#Instantiating the RF Model  
rfc=RandomForestClassifier(random_state=42)

#Performing the GridSearchCV
param_grid = {'n_estimators': [50, 100, 200, 500], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [4,5,6,7,8,9, 10, 11,12],'criterion':['gini', 'entropy']}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 10)
CV_rfc.fit(X_train, y_train)

bparam = CV_rfc.best_params_
print(bparam)

#Setting the Random Forest Using the Hyperparameters 
rfc1=RandomForestClassifier(random_state=42, max_features=bparam['max_features'], n_estimators= bparam['n_estimators'], max_depth=bparam['max_depth'], criterion=bparam['criterion'], oob_score = True )

#Training the Model 
rfc1.fit (X_train, y_train)
pred0=rfc1.predict(X_train)

#Training performance metrics
cf0=confusion_matrix(y_train, pred0)
print (cf0 )
train_acc= accuracy_score(y_train, pred0)
print ('Train Accuracy : ', train_acc) 
train_sensitivity= cf0[0,0]/(cf0[0,0]+cf0[0,1])
print('Train Sensitivity : ', train_sensitivity  )
train_specificity= cf0[1,1]/(cf0[1,0]+cf0[1,1])
print('Train Specificity : ', train_specificity)
train_auc = roc_auc_score(y_train, pred0)
print('Train ROC AUC : ', train_auc)


#Testing the Model 
pred=rfc1.predict(X_test)

#Testing performance metrics
cf1=confusion_matrix(y_test, pred)
print (cf1 )
acc=accuracy_score(y_test, pred)
print ('Test Accuracy : ', acc) 
sensitivity= cf1[0,0]/(cf1[0,0]+cf1[0,1])
print('Test Sensitivity : ', sensitivity  )
specificity= cf1[1,1]/(cf1[1,0]+cf1[1,1])
print('Test Specificity : ', specificity)
auc = roc_auc_score(y_test, pred)
print('Test ROC AUC : ', auc)


#Determining the Feature Importance 
imp= rfc1.feature_importances_
imp=np.transpose(imp)
print (rfc1.feature_importances_)
impexcel=pd.DataFrame(imp)
#create file in your local drive  
export_excel = impexcel.to_excel(r'file_address/path', index = None, header=True)