# -*- coding: utf-8 -*-
"""
Created on Sat May 23 21:57:10 2015

@author: llq
"""
import numpy as np
from pandas import DataFrame

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import time


def getCrossValidation(X,y,classifier,featurename):
    folds = KFold(len(y),n_folds=5,shuffle=True,random_state=np.random.RandomState(1))
    predicted_probability=-np.ones(len(y))
    predicted_score=-np.ones(len(y))
    X=np.array(X)
    y=np.array(y)
    for train_index, test_index in folds:
        X_train = X[train_index]
        X_test  = X[test_index]
        y_train = y[train_index]
        probability_test =(classifier.fit(X_train, y_train)).predict_proba(X_test)
        predicted_probability[test_index]=probability_test[:, 1]
        predicted_score[test_index]=(classifier.fit(X_train, y_train)).predict(X_test)   
    result=model_evaluation(y,predicted_probability,predicted_score,featurename)
    return result

def model_evaluation(y,predicted_probability,predicted_score,featurename):
    precision, recall, thresholds = precision_recall_curve(y, predicted_probability)
    aupr_score = auc(recall, precision)

    fpr, tpr, thresholds = roc_curve(y, predicted_probability,pos_label=1)
    auc_score = auc(fpr, tpr)
    accuracy=accuracy_score(y,predicted_score)
    precision=precision_score(y,predicted_score)
    recall=recall_score(y,predicted_score)
    
    print('results for feature:'+featurename)
    print('****AUC score:%.3f, AUPR score:%.3f, recall score:%.3f, precision score:%.3f, accuracy:%.3f****' %(auc_score,aupr_score,recall,precision,accuracy))
    result=[aupr_score,auc_score,accuracy,precision,recall]
    return result
    
    
#..............................................................................

if __name__ == '__main__':
      
    X=np.loadtxt("Feature3Lamda0.5.txt")
    labels=np.loadtxt("labels.txt")
    #classifier=svm.SVC(probability=True)
    classifier=RandomForestClassifier(random_state=1,n_estimators=100)
    
    start=time.clock() 
    result=getCrossValidation(X, labels, classifier, 'Feature3Lamda0.5')
    end=time.clock()
    
    print ('runing time %.3f minutes' %((end-start)/60.0))
  