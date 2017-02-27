# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 19:26:01 2015

@author: lifeng
"""

import numpy as np
from sklearn import svm
from sklearn.cross_validation import KFold
import glob
import matplotlib.pyplot as plt
import cv2



def autonorm1(mat):
    M = mat.max()
    m = mat.min()
    norm_mat = (mat-m)*1.0/(M-m)
    return norm_mat

def get_img(path):
    images=[]
    for file in glob.glob(path):
        img=plt.imread(file)
        img=cv2.equalizeHist(img)
        img=autonorm1(img)
        L=img.size
        img.shape=[1,L]
        images.append(img[0])
    return np.array(images)

def svmclassify(x_train,y_train,x_test,y_test):
    svc = svm.SVC(kernel='rbf',C=4,gamma=0.00125)
    #svc = svm.SVC(kernel='poly',degree=3)
    svc.fit(x_train,y_train)
    svc.predict(x_test)
    return svc.score(x_test,y_test)


males=get_img('H:\\weishi\\face_recognization\\pythoncode\\galleryset_m\\*.bmp')
females=get_img('H:\\weishi\\face_recognization\\pythoncode\\galleryset_f\\*.bmp')
faces= np.row_stack((females,males))


target=np.array([1]*males.shape[0]+[0]*females.shape[0])       


num_fold=10
def m():
    folds = KFold(faces.shape[0],n_folds=num_fold,shuffle=True)
    svm_score=0
    for train_index, test_index in folds:
        X_train = faces[train_index]
        X_test  = faces[test_index]
        Y_train = target[train_index]
        Y_test = target[test_index]
        svm_single=svmclassify(X_train,Y_train,X_test,Y_test)
        svm_score +=svm_single
        print 'svm_single',svm_single
    return svm_score/num_fold







   