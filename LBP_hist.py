# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 22:09:00 2015

@author: lifeng
"""

import numpy as np
import random
from sklearn import datasets,svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from PIL import Image
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import cv2

def binary2dec(lst):
    L=len(lst)
    dec=0
    lst1=lst[:]
    lst1.reverse()
    for i in range(L):
        dec+=lst1[i]*(2**i)
    return dec
    
    
def min_value_of_binary2dec(lst):
    L=len(lst)
    min_value=binary2dec(lst)
    for i in range(1,L):
        temp_lst=lst[i:]+lst[:i]
        temp_value=binary2dec(temp_lst)
        min_value=min(min_value,temp_value)
    return min_value
    
    
def LBP(face_mat,R=2,PP=8):
    '''
    R is the radius，PP is the number of points
    '''
    height,width=face_mat.shape
    pi=math.pi
    face_LBP=np.zeros([height,width],dtype=np.uint8)
    for x in range(height):
        for y in range(width):
            center=face_mat[x,y]
            er=list()
            for p in range(1,PP+1):     
                p=float(p)
                xp= x+R*math.cos(2*pi*(p/PP))
                yp= y-R*R*math.sin(2*pi*(p/PP))
                xp_low=min(max(math.floor(xp),0),height-1)
                xp_upper=min(max(math.ceil(xp),0),height-1)
                yp_low=min(max(math.floor(yp),0),width-1)
                yp_upper=min(max(math.ceil(yp),0),width-1)
                dx=xp-xp_low
                dy=yp-yp_low
                f00=face_mat[xp_low,yp_low]
                f01=face_mat[xp_low,yp_upper]
                f11=face_mat[xp_upper,yp_upper]
                f10=face_mat[xp_upper,yp_low]
                pixel=f00*(1-dx)*(1-dy)+f01*(1-dx)*dy+f11*dx*dy+f10*dx*(1-dy)
                if pixel>=center:
                    er+=[1]
                else:
                    er+=[0]
                face_LBP[x,y]=min_value_of_binary2dec(er)
    return face_LBP



def autonorm1(mat):
    M = mat.max()
    m = mat.min()
    norm_mat = (mat-m)*1.0/(M-m)
    return norm_mat
    
def face_picture_import(s_path):
    count=0 #图像的数量
    faces_LBPs=[]
    for bmpfile in glob.glob(s_path):
        Im=plt.imread(bmpfile)
        img=cv2.equalizeHist(Im)  #直方图均衡化
        img_LBP=LBP(img,2,10)
        count+=1
        faces_LBPs.append(img_LBP)
    faces_LBPs=np.array(faces_LBPs)
    return faces_LBPs,count
    
female_LBP,female_count=face_picture_import('L:\\weishi\\face_recognization\\pythoncode\\galleryset_f\\*.bmp')#导入女性图像
male_LBP,male_count=face_picture_import('L:\\weishi\\face_recognization\\pythoncode\\galleryset_m\\*.bmp')#导入男性图像
    
faces_LBP=np.vstack((female_LBP,male_LBP))

    
def divide(img,a=35,b=70,PP=8):
    upper=img[a,:]
    middle=img[a:b,:]
    low=img[b:,:]
    upper_hist = cv2.calcHist([upper],[0],None,[2**PP],[0.0,2**PP-1]) 
    middle_hist = cv2.calcHist([middle],[0],None,[2**PP],[0.0,2**PP-1]) 
    low_hist = cv2.calcHist([low],[0],None,[2**PP],[0.0,2**PP-1]) 
    a,b=low_hist.shape
    upper_hist.shape=[b,a]
    middle_hist.shape=[b,a]
    low_hist.shape=[b,a]
    upper_hist=upper_hist[0]
    middle_hist=middle_hist[0]
    low_hist=low_hist[0]
    img_hist=np.hstack((upper_hist,middle_hist,low_hist))
    return img_hist
    
def svmclassify(x_train,y_train,x_test,y_test):
    #svc = svm.SVC(kernel='rbf',C=4,gamma=0.00125)
    svc = svm.SVC(kernel='linear')
    svc.fit(x_train,y_train)
    svc.predict(x_test)
    return svc.score(x_test,y_test)

def randomforestclassify(x_train,y_train,x_test,y_test):
    clf = RandomForestClassifier(random_state=1,n_estimators=10)
    clf = clf.fit(x_train,y_train)
    return clf.score(x_test,y_test)

count=faces_LBP.shape[0]

faces_hist=[]
for i in range(count):
    img=faces_LBP[i,:,:]
    img_hist=divide(img)
    faces_hist.append(img_hist)
faces_hist=np.array(faces_hist)

target=np.array([1]*female_count+[0]*male_count)

num_fold=10
def m():
    folds = KFold(faces_hist.shape[0],n_folds=num_fold,shuffle=True)
    rf_score=0
    for train_index, test_index in folds:
        X_train = faces_hist[train_index]
        X_test  = faces_hist[test_index]
        Y_train = target[train_index]
        Y_test = target[test_index]
        rf_single=svmclassify(X_train,Y_train,X_test,Y_test)
        #rf_single=randomforestclassify(X_train,Y_train,X_test,Y_test)
        rf_score +=rf_single
        print 'rf_single',rf_single
    return rf_score/num_fold


for i in range(male_count):
    img=male_LBP[i,:,:]
    plt.imsave('C:\\Users\\lifeng\\Desktop\\male\\female{}'.format(i),img,cmap=cm.gray)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    